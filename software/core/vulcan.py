import sys
from pathlib import Path

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "software" / "vision"))
sys.path.insert(0, str(PROJECT_ROOT / "software" / "utils"))
sys.path.insert(0, str(PROJECT_ROOT / "software" / "llm" / "rag"))

import multiprocessing.shared_memory
import cv2
import numpy as np
from PIL import Image
import concurrent.futures
import struct
import time
import subprocess
import multiprocessing
import vision.m_expression as exp_re
import sounddevice as sd
from faster_whisper import WhisperModel
from threading import Event, Lock
from llm.rag.app_lang2 import response_rag
import serial
import mediapipe as mp
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


# --- Config ---
ARDUINO_PORT = 'COM8'
ARDUINO_BAUD = 115200
YOLO_MODEL_PATH = str(PROJECT_ROOT / "software" / "models" / "model.pt")
YOLO_CONFIDENCE = 0.5
SERVO_COOLDOWN = 0.15          # seconds between servo commands
EXPRESSION_COOLDOWN = 1.0      # seconds between expression updates
IDLE_CMD_6_INTERVAL = 50       # seconds
IDLE_CMD_5_INTERVAL = 30       # seconds
TOLERANCE_X_RATIO = 0.08
TOLERANCE_Y_RATIO = 0.08

# --- Arduino setup ---
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=ARDUINO_BAUD, timeout=1)
    ARDUINO_CONNECTED = True
except Exception as e:
    print(f"Arduino not connected: {e}")
    ARDUINO_CONNECTED = False

# --- YOLO face detection ---
face_model = YOLO(YOLO_MODEL_PATH)

# --- MediaPipe face mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- Depth camera setup ---
ob_pipeline = Pipeline()
ob_config = Config()

color_profiles = ob_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.MJPG, 30)
ob_config.enable_stream(color_profile)

depth_profiles = ob_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = depth_profiles.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
ob_config.enable_stream(depth_profile)

ob_config.set_align_mode(OBAlignMode.HW_MODE)


# --- Shared state ---
conversation_started = False
terminate_event = Event()
expression_lock = Lock()
glob_expression = ""
glob_landmarks = None

# --- Shared memory ---
shm = multiprocessing.shared_memory.SharedMemory(create=True, size=250)


def send_command_to_arduino(command):
    """Send a command to Arduino."""
    if not ARDUINO_CONNECTED:
        return
    try:
        arduino.write(f"{command}\n".encode('utf-8'))
    except Exception as e:
        print(f"Failed to send command to Arduino: {e}")


def speak(text):
    """TTS with Piper + Arduino mouth sync."""
    send_command_to_arduino(8)
    subprocess.run(
        ["piper", "--model", "/home/aditi/piper-voices/en_US-lessac-medium.onnx",
         "--output_file", "/tmp/vulcan_tts.wav"],
        input=text, text=True, capture_output=True
    )
    subprocess.run(["aplay", "/tmp/vulcan_tts.wav"], capture_output=True)
    send_command_to_arduino(9)


whisper_model = WhisperModel("small", device="cpu", compute_type="int8")


def listen():
    """Record until 2s of silence, transcribe with Whisper."""
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.5
    SILENCE_THRESHOLD = 0.01
    SILENCE_LIMIT = 2.0

    print("Please speak:")
    chunks = []
    silence_time = 0
    has_speech = False

    while True:
        chunk = sd.rec(int(SAMPLE_RATE * CHUNK_DURATION),
                       samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        chunk = chunk.flatten()
        chunks.append(chunk)

        level = np.abs(chunk).mean()
        if level > SILENCE_THRESHOLD:
            has_speech = True
            silence_time = 0
        else:
            silence_time += CHUNK_DURATION

        if has_speech and silence_time >= SILENCE_LIMIT:
            break

    audio = np.concatenate(chunks)
    segments, _ = whisper_model.transcribe(audio, language="en")
    text = " ".join([s.text for s in segments]).strip()
    return text if text else None


def conversation_code():
    global conversation_started, glob_expression

    while not terminate_event.is_set():
        text = listen()
        if not text:
            continue

        print("You said:", text)

        if "hello" in text.lower() and not conversation_started:
            conversation_started = True
            speak("Hello! How can I assist you?")
            continue

        if "bye" in text.lower():
            speak("Goodbye! See You Soon!")
            conversation_started = False
            continue

        if "shut down" in text.lower():
            speak("Abort sequence initiated. Shutting down!")
            terminate_event.set()
            break

        if conversation_started:
            try:
                with expression_lock:
                    current_expression = glob_expression
                response = response_rag(question=text, user_expression=current_expression)
                speak(response)
            except Exception as e:
                print(f"Error: {e}")
        else:
            speak("Greet me with Hello Vulcan to start chatting!")


def vision():
    print("Vision Started.")
    ob_pipeline.start(ob_config)
    
    cv2.startWindowThread()
    cv2.namedWindow('Vulcan', cv2.WINDOW_AUTOSIZE)
    """YOLO face detection + MediaPipe landmarks + FER expression + depth tracking."""
    global conversation_started, glob_expression, glob_landmarks

    frame_width = 640
    frame_height = 480
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    tolerance_x = frame_width * TOLERANCE_X_RATIO
    tolerance_y = frame_height * TOLERANCE_Y_RATIO

    last_expression = ""
    last_update_time = time.time()
    last_servo_time = time.time()
    last_cmd6_time = time.time()
    last_cmd5_time = time.time()
    prev_frame_time = time.time()
    prev_norm_x = 50
    prev_norm_y = 50

    while not terminate_event.is_set():
        frameset = ob_pipeline.wait_for_frames(100)
        if frameset is None:
            print("no frameset")
            continue

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if color_frame is None:
            print("no color frame")
            continue

        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        frame = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"decode failed, data size: {len(color_data)}")
            continue

        print(f"frame: {frame.shape}")

        # Get depth data if available
        depth_data = None
        if depth_frame is not None:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        # --- Idle Arduino commands ---
        if current_time - last_cmd6_time >= IDLE_CMD_6_INTERVAL:
            send_command_to_arduino(6)
            last_cmd6_time = current_time
        if current_time - last_cmd5_time >= IDLE_CMD_5_INTERVAL:
            send_command_to_arduino(5)
            last_cmd5_time = current_time

        # --- YOLO face detection ---
        results = face_model(frame, verbose=False)[0]

        for box in results.boxes:
            if box.conf[0] < YOLO_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2

            # --- Get depth at face center ---
            depth_mm = 0
            if depth_data is not None:
                dy, dx = depth_data.shape
                cx = min(max(face_center_x, 5), dx - 5)
                cy = min(max(face_center_y, 5), dy - 5)
                depth_region = depth_data[cy-5:cy+5, cx-5:cx+5]
                valid_depths = depth_region[depth_region > 0]
                if len(valid_depths) > 0:
                    depth_mm = int(np.median(valid_depths))

            # --- Proportional servo tracking with depth ---
            if conversation_started and (current_time - last_servo_time > SERVO_COOLDOWN):
                offset_x = face_center_x - frame_center_x
                offset_y = face_center_y - frame_center_y

                norm_x = int(((offset_x / frame_width) + 0.5) * 99)
                norm_y = int(((offset_y / frame_height) + 0.5) * 99)

                # Depth-based smoothing
                if depth_mm > 0 and depth_mm < 4000:
                    depth_m = depth_mm / 1000.0
                    alpha = min(max(depth_m / 3.0, 0.2), 0.8)
                else:
                    alpha = 0.5  # fallback if no depth

                smooth_x = int(prev_norm_x + alpha * (norm_x - prev_norm_x))
                smooth_y = int(prev_norm_y + alpha * (norm_y - prev_norm_y))
                smooth_x = max(0, min(99, smooth_x))
                smooth_y = max(0, min(99, smooth_y))

                if abs(smooth_x - prev_norm_x) > 2:
                    send_command_to_arduino(100 + smooth_x)
                    prev_norm_x = smooth_x
                if abs(smooth_y - prev_norm_y) > 2:
                    send_command_to_arduino(200 + smooth_y)
                    prev_norm_y = smooth_y

                last_servo_time = current_time

            # --- MediaPipe landmarks ---
            face_crop = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(face_rgb)

            landmarks = None
            if mesh_results.multi_face_landmarks:
                face_lm = mesh_results.multi_face_landmarks[0]
                crop_h, crop_w = face_crop.shape[:2]
                landmarks = np.array(
                    [[lm.x * crop_w + x1, lm.y * crop_h + y1, lm.z]
                     for lm in face_lm.landmark],
                    dtype=np.float32
                )
                for pt in landmarks:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

            # --- FER expression ---
            face_gray = gray[y1:y2, x1:x2]
            if face_gray.size == 0:
                continue

            expression_result = exp_re.get_expression(face_gray)

            if expression_result != last_expression or (current_time - last_update_time) >= EXPRESSION_COOLDOWN:
                last_expression = expression_result
                with expression_lock:
                    glob_expression = expression_result
                    glob_landmarks = landmarks
                last_update_time = current_time

            # --- Draw ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            depth_text = f"{depth_mm/1000:.1f}m" if depth_mm > 0 else "no depth"
            cv2.putText(frame, f"{expression_result} ({conf:.2f}) {depth_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            break  # first face only

        # FPS counter
        fps = 1 / (current_time - prev_frame_time) if current_time != prev_frame_time else 0
        prev_frame_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Vulcan', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            terminate_event.set()
            break

    ob_pipeline.stop()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    import threading
    
    conv_thread = threading.Thread(target=conversation_code, daemon=True)
    conv_thread.start()
    
    try:
        vision()  # runs on main thread
    except KeyboardInterrupt:
        terminate_event.set()
    finally:
        ob_pipeline.stop()
        cv2.destroyAllWindows()
        shm.unlink()
        if ARDUINO_CONNECTED:
            arduino.close()