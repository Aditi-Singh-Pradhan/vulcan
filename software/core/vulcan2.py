"""
Vulcan main controller.

Two processes:
  - Main process: Orbbec camera, YOLO, MediaPipe, FER, OpenCV window, Arduino serial.
  - Audio process: microphone, faster-whisper, RAG LLM, Piper TTS.

faster-whisper is ONLY imported inside the audio process; loading it in the
same process as OpenCV's Qt window freezes the window.

Shared between processes:
  - terminate_event      (mp.Event)   stop everything
  - conversation_flag    (mp.Value)   1 = conversation active (servo tracking on)
  - shared_state["expression"]        latest facial expression for the RAG prompt
  - arduino_queue        (mp.Queue)   audio process asks main to send serial commands
"""

import sys
import time
import queue
import subprocess
import platform
import multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "software" / "vision"))
sys.path.insert(0, str(PROJECT_ROOT / "software" / "utils"))
sys.path.insert(0, str(PROJECT_ROOT / "software" / "llm" / "rag"))

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ARDUINO_PORT = "COM8"          # change to /dev/ttyUSB0 or /dev/ttyACM0 on Linux
ARDUINO_BAUD = 115200

YOLO_MODEL_PATH = PROJECT_ROOT / "software" / "models" / "model.pt"
YOLO_CONFIDENCE = 0.5

SERVO_COOLDOWN = 0.15
EXPRESSION_COOLDOWN = 1.0
IDLE_CMD_6_INTERVAL = 50
IDLE_CMD_5_INTERVAL = 30

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 0.20       # room noise on your mic sits around 0.05-0.15
SILENCE_LIMIT = 2.0
MAX_RECORD_SECONDS = 15

# Words Whisper doesn't know. Listing them here makes it much more likely to
# hear "Vulcan" instead of "Wilkin"/"Elkin". Add names/nicknames as needed.
WHISPER_HINT = "Hello Vulcan. Vulcan is a robot built by ERC at BITS Pilani Goa."

PIPER_MODEL_PATH = Path("/home/aditi/piper-voices/en_US-lessac-medium.onnx")
PIPER_EXECUTABLE = "piper"
AUDIO_PLAYER = "aplay"


# =====================================================================
# AUDIO PROCESS — everything here runs in the child process only
# =====================================================================
def audio_process(terminate_event, conversation_flag, shared_state, arduino_queue):
    import shutil
    import time
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from llm.rag.app_lang2 import response_rag, warmup
    import llm.rag.app_lang2 as rag   # to read rag.LAST_TIMING

    print("[AUDIO] Loading Whisper...")
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    print("[AUDIO] Whisper ready.")

    # Load the RAG vector store + LLM now, so the first question isn't slow.
    warmup()

    def arduino(cmd):
        # Main process owns the serial port; just ask it to send.
        try:
            arduino_queue.put_nowait(cmd)
        except Exception:
            pass

    def speak(text):
        """Returns (synth_seconds, playback_seconds)."""
        synth = play = 0.0
        if not text:
            return synth, play
        print(f"[VULCAN] {text}")
        arduino(8)  # mouth start
        try:
            piper_path = shutil.which(PIPER_EXECUTABLE)
            if piper_path and PIPER_MODEL_PATH.exists() and platform.system() != "Windows":
                t0 = time.perf_counter()
                result = subprocess.run(
                    [piper_path, "--model", str(PIPER_MODEL_PATH),
                     "--output_file", "/tmp/vulcan_tts.wav"],
                    input=text, text=True, capture_output=True, timeout=60,
                )
                synth = time.perf_counter() - t0
                if result.returncode == 0:
                    t0 = time.perf_counter()
                    subprocess.run([AUDIO_PLAYER, "/tmp/vulcan_tts.wav"],
                                   capture_output=True, timeout=60)
                    play = time.perf_counter() - t0
                else:
                    print("[TTS] Piper failed:", result.stderr[-500:])
            else:
                print("[TTS] Piper or voice model not found.")
        except Exception as exc:
            print(f"[TTS] Error: {exc}")
        finally:
            arduino(9)  # mouth stop
        return synth, play

    timing = {}   # filled per turn, printed after Vulcan replies

    def listen():
        timing.clear()
        print("[AUDIO] Listening...")
        chunks = []
        silence_time = 0.0
        has_speech = False
        max_chunks = int(MAX_RECORD_SECONDS / CHUNK_DURATION)

        try:
            for _ in range(max_chunks):
                if terminate_event.is_set():
                    return None
                chunk = sd.rec(int(SAMPLE_RATE * CHUNK_DURATION),
                               samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                chunk = chunk.flatten()
                chunks.append(chunk)

                level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                if level > SILENCE_THRESHOLD:
                    has_speech = True
                    silence_time = 0.0
                elif has_speech:
                    silence_time += CHUNK_DURATION
                if has_speech and silence_time >= SILENCE_LIMIT:
                    break

            if not has_speech:
                return None

            print("[AUDIO] Transcribing...")
            t0 = time.perf_counter()
            segments, _ = whisper_model.transcribe(
                np.concatenate(chunks),
                language="en",
                initial_prompt=WHISPER_HINT,
            )
            # segments is lazy: the actual transcription happens in this join
            text = " ".join(s.text for s in segments).strip()
            timing["transcribe"] = time.perf_counter() - t0
            timing["silence_wait"] = SILENCE_LIMIT
            print(f"[AUDIO] Recognized: {text if text else '(nothing)'}")
            return text or None
        except Exception as exc:
            print(f"[AUDIO ERROR] {exc}")
            return None

    def print_timing():
        """One line showing where the time went for the last question."""
        order = [("silence_wait", "silence wait"), ("transcribe", "transcribe"),
                 ("retrieve", "retrieve"), ("llm", "LLM"), ("tts", "TTS")]
        parts = [f"{label} {timing[k]:.1f}s" for k, label in order if k in timing]
        wait = sum(timing.get(k, 0) for k, _ in order)
        print(f"[TIMING] {' | '.join(parts)}  =>  {wait:.1f}s until Vulcan starts "
              f"talking (then {timing.get('playback', 0):.1f}s of speech)")

    while not terminate_event.is_set():
        text = listen()
        if not text:
            continue
        lowered = text.lower()

        if "shut down" in lowered:
            speak("Abort sequence initiated. Shutting down.")
            terminate_event.set()
            break

        if "hello" in lowered and not conversation_flag.value:
            conversation_flag.value = 1
            speak("Hello! How can I assist you?")
            continue

        if "bye" in lowered:
            speak("Goodbye! See you soon!")
            conversation_flag.value = 0
            continue

        if conversation_flag.value:
            try:
                expression = shared_state.get("expression", "")
                response = response_rag(question=text, user_expression=expression)
                timing.update(rag.LAST_TIMING)
                synth, play = speak(str(response))
                timing["tts"] = synth
                timing["playback"] = play
                print_timing()
            except Exception as exc:
                print(f"[LLM ERROR] {exc}")
                speak("Sorry, I ran into a problem answering that.")
        else:
            speak("Say Hello Vulcan to start chatting!")

    print("[AUDIO] Stopped.")


# =====================================================================
# MAIN PROCESS — vision, window, Arduino
# =====================================================================
def run_vision(terminate_event, conversation_flag, shared_state, arduino_queue):
    import cv2
    import numpy as np
    import serial
    import mediapipe as mp_lib
    from ultralytics import YOLO
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    import vision.m_expression as exp_re

    # --- Arduino (only this process touches the serial port) ---
    try:
        arduino = serial.Serial(port=ARDUINO_PORT, baudrate=ARDUINO_BAUD, timeout=1)
        print(f"[ARDUINO] Connected on {ARDUINO_PORT}")
    except Exception as exc:
        arduino = None
        print(f"[ARDUINO] Not connected on {ARDUINO_PORT}: {exc}")

    def send(cmd):
        if arduino is None:
            return
        try:
            arduino.write(f"{cmd}\n".encode("utf-8"))
        except Exception as exc:
            print(f"[ARDUINO] Send failed: {exc}")

    # --- Models ---
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    face_model = YOLO(str(YOLO_MODEL_PATH))
    face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    # --- Camera ---
    def make_color_profile(pipeline):
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            return profiles.get_video_stream_profile(640, 480, OBFormat.MJPG, 30)
        except Exception:
            return profiles.get_default_video_stream_profile()

    depth_enabled = False
    try:
        pipeline = Pipeline()
        config = Config()
        config.enable_stream(make_color_profile(pipeline))
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception:
            pass
        pipeline.start(config)
        depth_enabled = True
        print("[CAMERA] Color + depth started.")
    except Exception as exc:
        print(f"[CAMERA] Depth failed ({exc}), trying color only...")
        pipeline = Pipeline()
        config = Config()
        config.enable_stream(make_color_profile(pipeline))
        pipeline.start(config)
        print("[CAMERA] Color-only started.")

    def decode(color_frame):
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        fmt = color_frame.get_format()
        w, h = color_frame.get_width(), color_frame.get_height()
        if fmt == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape(h, w, 3).copy()
        return None

    def depth_array(depth_frame):
        if depth_frame is None:
            return None
        try:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            return np.frombuffer(depth_frame.get_data(), dtype=np.uint16)[:w * h].reshape(h, w)
        except Exception:
            return None

    cv2.namedWindow("Vulcan", cv2.WINDOW_AUTOSIZE)
    print("[VISION] Window created.")

    last_expression = ""
    last_update_time = time.time()
    last_servo_time = time.time()
    last_cmd6_time = time.time()
    last_cmd5_time = time.time()
    prev_frame_time = time.time()
    prev_norm_x, prev_norm_y = 50, 50

    try:
        while not terminate_event.is_set():
            # Forward serial commands requested by the audio process (mouth sync).
            while True:
                try:
                    send(arduino_queue.get_nowait())
                except queue.Empty:
                    break

            try:
                frameset = pipeline.wait_for_frames(1000)
            except Exception:
                cv2.waitKey(1)
                continue
            if frameset is None:
                cv2.waitKey(1)
                continue

            color_frame = frameset.get_color_frame()
            if color_frame is None:
                cv2.waitKey(1)
                continue
            frame = decode(color_frame)
            if frame is None:
                cv2.waitKey(1)
                continue

            depth_data = depth_array(frameset.get_depth_frame()) if depth_enabled else None
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            if now - last_cmd6_time >= IDLE_CMD_6_INTERVAL:
                send(6)
                last_cmd6_time = now
            if now - last_cmd5_time >= IDLE_CMD_5_INTERVAL:
                send(5)
                last_cmd5_time = now

            try:
                results = face_model(frame, verbose=False)[0]
            except Exception:
                results = None

            if results is not None:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < YOLO_CONFIDENCE:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    face_cx, face_cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Depth at face center (median of 10x10 patch)
                    depth_mm = 0
                    if depth_data is not None and depth_data.size:
                        dy, dx = depth_data.shape
                        cx = min(max(face_cx, 5), dx - 5)
                        cy = min(max(face_cy, 5), dy - 5)
                        region = depth_data[cy - 5:cy + 5, cx - 5:cx + 5]
                        valid = region[region > 0]
                        if valid.size:
                            depth_mm = int(np.median(valid))

                    # Proportional servo tracking (only during conversation)
                    if conversation_flag.value and (now - last_servo_time > SERVO_COOLDOWN):
                        norm_x = int((face_cx / max(w, 1)) * 99)
                        norm_y = int((face_cy / max(h, 1)) * 99)
                        alpha = min(max(depth_mm / 3000.0, 0.2), 0.8) if 0 < depth_mm < 4000 else 0.5

                        smooth_x = max(0, min(99, int(prev_norm_x + alpha * (norm_x - prev_norm_x))))
                        smooth_y = max(0, min(99, int(prev_norm_y + alpha * (norm_y - prev_norm_y))))

                        if abs(smooth_x - prev_norm_x) > 2:
                            send(100 + smooth_x)
                            prev_norm_x = smooth_x
                        if abs(smooth_y - prev_norm_y) > 2:
                            send(200 + smooth_y)
                            prev_norm_y = smooth_y
                        last_servo_time = now

                    # MediaPipe landmarks
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size:
                        mesh = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        if mesh.multi_face_landmarks:
                            crop_h, crop_w = face_crop.shape[:2]
                            for lm in mesh.multi_face_landmarks[0].landmark:
                                px = min(max(int(lm.x * crop_w + x1), 0), w - 1)
                                py = min(max(int(lm.y * crop_h + y1), 0), h - 1)
                                cv2.circle(frame, (px, py), 1, (0, 255, 255), -1)

                    # Expression
                    expression = "unknown"
                    face_gray = gray[y1:y2, x1:x2]
                    if face_gray.size:
                        try:
                            expression = str(exp_re.get_expression(face_gray))
                        except Exception:
                            pass
                    if expression != last_expression or now - last_update_time >= EXPRESSION_COOLDOWN:
                        last_expression = expression
                        shared_state["expression"] = expression
                        last_update_time = now

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    depth_text = f"{depth_mm / 1000:.1f}m" if depth_mm > 0 else "no depth"
                    cv2.putText(frame, f"{expression} ({conf:.2f}) {depth_text}",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    break  # first face only

            fps = 1.0 / max(now - prev_frame_time, 1e-6)
            prev_frame_time = now
            status = "TALKING" if conversation_flag.value else "IDLE"
            cv2.putText(frame, f"FPS: {fps:.1f}  {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Vulcan", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                terminate_event.set()
                break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        face_mesh.close()
        if arduino is not None:
            arduino.close()
        print("[VISION] Stopped.")


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    shared_state = manager.dict(expression="")
    terminate_event = mp.Event()
    conversation_flag = mp.Value("i", 0)
    arduino_queue = mp.Queue()

    audio = mp.Process(
        target=audio_process,
        args=(terminate_event, conversation_flag, shared_state, arduino_queue),
        daemon=True,
    )
    audio.start()

    try:
        run_vision(terminate_event, conversation_flag, shared_state, arduino_queue)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_event.set()
        audio.join(timeout=5)
        if audio.is_alive():
            audio.terminate()
        print("[MAIN] Vulcan shut down.")