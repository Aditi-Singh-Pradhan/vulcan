import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh                     #mediapipe face mesh: mesh is a set of points that represent the shape of a face
mp_face_detection = mp.solutions.face_detection           #mediapipe face detection: it works by detecting the presence of a face in an image and returning the coordinates of the bounding box around the face

face_mesh = mp_face_mesh.FaceMesh(                        # Initialize the face mesh model
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,                                # enables 478 landmarks (includes iris)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,              # 0 = short range (<2m), 1 = long range (<5m)
    min_detection_confidence=0.5
)


def get_face(frame):
    """
    Detect a face using MediaPipe Face Detection.
    Returns:
        bbox: list of [x1, y1, x2, y2] for each detected face
        gray: grayscale version of the frame (for expression model)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detector.process(frame_rgb)

    bbox = []
    if results.detections:
        for detection in results.detections:
            bb = detection.location_data.relative_bounding_box

            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)

            # expand by 20%
            bw = x2 - x1
            bh = y2 - y1
            x1 -= int(bw * 0.2)
            y1 -= int(bh * 0.2)
            x2 += int(bw * 0.2)
            y2 += int(bh * 0.2)

            # clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bbox.append([x1, y1, x2, y2])

    return bbox, gray


def get_landmarks(frame):
    """
    Run MediaPipe Face Mesh on a frame.
    Returns:
        landmarks: numpy array of shape (478, 3) with x, y, z coords
                   normalized to pixel coordinates (x, y) and raw z.
                   Returns None if no face detected.
    """
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]
    landmarks = np.array(
        [[lm.x * w, lm.y * h, lm.z] for lm in face.landmark],
        dtype=np.float32
    )
    return landmarks


def get_face_and_landmarks(frame):
    """
    Single call that returns both bbox and landmarks.
    More efficient than calling get_face + get_landmarks separately
    since it only converts to RGB once.
    Returns:
        bbox: list of [x1, y1, x2, y2] or empty list
        gray: grayscale frame
        landmarks: numpy array (478, 3) or None
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detection 
    det_results = face_detector.process(frame_rgb)
    bbox = []
    if det_results.detections:
        for detection in det_results.detections:
            bb = detection.location_data.relative_bounding_box

            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)

            bw = x2 - x1
            bh = y2 - y1
            x1 -= int(bw * 0.2)
            y1 -= int(bh * 0.2)
            x2 += int(bw * 0.2)
            y2 += int(bh * 0.2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bbox.append([x1, y1, x2, y2])

    # landmarks: landmarks are points on the face that can be used to track facial expressions, head pose, and gaze direction. 
    # The landmarks are returned as a numpy array of shape (478, 3) where each row corresponds to a landmark and the columns correspond to x, y, z coordinates. 
    mesh_results = face_mesh.process(frame_rgb)
    landmarks = None
    if mesh_results.multi_face_landmarks:
        face = mesh_results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z] for lm in face.landmark],
            dtype=np.float32
        )

    return bbox, gray, landmarks


def close():
    """Release MediaPipe resources."""
    face_mesh.close()
    face_detector.close()