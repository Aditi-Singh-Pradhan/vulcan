from fer.fer import FER
import cv2
import numpy as np

detector = FER(mtcnn=False)

def get_expression(roi_gray):
    # add padding around the crop so FER's internal detector works
    h, w = roi_gray.shape[:2]
    pad = int(max(h, w) * 0.3)
    padded = cv2.copyMakeBorder(roi_gray, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=0)
    roi_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    result = detector.detect_emotions(roi_bgr)
    if result:
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        if confidence < 0.3:
            return "Uncertain"
        return top_emotion.capitalize()
    return "Uncertain"
