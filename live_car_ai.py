
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DETECTOR_PATH = BASE_DIR / "yolov8n.pt"
CLASSIFIER_PATH = (
    BASE_DIR / "runs" / "brand_classifier" / "weights" / "best.pt"
)

CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_DSHOW

DEVICE = 0 if torch.cuda.is_available() else "cpu"

DETECTION_CONFIDENCE = 0.35
BRAND_CONFIDENCE = 0.40


# ==========================================
# 2. LOAD AI MODELS
# ==========================================

print("Loading AI models...")

if not DETECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Vehicle detector not found: {DETECTOR_PATH}"
    )

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(
        f"Brand classifier not found: {CLASSIFIER_PATH}"
    )

detector = YOLO(str(DETECTOR_PATH))
classifier = YOLO(str(CLASSIFIER_PATH))

print("AI models loaded successfully!")
print(f"Running on: {DEVICE}")


# ==========================================
# 3. CAR COLOR RECOGNITION
# ==========================================

def detect_color(image):
    h, w = image.shape[:2]

    if h < 10 or w < 10:
        return "unknown"

    # Select central region of detected vehicle.
    roi = image[
        int(h * 0.35):int(h * 0.75),
        int(w * 0.15):int(w * 0.85)
    ]

    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape(-1, 3)

    H, S, V = np.median(pixels, axis=0)

    # Black
    if V < 65:
        return "black"

    # White and grey (including silver)
    if S < 35:
        if V > 190:
            return "white"
        return "grey"

    # Red
    if H < 10 or H >= 170:
        return "red"

    # Brown, beige, gold
    if H < 20:
        if V < 130 or S < 100:
            return "brown"
        return "yellow"

    # Yellow and orange
    if H < 35:
        if S < 100:
            return "brown"
        return "yellow"

    # Green
    if H < 85:
        return "green"

    # Blue
    if H < 135:
        return "blue"

    return "unknown"


# ==========================================
# 4. INITIALIZE CAMERA
# ==========================================

print("Connecting to camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    CAMERA_BACKEND
)

if not cap.isOpened():
    raise RuntimeError(
        "Camera unavailable. Close other camera apps "
        "and check Windows camera permissions."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera connected successfully!")
print("LIVE CAR AI STARTED")
print("Press Q to quit.")


# ==========================================
# 5. REAL-TIME VEHICLE RECOGNITION
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame.")
            break

        # Detect vehicles using YOLOv8.
        results = detector.predict(
            source=frame,
            classes=[2, 5, 7],
            conf=DETECTION_CONFIDENCE,
            imgsz=640,
            device=DEVICE,
            verbose=False
        )[0]

        detected_count = 0

        for box in results.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep bounding box inside frame.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # ==================================
            # 6. BRAND CLASSIFICATION
            # ==================================

            prediction = classifier.predict(
                source=crop,
                imgsz=224,
                device=DEVICE,
                verbose=False
            )[0]

            idx = prediction.probs.top1

            confidence = float(
                prediction.probs.top1conf.item()
            )

            if confidence < BRAND_CONFIDENCE:
                brand = "UNKNOWN"
            else:
                brand = prediction.names[idx].upper()

            # ==================================
            # 7. COLOR RECOGNITION
            # ==================================

            color = detect_color(crop).upper()

            detected_count += 1

            # ==================================
            # 8. DISPLAY RESULTS
            # ==================================

            label = (
                f"{brand} | {color} | "
                f"{confidence:.0%}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # Display vehicle count.
        cv2.putText(
            frame,
            f"Vehicles detected: {detected_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "CAR BRAND AND COLOR AI",
            frame
        )

        # Press Q to exit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # ==========================================
    # 9. CLEANUP
    # ==========================================

    cap.release()
    cv2.destroyAllWindows()

    print("CAR AI STOPPED")
