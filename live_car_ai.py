import cv2
import numpy as np
from ultralytics import YOLO

detector = YOLO("yolov8n.pt")
classifier = YOLO(r"D:\700\prjct\runs\brand_classifier\weights\best.pt")

def detect_color(image):
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return "unknown"

    # Sample the central body region, avoiding some background.
    roi = image[int(h*0.35):int(h*0.75), int(w*0.15):int(w*0.85)]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    H, S, V = np.median(pixels, axis=0)

    if V < 65:
        return "black"
    if S < 35:
        if V > 190:
            return "white"
        if V > 125:
            return "silver"
        return "gray"
    if 5 <= H < 20:
        return "brown" if V < 130 else "beige"
    if H < 10 or H >= 170:
        return "red"
    if H < 35:
        return "beige"
    if H < 85:
        return "green"
    if H < 135:
        return "blue"
    return "unknown"

cap = cv2.VideoCapture(1, cv2.CAP_MSMF); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Camera unavailable. Check Windows camera permissions.")

print("LIVE CAR AI STARTED. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.predict(frame, classes=[2, 5, 7], conf=0.35,
                               imgsz=640, device=0, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        prediction = classifier.predict(crop, imgsz=224,
                                        device=0, verbose=False)[0]

        idx = prediction.probs.top1
        brand = prediction.names[idx].upper()
        confidence = prediction.probs.top1conf.item()
        color = detect_color(crop).upper()

        label = f"{brand} | {color} | {confidence:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(25, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2)

    cv2.imshow("CAR BRAND AND COLOR AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
