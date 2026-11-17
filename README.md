# Car Brand and Color Recognition AI

Detects vehicles using YOLOv8, identifies brands using a custom-trained classifier, and estimates colors using OpenCV.

## Installation
Run: pip install -r requirements.txt

## Usage
Run: python live_car_ai.py
Press Q to quit.

## Models
- yolov8n.pt: Vehicle detection
- runs/brand_classifier/weights/best.pt: Brand classification

## Hardware
Tested on NVIDIA RTX 4060 Laptop GPU.

## Limitations
Accuracy depends on lighting, camera quality, and training data.
