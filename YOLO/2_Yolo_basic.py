from ultralytics import YOLO
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Predict on an image
detection_output = model.predict(
    source=r"C:\Users\polin\Desktop\GenAI_AgenticAI_Prac\Practicals\YOLO\cycle.jpg",
    conf=0.25,
    save=True
)

# Display detection results
print(detection_output)