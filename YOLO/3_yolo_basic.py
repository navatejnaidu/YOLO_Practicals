
import cv2
from ultralytics import YOLO

# Load the standard YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("YOLOv8 Object Detection Started!")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects
    results = model(frame)

    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

    # Display webcam feed
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
