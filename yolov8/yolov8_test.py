import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\fruits_detector7\weights\best.pt" # Update with your actual path
model = YOLO(model_path)  # Load YOLOv8 model

# Initialize webcam/
cap = cv2.VideoCapture(0)  # 0 for the default laptop camera

# Set frame size
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Draw detection boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class index
            label = f"{model.names[class_id]}: {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
