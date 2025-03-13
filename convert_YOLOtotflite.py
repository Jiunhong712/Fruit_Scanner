import os
from ultralytics import YOLO

# Define the model path and export directory
model_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\fruits_detector10\weights\best.pt"
export_dir = os.path.dirname(model_path)  # Save in the same folder

# Load the YOLO model
model = YOLO(model_path)

# Export the model to TFLite format
model.export(format="tflite", save_dir=export_dir)

print(f"TFLite model successfully saved in: {export_dir}")
