import torch
from ultralytics import YOLO
import os
import shutil

def main():
    # Check and print CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B"  # Directory to save model

    # Define dataset path (relative to script location)
    data_config = os.path.join(script_dir, "dataset", "data.yaml")

    # Check if dataset YAML file exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset YAML not found at: {data_config}")

    print(f"Using dataset config: {data_config}")
    
    # Specify the classes to be used (Modify this list as needed)
    selected_classes = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23]

    # Create YOLO model instance
    model = YOLO("yolov8n.pt")
    model.to(device)

    # Start training with filtered classes
    print("Starting training...")
    model.train(
        data=data_config,
        epochs=1,
        imgsz=512,
        device=device,
        patience=5,
        project=save_dir,
        name='fruits_detector',
        save=True,  # Save best model
        classes=selected_classes,  # Filter unwanted classes
        workers=8,
        cache=False
    )
    print("Training complete!")

    # Save best model
    pt_save_dir = os.path.join(save_dir, 'fruits_detector', 'weights')
    os.makedirs(pt_save_dir, exist_ok=True)
    pt_path = os.path.join(pt_save_dir, 'best.pt')
    
    runs_best_pt = os.path.join(save_dir, 'fruits_detector', 'weights', 'best.pt')
    if os.path.exists(runs_best_pt):
        shutil.copy(runs_best_pt, pt_path)
        print(f"PyTorch model saved to {pt_path}")

    # Export model to ONNX format
    onnx_path = os.path.join(pt_save_dir, 'best.onnx')
    model.export(format='onnx', save_dir=pt_save_dir)
    print(f"ONNX model exported to {onnx_path}")

    # Evaluate model on test dataset with filtered classes
    print("Evaluating model on test dataset...")
    results = model.val(
        data=data_config,
        imgsz=512,
        device=device,
        classes=selected_classes  # Ensure only selected classes are evaluated
    )

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Mean Precision: {results.box.mp:.4f}")
    print(f"Mean Recall: {results.box.mr:.4f}")
    print(f"Mean F1-Score: {results.box.f1.mean():.4f}")
    print(f"Speed: {results.speed['inference']:.2f}ms")

if __name__ == "__main__":
    main()
