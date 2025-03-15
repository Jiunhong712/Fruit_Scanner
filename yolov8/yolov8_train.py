import torch
from ultralytics import YOLO
import os
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr0 = trial.suggest_float("lr0", 0.001, 0.02, log=True)
    epochs = trial.suggest_int("epochs", 50, 150)  # limited epochs for tuning
    imgsz = trial.suggest_categorical("imgsz", [416, 512, 640])
    
    # Define dataset and output paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_config = os.path.join(script_dir, "dataset", "data.yaml")
    project_dir = os.path.join(script_dir, "hyperparam_trials")
    os.makedirs(project_dir, exist_ok=True)
    
    # Specify the classes to be used
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]
    
    # Create and set up the YOLO model instance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n.pt")
    model.to(device)
    
    trial_name = f"trial_{trial.number}"
    
    # Train the model using suggested hyperparameters
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        patience=5,
        project=project_dir,
        name=trial_name,
        save=True,
        classes=selected_classes,
        workers=8,
        cache=False,
        augment=True,
        verbose=False,
        lr0=lr0
    )
    
    # Evaluate on the validation set
    results = model.val(
        data=data_config,
        imgsz=imgsz,
        device=device,
        classes=selected_classes,
        verbose=False,
    )
    
    mAP50 = results.box.map50
    print(f"Trial {trial.number}: mAP50 = {mAP50:.4f} with lr0={lr0}, epochs={epochs}, imgsz={imgsz}")
    return mAP50

def main():
    # Create an Optuna study to maximize mAP50
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Increase n_trials for a thorough search
    
    print("Best hyperparameters found:")
    print(study.best_trial.params)
    
    # Retrieve best parameters
    best_params = study.best_trial.params
    
    # Define paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_config = os.path.join(script_dir, "dataset", "data.yaml")
    final_project_dir = os.path.join(script_dir, "final_model")
    os.makedirs(final_project_dir, exist_ok=True)
    
    # Specify the classes to be used
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]
    
    # Create a YOLO model instance for final training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_model = YOLO("yolov8n.pt")
    final_model.to(device)
    
    # Retrain the model on the full dataset with the best hyperparameters
    # You can increase epochs for the final training schedule (e.g., 300 epochs)
    final_epochs = 300
    final_imgsz = best_params.get("imgsz", 512)
    final_lr0 = best_params.get("lr0", 0.01)
    
    print("Starting final training with best hyperparameters...")
    final_model.train(
        data=data_config,
        epochs=final_epochs,
        imgsz=final_imgsz,
        device=device,
        patience=5,
        project=final_project_dir,
        name='fruits_detector_final',
        save=True,
        classes=selected_classes,
        workers=8,
        cache=False,
        augment=True,
        verbose=True,
        lr0=final_lr0
    )
    print("Final training complete!")
    
    # Evaluate the final model on the test dataset
    print("Evaluating final model on test dataset...")
    final_results = final_model.val(
        data=data_config,
        imgsz=final_imgsz,
        device=device,
        classes=selected_classes,
        verbose=True,
    )
    
    print("\nFinal Evaluation Metrics:")
    print(f"mAP50: {final_results.box.map50:.4f}")
    print(f"mAP50-95: {final_results.box.map:.4f}")
    print(f"Mean Precision: {final_results.box.mp:.4f}")
    print(f"Mean Recall: {final_results.box.mr:.4f}")
    print(f"Mean F1-Score: {final_results.box.f1.mean():.4f}")
    print(f"Speed: {final_results.speed['inference']:.2f}ms")

if __name__ == "__main__":
    main()
