# 🍎 Fruit Scanner

A computer vision-based fruit recognition system built using **YOLOv8**. The model was integrated into Raspberry Pi 5 to enable real-time fruit detection and classification.

---

## 🧠 Project Description

The **Fruit Scanner** uses **YOLOv8**, a state-of-the-art object detection model, to detect and classify different types of fruits from images. The system is designed for real-time performance and high accuracy, making it suitable for automation tasks in agriculture, retail, and food tech.

The project initially started with MobileNetV2, but after comparative testing, it was replaced by YOLOv8 due to a higher accuracy and faster inference speed compared to MobileNetV2.


---

## 🧰 Tech Stack

### 👨‍💻 Languages
- Python 3.10

### ⚙️ Frameworks & Libraries (YOLOv8)
- Ultralytics YOLOv8 – Load and run YOLOv8 model for object detection and classification
- torch – PyTorch library for deep learning
- os – Interacts with the operating system for file path management
- shutil – High-level file operations such as copying models and managing directories

### ⚙️ Frameworks & Libraries (MobileNetV2)
- torch – PyTorch library for deep learning
- numpy - Provides support for numerical data manipulation
- tensorflow - Deep learning framework used for building and training models
- matplotlib - Data visualization
- scikit-learn - Model performance evaluation
- keras-tuner - Hyperparameter tuning

---

## 🗂️ Dataset

- **Source:** Roboflow, Kaggle
- **Total Images:** 11000
- **Fruits:** Apple, Banana, Orange, Mango
- **Preprocessing:** Image resizing, augmentation, and YOLO-format annotation
- **Split:** 80% training / 20% testing
