# Common imports
import os
import numpy as np
import tensorflow as tf
from typing import Tuple

# Data imports
from tqdm import tqdm
from glob import glob
from tensorflow import image as tfi

# Data visualization
import matplotlib.pyplot as plt

# Model architecture
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# Model training
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Model hypertunning
import keras_tuner as kt

# Constants
IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# Directory paths
train_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\Dataset3\train" 
valid_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\Dataset3\valid" 
test_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\Dataset3\test" 

# Model training
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

# Random seed for reproducibility
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# Collect the class names
class_names = sorted(os.listdir(train_dir))
n_classes = len(class_names)
print(f"Total number of classes: {n_classes}")
print(f"Classes: {class_names}")


def load_image(image_path: str) -> tf.Tensor:
    # Check if image path exists
    assert os.path.exists(image_path), f'Invalid image path: {image_path}'

    # Read the image file
    image = tf.io.read_file(image_path)

    # Load the image
    try:
        image = tfi.decode_jpeg(image, channels=3)
    except:
        image = tfi.decode_png(image, channels=3)

    # Change the image data type
    image = tfi.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tfi.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Convert image data type to tf.float32
    image = tf.cast(image, tf.float32)

    return image


def load_dataset(root_path: str, class_names: list, trim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if trim:
        # Trim the size of the data
        n_samples = len(class_names) * trim
    else:
        # Collect total number of data samples
        n_samples = sum([len(os.listdir(os.path.join(root_path, name))) for name in class_names])

    # Create arrays to store images and labels
    images = np.empty(shape=(n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty(shape=(n_samples, 1), dtype=np.int32)

    # Loop over all the image file paths, load and store the images with respective labels
    n_image = 0
    for class_name in tqdm(class_names, desc="Loading"):
        class_path = os.path.join(root_path, class_name)
        image_paths = list(glob(os.path.join(class_path, "*")))[:trim]
        for file_path in image_paths:
            # Load the image
            image = load_image(file_path)

            # Assign label
            label = class_names.index(class_name)

            # Store the image and the respective label
            images[n_image] = image
            labels[n_image] = label

            # Increment the number of images processed
            n_image += 1

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]

    return images, labels


# Load the training dataset
X_train, y_train = load_dataset(root_path=train_dir, class_names=class_names, trim=1000)  # 1000 images per class

# # Load the validation dataset
X_valid, y_valid = load_dataset(root_path=valid_dir, class_names=class_names)

# Load the testing dataset
X_test, y_test = load_dataset(root_path=test_dir, class_names=class_names)


def show_images(images: np.ndarray, labels: np.ndarray, n_rows: int = 1, n_cols: int = 5, figsize: tuple = (25, 8),
                model: tf.keras.Model = None) -> None:
    # Loop over each row of the plot
    for row in range(n_rows):
        # Create a new figure for each row
        plt.figure(figsize=figsize)

        # Generate a random index for each column in the row
        rand_indices = np.random.choice(len(images), size=n_cols, replace=False)

        # Loop over each column of the plot
        for col, index in enumerate(rand_indices):
            # Get the image and label at the random index
            image = images[index]
            label = class_names[int(labels[index])]

            # If a model is provided, make a prediction on the image
            if model:
                prediction = model.predict(np.expand_dims(tf.squeeze(image), axis=0), verbose=0)[0]
                conf = np.max(prediction, axis=-1)
                label += "\nPrediction: {} - {:.4}".format(class_names[np.argmax(prediction)], conf)

            # Plot the image and label
            plt.subplot(1, n_cols, col + 1)
            plt.imshow(image)
            plt.title(label.title())
            plt.axis("off")

        # Show the row of images
        plt.show()


# Visualize Training Dataset
# show_images(images=X_train, labels=y_train, n_rows=5)

# Mobilenet Backbone
print("Loading MobileNet Backbone: ")
mobilenet = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', include_top=False)

# Freeze the model weights
mobilenet.trainable = False

# The Mobilenet Model baseline
mobilenet = Sequential([
    mobilenet,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# Compile the Baseline
mobilenet.compile(
    loss=LOSS,
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=METRICS
)

# Train the Xception Baseline Model
print("\nTraining Baseline Model: ")
mobilenet.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("MobileNetV2Baseline.h5", save_best_only=True)
    ],
    batch_size=BATCH_SIZE
)

# Save the final model
mobilenet.save('IDP_MobilenetV2_model.h5')
print("Model saved successfully!")

# Testing Evaluation
mobilenet_test_loss, mobilenet_test_acc = mobilenet.evaluate(X_test, y_test)
print(f"\nMobileNet Baseline Testing Loss     : {mobilenet_test_loss}.")
print(f"MobileNet Baseline Testing Accuracy : {mobilenet_test_acc}.")


def build_model(hp):
    # Define all hyperparameters
    mobilenet = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', include_top=False)
    mobilenet.trainable = False
    n_layers = hp.Choice('n_layers', [2, 4])
    dropout_rate = hp.Choice('rate', [0.4, 0.7])
    n_units = hp.Choice('units', [256, 512])
    learning_rate = hp.Choice('lr', [LEARNING_RATE, LEARNING_RATE * 0.1, LEARNING_RATE * 0.01])

    # Mode architecture
    model = Sequential([
        mobilenet,
        GlobalAveragePooling2D(),
    ])

    # Add hidden/top layers
    for _ in range(n_layers):
        model.add(Dense(n_units, activation='relu', kernel_initializer='he_normal'))

    # Add Dropout Layer
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss=LOSS,
        optimizer=Adam(learning_rate),
        metrics=METRICS
    )

    # Return model
    return model


# Initialize Random Searcher
random_searcher = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=10,
    seed=42,
    project_name="MobileNetSearch",
    loss=LOSS)

# Start Searching
search = random_searcher.search(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=10,
    batch_size=BATCH_SIZE
)

# Best hyper parameters
best_hps = random_searcher.get_best_hyperparameters()[0]
print(f"Best Hyper Parameters founded: {best_hps.values}\n")

# Build the best model
mobile_net_model = build_model(best_hps)
mobile_net_model.summary()

# Compile the model
print("\nTraining Best Model Architecture : ")
mobile_net_model_history = mobile_net_model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('BestMobileNet.h5', save_best_only=True)
    ]
)
test_loss, test_acc = mobile_net_model.evaluate(X_test, y_test)
print("Training Loss    : {:.4} | Baseline : {:.4}".format(test_loss, mobilenet_test_loss))
print("Training Accuracy: {:.4}% | Baseline : {:.4}%".format(test_acc * 100, mobilenet_test_acc * 100))
# Make predictions
baseline_pred = np.argmax(mobilenet.predict(X_test, verbose=0), axis=-1)
best_pred = np.argmax(mobile_net_model.predict(X_test, verbose=0), axis=-1)

# Evaluate prediction : Precision
baseline_pre = precision_score(y_test, baseline_pred, average='macro')
best_pre = precision_score(y_test, best_pred, average='macro')

# Evaluate prediction : Recall
baseline_recall = recall_score(y_test, baseline_pred, average='macro')
best_recall = recall_score(y_test, best_pred, average='macro')

# Evaluate prediction : F1 Score
baseline_f1 = f1_score(y_test, baseline_pred, average='macro')
best_f1 = f1_score(y_test, best_pred, average='macro')

print("{:20} | {:20}".format("Baseline Performance", "Best Performance\n"))
print("{:10} : {:.5}  | {:.5}".format("Precision", baseline_pre, best_pre))
print("{:10} : {:.5}    | {:.5}".format("Recall", baseline_recall, best_recall))
print("{:10} : {:.5} | {:.5}".format("F1 Score", baseline_f1, best_f1))
print(classification_report(y_test, baseline_pred))
print(classification_report(y_test, best_pred))