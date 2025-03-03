import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\IDP_CNN_MobileNetV2.keras"
model = load_model(model_path)

# Load an image for prediction
img_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\sample3.jpeg"

img = image.load_img(img_path, target_size=(160, 160))

# Convert image to array
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Load class names
classes = sorted([d for d in os.listdir(r'C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\Dataset3\train') if
                  os.path.isdir(os.path.join(r'C:\Users\xavie\OneDrive\Documents\Y4S1\IDP B\Dataset3\train', d))])

# Print predicted class
print("Predicted class:", classes[predicted_class])

