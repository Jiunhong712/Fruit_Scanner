import os
import shutil
from tqdm import tqdm

# Paths
images_dir = r'C:\Users\xavie\Downloads\apples\train\images'
labels_dir = r'C:\Users\xavie\Downloads\apples\train\labels'

# Define class names
class_names = ['100-_ripeness', '20-_ripeness', '50-_ripeness', '75-_ripeness', 'rotten_apple']

# Optional: where to save renamed images
renamed_dir = r'C:\Users\xavie\Downloads\apples\train\renamed_data'
os.makedirs(renamed_dir, exist_ok=True)

counter = {cls: 0 for cls in class_names}

# Get all label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# Process with tqdm progress bar
for label_file in tqdm(label_files, desc="Renaming images"):
    label_path = os.path.join(labels_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + '.jpg'  # adjust for .png if needed
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            continue

        # Use the first class only
        class_id = int(lines[0].split()[0])
        class_name = class_names[class_id]
        counter[class_name] += 1

        new_img_name = f"{class_name}_{counter[class_name]:04d}.jpg"
        new_lbl_name = f"{class_name}_{counter[class_name]:04d}.txt"

        # Copy and rename
        shutil.copy2(image_path, os.path.join(renamed_dir, new_img_name))
        shutil.copy2(label_path, os.path.join(renamed_dir, new_lbl_name))

print("All done! Renamed files are saved to:", renamed_dir)