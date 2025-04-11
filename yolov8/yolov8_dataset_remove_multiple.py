import os
import shutil
from collections import defaultdict
from tqdm import tqdm
import random

# Define paths
images_dir = r"C:\Users\xavie\Downloads\apples\train\images"
labels_dir = r"C:\Users\xavie\Downloads\apples\train\labels"
backup_dir = r"C:\Users\xavie\Downloads\apples\removed"

# Make sure backup folder exists
os.makedirs(backup_dir, exist_ok=True)

# Class name mapping (optional, for readability)
class_names = [
    'Apple Ripe', 'Apple Rotten', 'Apple Unripe',
    'Banana Ripe', 'Banana Rotten', 'Banana Unripe',
    'Grape Ripe', 'Grape Rotten', 'Grape Unripe',
    'Mango Ripe', 'Mango Rotten', 'Mango Unripe',
    'Orange Ripe', 'Orange Rotten', 'Orange Unripe'
]

# Group image-label pairs by class
class_samples = defaultdict(list)
image_labels = defaultdict(set)  # To track classes per image
class_object_count = defaultdict(int)  # To track total number of objects for each class

print("Scanning label files...")

# Scan through all label files and count objects per class
for lbl_file in os.listdir(labels_dir):
    if not lbl_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, lbl_file)
    image_name = os.path.splitext(lbl_file)[0] + ".jpg"
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            continue

        # Collect all classes for this image (since an image can have multiple objects of different classes)
        classes_in_image = set()
        for line in lines:
            class_id = int(line.split()[0])
            classes_in_image.add(class_id)
            class_samples[class_id].append((image_path, label_path))
            class_object_count[class_id] += 1

        image_labels[image_path] = classes_in_image

# Get the minimum number of object instances across all classes
min_count = min(class_object_count.values())
print(f"Minimum object count per class: {min_count}")

# Balancing: Remove images where any class exceeds the threshold for object instances
removed_images = set()  # Track removed images to avoid double removal

for cls_id, class_count in class_object_count.items():
    class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
    print(f"Balancing class: {class_name} (original: {class_count}, keeping: {min_count} objects)")

    # Shuffle to randomize (optional)
    random.shuffle(class_samples[cls_id])

    # Remove extras from images where the class is overrepresented
    objects_to_remove = class_count - min_count
    removed_objects = 0

    for img_path, lbl_path in tqdm(class_samples[cls_id], desc=f"Removing extras for {class_name}"):
        if img_path in removed_images:
            continue

        # Count how many instances of the current class are in this image
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            object_count_in_image = sum(1 for line in lines if int(line.split()[0]) == cls_id)

        if removed_objects + object_count_in_image > objects_to_remove:
            # If removing this image would exceed the balance threshold, skip it
            break

        # Otherwise, move this image and label to the backup folder
        shutil.move(img_path, os.path.join(backup_dir, os.path.basename(img_path)))
        shutil.move(lbl_path, os.path.join(backup_dir, os.path.basename(lbl_path)))
        removed_images.add(img_path)
        removed_objects += object_count_in_image

print("\nDataset balanced! Extra samples backed up at:", backup_dir)
