import os
import shutil
from collections import defaultdict
from tqdm import tqdm

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

print("Scanning label files...")
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

        # Use the class from the first line (main class)
        class_id = int(lines[0].split()[0])
        class_samples[class_id].append((image_path, label_path))

# Get the minimum number of samples across all classes
min_count = min(len(samples) for samples in class_samples.values())
print(f"Minimum class sample count: {min_count}")

# Balance classes
for cls_id, samples in class_samples.items():
    class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
    print(f"Balancing class: {class_name} (original: {len(samples)}, keeping: {min_count})")

    # Shuffle to randomize (optional)
    import random
    random.shuffle(samples)

    # Keep only first min_count samples, move extras
    for extra_sample in tqdm(samples[min_count:], desc=f"Removing extras from {class_name}"):
        img_path, lbl_path = extra_sample
        shutil.move(img_path, os.path.join(backup_dir, os.path.basename(img_path)))
        shutil.move(lbl_path, os.path.join(backup_dir, os.path.basename(lbl_path)))

print("\nDataset balanced! Extra samples backed up at:", backup_dir)
