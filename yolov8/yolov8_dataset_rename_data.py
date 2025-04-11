import os
from tqdm import tqdm

# Paths
images_dir = r"C:\Users\xavie\Downloads\banana\train\images"
labels_dir = r"C:\Users\xavie\Downloads\banana\train\labels"

# Define class names
class_names = [
    'Apple Overripe', 'Apple Ripe', 'Apple Rotten', 'Apple Unripe',
    'Banana Overripe', 'Banana Ripe', 'Banana Rotten', 'Banana Unripe',
    'Grape Overripe', 'Grape Ripe', 'Grape Rotten', 'Grape Unripe',
    'Mango Overripe', 'Mango Ripe', 'Mango Rotten', 'Mango Unripe',
    'Melon Overripe', 'Melon Ripe', 'Melon Rotten', 'Melon Unripe',
    'Orange Overripe', 'Orange Ripe', 'Orange Rotten', 'Orange Unripe',
    'Peach Overripe', 'Peach Ripe', 'Peach Rotten', 'Peach Unripe',
    'Pear Overripe', 'Pear Ripe', 'Pear Rotten', 'Pear Unripe'
]

# Counter for renamed files per class
counter = {cls: 1 for cls in class_names}

# Get all label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

for label_file in tqdm(label_files, desc="Renaming images and labels"):
    label_path = os.path.join(labels_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        print(f"⚠️ Image missing for label: {label_file}")
        continue

    try:
        with open(label_path, 'r') as f:
            # Remove whitespace-only lines
            lines = [line.strip() for line in f if line.strip()]
    except PermissionError:
        print(f"⚠️ Cannot read locked file: {label_path}")
        continue

    if not lines:
        print(f"⚠️ Skipping empty or invalid label file: {label_file}")
        continue

    try:
        class_id = int(lines[0].split()[0])
        class_name = class_names[class_id]
    except (IndexError, ValueError):
        print(f"⚠️ Skipping malformed label: {label_file}")
        continue

    # Try to generate a unique new name with `x` at the end
    while True:
        new_base_name = f"{class_name}_{counter[class_name]:04d}z"
        new_img_name = new_base_name + '.jpg'
        new_lbl_name = new_base_name + '.txt'
        new_image_path = os.path.join(images_dir, new_img_name)
        new_label_path = os.path.join(labels_dir, new_lbl_name)

        if not os.path.exists(new_image_path) and not os.path.exists(new_label_path):
            break

        counter[class_name] += 1

    try:
        os.rename(image_path, new_image_path)
        os.rename(label_path, new_label_path)
        counter[class_name] += 1
    except PermissionError:
        print(f"⚠️ Skipping locked file: {label_path}")
        continue

print("\n✅ All files renamed. Empty and malformed labels were skipped.")
