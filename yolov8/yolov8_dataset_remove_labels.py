import os
from tqdm import tqdm

# Paths
images_dir = r"C:\Users\xavie\Downloads\banana\train\images"
labels_dir = r"C:\Users\xavie\Downloads\banana\train\labels"

# Get all label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

deleted_count = 0

for label_file in tqdm(label_files, desc="Checking for missing images"):
    image_name = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(images_dir, image_name)
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(image_path):
        try:
            os.remove(label_path)
            deleted_count += 1
        except PermissionError:
            print(f"⚠️ Could not delete locked label: {label_file}")

print(f"\n✅ Done! Deleted {deleted_count} label files without images.")
