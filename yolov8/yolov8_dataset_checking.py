import os
from collections import defaultdict

# Path to your YOLOv8 labels folder
labels_dir = r"C:\Users\xavie\Downloads\dataset\train\labels"

# Define your class names
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

# Only show these class IDs
selected_classes = [1, 2, 3, 5, 6, 7, 13, 14, 15, 21, 22]

# Dictionary to count number of images containing each class
class_counts = defaultdict(int)

# Loop over label files
for file in os.listdir(labels_dir):
    if not file.endswith('.txt'):
        continue

    file_path = os.path.join(labels_dir, file)
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            class_ids = set(int(line.split()[0]) for line in lines)
            for class_id in class_ids:
                if class_id in selected_classes:
                    class_counts[class_id] += 1
    except:
        print(f"⚠️ Could not read: {file}")

# Display only selected class counts
print("\n📊 Image counts for selected classes:")
for class_id in selected_classes:
    class_name = class_names[class_id]
    count = class_counts[class_id]
    print(f"{class_id:02d} - {class_name:<20}: {count}")
