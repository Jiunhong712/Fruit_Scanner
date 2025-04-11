import os
from tqdm import tqdm

# Path where both images and YOLO .txt label files are stored
labels_folder = r"C:\Users\xavie\Downloads\apples\train\renamed_data"

# Original class names
old_class_names = ['100-_ripeness', '20-_ripeness', '50-_ripeness', '75-_ripeness', 'rotten_apple']

# Final desired class names
new_class_names = ["Apple Ripe", "Apple Unripe", "Apple Rotten"]

# Mapping: old class ID -> new class ID
merge_map = {
    0: 0,
    2: 1,
    3: 0,
    4: 2,
}

# List all .txt files in the folder
label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

# Loop with progress bar
for file in tqdm(label_files, desc="Merging label classes"):
    file_path = os.path.join(labels_folder, file)

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        old_class_id = int(parts[0])
        if old_class_id not in merge_map:
            continue  # skip unknown class

        new_class_id = merge_map[old_class_id]
        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))

    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("\nLabel files have been updated with merged classes.")
