import os
from tqdm import tqdm

labels_folder = r"C:\Users\xavie\Downloads\banana\train\labels"

# Original class names
old_class_names = ['EXPORTACION', 'Rechazo']

# Final desired class names
new_class_names = ['Apple Overripe', 'Apple Ripe', 'Apple Rotten', 'Apple Unripe', 'Banana Overripe', 'Banana Ripe', 'Banana Rotten', 'Banana Unripe', 'Grape Overripe', 'Grape Ripe', 'Grape Rotten', 'Grape Unripe', 'Mango Overripe', 'Mango Ripe', 'Mango Rotten', 'Mango Unripe', 'Melon Overripe', 'Melon Ripe', 'Melon Rotten', 'Melon Unripe', 'Orange Overripe', 'Orange Ripe', 'Orange Rotten', 'Orange Unripe', 'Peach Overripe', 'Peach Ripe', 'Peach Rotten', 'Peach Unripe', 'Pear Overripe', 'Pear Ripe', 'Pear Rotten', 'Pear Unripe']

# Mapping: old class ID -> new class ID
merge_map = {
    0: 7,
    1: 7
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
