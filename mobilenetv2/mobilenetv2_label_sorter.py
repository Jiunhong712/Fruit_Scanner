import os
import shutil
import pandas as pd


def classify_images(dataset_path):
    # Define class mapping
    class_mapping = {
        "ripe": "ripe",
        "unripe": "unripe"
    }

    for split in ["train", "test", "valid"]:
        split_path = os.path.join(dataset_path, split)
        annotations_file = os.path.join(split_path, "_annotations.csv")

        if not os.path.exists(annotations_file):
            print(f"Skipping {split}: _annotations.csv not found.")
            continue

        # Read annotation file
        df = pd.read_csv(annotations_file)

        # Move images based on label
        for _, row in df.iterrows():
            filename = str(row.iloc[0])
            label = class_mapping.get(str(row.iloc[-1]))

            if not label:
                print(f"Warning: Label '{row.iloc[-1]}' not found in class_mapping. Skipping {filename}.")
                continue

            source_path = os.path.join(split_path, filename)
            destination_folder = os.path.join(split_path, label)

            # Ensure the destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Move file if it exists
            if os.path.exists(source_path):
                shutil.move(source_path, os.path.join(destination_folder, filename))
            else:
                print(f"Warning: {filename} not found in {split}/. Skipping...")

        print(f"Classification done for {split}")


# Classification
dataset_path = os.path.expanduser(
    r"C:\Users\xavie\Downloads\strawberry.v1i.retinanet")
classify_images(dataset_path)

print("Classification complete! Check train, test, and valid folders.")
