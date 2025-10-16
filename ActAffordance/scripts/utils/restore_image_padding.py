import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def pad_image_to_original_size(image, target_box, original_size=(850, 850)):
    x_min, y_min, x_max, y_max = target_box
    padded = Image.new("RGB", original_size)
    padded.paste(image, (x_min, y_min))
    return padded


def process_images(cropped_dir, annotation_dir, output_dir, original_size=(850, 850)):
    cropped_dir = Path(cropped_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    for root, _, files in os.walk(cropped_dir):
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        rel_path = Path(root).relative_to(cropped_dir)
        annotation_path = annotation_dir / rel_path / "annotation.json"
        output_folder = output_dir / rel_path
        output_folder.mkdir(parents=True, exist_ok=True)

        if not annotation_path.exists():
            print(f"Warning: Missing annotation.json in {annotation_path}")
            continue

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
            target_box = annotation.get("target_box")
            if not target_box or len(target_box) != 4:
                print(f"Invalid or missing 'target_box' in {annotation_path}")
                continue

        for image_name in images[:2]:  # Max 2 images per folder
            image_path = Path(root) / image_name
            with Image.open(image_path) as img:
                padded_img = pad_image_to_original_size(img, target_box, original_size)
                padded_img.save(output_folder / image_name)


def main():
    parser = argparse.ArgumentParser(description="Pad cropped images to original size using annotations.")
    parser.add_argument("--cropped_dir", type=str, required=True, help="Directory with cropped images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with annotation.json files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save padded images")

    args = parser.parse_args()

    process_images(args.cropped_dir, args.annotation_dir, args.output_dir)


if __name__ == "__main__":
    main()
