import os
import argparse
from pathlib import Path
import cv2
import numpy as np


def process_image_inplace(image_path, kernel_size=(7, 7), threshold_value=0.5):
    # Load image in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Failed to load image {image_path}")
        return

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, 0)

    # Normalize and threshold
    normalized = blurred / 255.0
    binary = (normalized > threshold_value).astype(np.uint8) * 255

    # Overwrite original file
    cv2.imwrite(str(image_path), binary)


def process_directory_inplace(input_dir, kernel_size=(7, 7), threshold_value=0.5):
    input_dir = Path(input_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                image_path = Path(root) / file
                process_image_inplace(image_path, kernel_size, threshold_value)


def main():
    parser = argparse.ArgumentParser(description="Apply Gaussian filter and binary threshold in-place to images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold value (default: 0.5)")
    args = parser.parse_args()

    process_directory_inplace(args.input_dir, kernel_size=(7, 7), threshold_value=args.threshold)


if __name__ == "__main__":
    main()
