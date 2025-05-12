import os
import cv2
import numpy as np
import argparse

def calculate_iou(mask1, mask2):
    """Calculate the Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:  # Avoid division by zero
        return 0
    return intersection / union

def add_caption(image, text, position="top", font_scale=1, thickness=2):
    """Add a caption to an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10 if position == "top" else image.shape[0] - 10

    captioned_image = image.copy()
    cv2.putText(captioned_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return captioned_image

def process_masks(image_folder, mask1_folder, mask2_folder, original_folder, output_folder, iou_threshold):
    """Process masks and save visualizations based on IoU."""
    os.makedirs(output_folder, exist_ok=True)

    # Collect file base names from all folders
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder)}
    mask1_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask1_folder)}
    mask2_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask2_folder)}
    original_files = {os.path.splitext(f)[0]: f for f in os.listdir(original_folder)}

    # Find shared base names
    shared_names = set(image_files.keys()) & set(mask1_files.keys()) & set(mask2_files.keys()) & set(original_files.keys())

    for base_name in shared_names:
        # Get file paths using base names
        image_path = os.path.join(image_folder, image_files[base_name])
        mask1_path = os.path.join(mask1_folder, mask1_files[base_name])
        mask2_path = os.path.join(mask2_folder, mask2_files[base_name])
        original_path = os.path.join(original_folder, original_files[base_name])

        # Read images and masks
        image = cv2.imread(image_path)
        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_path)

        if image is None or mask1 is None or mask2 is None or original_image is None:
            print(f"Could not read one or more files for {base_name}, skipping...")
            continue

        # Resize masks to match image dimensions if needed
        if mask1.shape[:2] != image.shape[:2]:
            mask1 = cv2.resize(mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask2.shape[:2] != image.shape[:2]:
            mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Resize original image to match dimensions of the image
        if original_image.shape[:2] != image.shape[:2]:
            original_image = cv2.resize(original_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Calculate IoU
        iou = calculate_iou(mask1 > 0, mask2 > 0)
        print(f"IoU for {base_name}: {iou:.2f}")

        if iou > iou_threshold:
            continue

        # Create overlay images
        colored_mask1 = np.zeros_like(image)
        colored_mask1[:, :, 1] = mask1  # Green overlay for mask1
        overlay1 = cv2.addWeighted(image, 0.7, colored_mask1, 0.3, 0)

        colored_mask2 = np.zeros_like(image)
        colored_mask2[:, :, 2] = mask2  # Red overlay for mask2
        overlay2 = cv2.addWeighted(image, 0.7, colored_mask2, 0.3, 0)

        # Add captions
        original_image = add_caption(original_image, "Original", position="top")
        overlay1 = add_caption(overlay1, "MI-GAN", position="top")
        overlay2 = add_caption(overlay2, "SAM2", position="top")

        # Concatenate images horizontally
        concatenated = np.hstack((original_image, overlay1, overlay2))

        # Save the result
        output_path = os.path.join(output_folder, base_name + ".png")
        cv2.imwrite(output_path, concatenated)
        print(f"Saved visualization for {base_name} to {output_path}")

if __name__ == "__main__":
    # Argument parser for input and output directories
    parser = argparse.ArgumentParser(description="Overlay masks on images and calculate IoU.")
    parser.add_argument("image_folder", help="Path to the folder containing the images.")
    parser.add_argument("mask1_folder", help="Path to the folder containing the first set of masks.")
    parser.add_argument("mask2_folder", help="Path to the folder containing the second set of masks.")
    parser.add_argument("original_folder", help="Path to the folder containing the original images.")
    parser.add_argument("output_folder", help="Path to the folder where results will be saved.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for processing (default: 0.5).")

    args = parser.parse_args()

    # Call the function
    process_masks(args.image_folder, args.mask1_folder, args.mask2_folder, args.original_folder, args.output_folder, args.iou_threshold)
