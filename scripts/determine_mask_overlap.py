import os
import cv2
import numpy as np
import argparse

def pad_image(image):
    h, w = image.shape[:2]
    if h > w:
        padding = (h - w, 0)
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding[0], 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padding = (w - h, 0)
        padded_image = cv2.copyMakeBorder(image, padding[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image


def process_mask_overlap(mask_differences, hand_masks):
    """
    Updates the masks in 'mask_differences' folder to include only the overlapping regions
    with the masks in the 'hand_masks' folder.
    """
    for file_name in os.listdir(mask_differences):
        # Generate file paths
        mask_diff_path = os.path.join(mask_differences, file_name)
        hand_mask_path = os.path.join(hand_masks, file_name)

        # Ensure the corresponding file exists in both folders
        if not os.path.isfile(hand_mask_path):
            print(f"Skipping {file_name}: No corresponding file in {hand_masks}")
            continue

        # Read the masks
        mask_diff = cv2.imread(mask_diff_path, cv2.IMREAD_GRAYSCALE)
        hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)


        if mask_diff is None or hand_mask is None:
            print(f"Skipping {file_name}: Could not read one or both masks.")
            continue

        hand_mask = pad_image(hand_mask)

        # Ensure the masks have the same dimensions
        if mask_diff.shape != hand_mask.shape:
            print(f"Resizing {file_name} to match dimensions of {hand_mask_path}.")
            hand_mask = cv2.resize(hand_mask, (mask_diff.shape[1], mask_diff.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Calculate the overlapping region
        overlap = cv2.bitwise_and(mask_diff, hand_mask)

        # Save the updated mask back to the mask_differences folder
        cv2.imwrite(mask_diff_path, overlap)
        print(f"Updated mask saved: {mask_diff_path}")

if __name__ == "__main__":
    # Argument parser for input folders
    parser = argparse.ArgumentParser(description="Process mask overlaps between two folders.")
    parser.add_argument("mask_differences", help="Path to the folder containing the first set of masks (to be updated).")
    parser.add_argument("hand_masks", help="Path to the folder containing the second set of masks (reference masks).")
    args = parser.parse_args()

    # Process the masks
    process_mask_overlap(args.mask_differences, args.hand_masks)
