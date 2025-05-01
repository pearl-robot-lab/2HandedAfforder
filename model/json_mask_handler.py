import cv2
import os
import ujson as json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

def load_json_file_for_index(index, json_folder):
    """Load the JSON file that contains the given index based on filename range."""
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            # Extract the range from the filename
            range_part = json_file.split('_')[0]  # e.g., "1000-2510"
            start_idx, end_idx = map(int, range_part.split('-'))

            # Check if the index falls within this range
            if start_idx <= index <= end_idx:
                json_path = os.path.join(json_folder, json_file)
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                return json_data, index - start_idx

    raise ValueError(f"No JSON file found for index {index}.")

def load_h5_file_for_index(index, h5_folder):
    """Load the H5 file corresponding to the index based on filename range."""
    for h5_file in os.listdir(h5_folder):
        if h5_file.endswith('.h5'):
            # Extract the range from the filename
            range_part = h5_file.split('_')[0]  # e.g., "1000-2510"
            start_idx, end_idx = map(int, range_part.split('-'))

            # Check if the index falls within this range
            if start_idx <= index <= end_idx:
                h5_path = os.path.join(h5_folder, h5_file)
                return h5_path, start_idx, end_idx

    raise ValueError(f"No H5 file found for index {index}.")

def recreate_mask_from_contours(contours, shape):
    """Reconstruct a binary mask from OpenCV contours."""
    mask = np.zeros(shape, dtype=np.uint8)  # Create a blank mask with the same size as original
    for contour in contours:
        # Draw the contour on the mask
        cv2.drawContours(mask, [np.array(contour, dtype=np.int32)], -1, (1), thickness=cv2.FILLED)
    return mask

def get_masks_and_overlay(index, json_folder, h5_folder):
    """Retrieve full masks recreated from contours in JSON file, and overlay them on the 'inpainted' image from H5."""
    # Load JSON data
    json_data, local_index = load_json_file_for_index(index, json_folder)
    
    # Get the mask data for the local_index
    mask_data = json_data.get(str(local_index))
    if mask_data is None:
        raise IndexError(f"Invalid index {local_index} for masks in {json_data['original_size']}.")

    # Recreate masks as numpy arrays from contours for all 4 masks
    original_size = mask_data['original_size']
    recreated_masks = {}

    for mask_key in ['aff_left', 'aff_right', 'obj_left', 'obj_right']:
        contours = mask_data[mask_key]
        recreated_masks[mask_key] = recreate_mask_from_contours(contours, shape=original_size)

    # Now load the 'inpainted' image from the corresponding H5 file
    h5_file_path, start_idx, end_idx = load_h5_file_for_index(index, h5_folder)

    with h5py.File(h5_file_path, 'r') as f:
        group_name = list(f.keys())[0]  # Assume only one group per file
        group = f[group_name]

        # Load the 'inpainted' field
        inpainted_image = np.array(group['inpainted'])[local_index]
        inpainted_image = cv2.resize(inpainted_image, original_size)
        #assert inpainted_image.shape[:2] == original_size, "The inpainted image size doesn't match the expected size."

    # Overlay masks on the inpainted image and store them
    overlays = {}
    for mask_key, mask in recreated_masks.items():
        # Create an RGB image for the overlay (if inpainted is grayscale, expand to RGB)
        if len(inpainted_image.shape) == 2:
            inpainted_rgb = np.stack([inpainted_image] * 3, axis=-1)
        else:
            inpainted_rgb = inpainted_image  # Assume it's already an RGB image
        
        # Create the overlay: put the mask on top of the image (red channel, for example)
        overlay = inpainted_rgb.copy()
        overlay[mask == 1] = [255, 0, 0]  # Red color for the masked area
        overlays[mask_key] = overlay

    return recreated_masks, overlays

# Example usage:
"""
index = 2000
json_folder = '../EPIC_DATA/hdf5_sets/datasets/dset_test_updated_2/jsons'
h5_folder = '../EPIC_DATA/hdf5_sets/datasets/dset_test_updated_2/h5'
mask_data, overlays = get_masks_and_overlay(index, json_folder, h5_folder)
counter = 0
for mask in overlays.values():
    counter += 1
    Image.fromarray(mask).save('test_' + str(counter) + '.png')
"""
# display_overlay(overlays)  # Show the overlaid masks
