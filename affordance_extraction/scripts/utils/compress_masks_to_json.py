import h5py
import numpy as np
import json
import os
import argparse
import cv2

def convert_h5_to_json(h5_folder):
    json_folder = os.path.join(h5_folder, '../jsons')
    os.makedirs(json_folder, exist_ok=True)

    # Iterate over all H5 files in the directory
    for h5_file in os.listdir(h5_folder):
        if h5_file.endswith('.h5'):
            h5_path = os.path.join(h5_folder, h5_file)
            json_path = os.path.join(json_folder, f"{os.path.splitext(h5_file)[0]}.json")
            print(f"Processing {h5_file}...")

            with h5py.File(h5_path, 'r') as f:
                group_name = list(f.keys())[0]  # Assume only one group per file
                group = f[group_name]

                # Extract the object masks (4 mask stacks, each 4-dimensional)
                affs_left = np.array(group['aff_left'])
                affs_right = np.array(group['aff_right'])
                objs_left = np.array(group['obj_mask_left'])
                objs_right = np.array(group['obj_mask_right'])
                mask_stacks = {
                    'aff_left': affs_left,
                    'aff_right': affs_right,
                    'obj_mask_left': objs_left,
                    'obj_mask_right': objs_right
                }

                # Prepare the final JSON structure
                json_data = {}

                # Iterate over each individual mask in the 4 mask stacks
                for local_index in range(affs_left.shape[0]):  # Assumes all masks have the same shape
                    local_masks = []
                    for mask_stack in mask_stacks.values():
                        individual_mask = mask_stack[local_index]  # Extract the individual mask
                        contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        local_masks.append([contour[:, 0, :].tolist() for contour in contours])  # Flatten contours for JSON

                    # Save the individual mask data into the JSON structure
                    json_data[local_index] = {
                        'original_size': mask_stacks['aff_left'].shape[1:],  # (height, width)
                        'aff_left': local_masks[0],
                        'aff_right': local_masks[1],
                        'obj_left': local_masks[2],
                        'obj_right': local_masks[3]
                    }

                # Write the data into the JSON file
                with open(json_path, 'w') as json_file:
                    json.dump(json_data, json_file)
            print(f"Saved {json_path}")

def convert_masks_to_json(affs_left, affs_right, objs_left, objs_right, out):
    json_path = out
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    mask_stacks = {
        'aff_left': affs_left,
        'aff_right': affs_right,
        'obj_mask_left': objs_left,
        'obj_mask_right': objs_right
    }

    # Prepare the final JSON structure
    json_data = {}

    # Iterate over each individual mask in the 4 mask stacks
    for local_index in range(affs_left.shape[0]):  # Assumes all masks have the same shape
        local_masks = []
        for mask_stack in mask_stacks.values():
            individual_mask = mask_stack[local_index]  # Extract the individual mask
            contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            local_masks.append([contour[:, 0, :].tolist() for contour in contours])  # Flatten contours for JSON

        # Save the individual mask data into the JSON structure
        json_data[local_index] = {
            'original_size': mask_stacks['aff_left'].shape[1:],  # (height, width)
            'aff_left': local_masks[0],
            'aff_right': local_masks[1],
            'obj_left': local_masks[2],
            'obj_right': local_masks[3]
        }

    # Write the data into the JSON file
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert H5 files to JSON polygon representation")
    parser.add_argument("--h5_folder", help="Path to the folder containing H5 files")
    args = parser.parse_args()
    if args.h5_folder:
        convert_h5_to_json(args.h5_folder)
