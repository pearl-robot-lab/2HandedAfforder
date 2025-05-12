import os
import json
import cv2
import numpy as np
import argparse

def mask_to_polygon(mask):
    """
    Convert a binary mask to a polygon representation.
    Args:
        mask (numpy.ndarray): Binary mask (0 and 255 values).
    Returns:
        list: List of polygons represented as lists of (x, y) coordinates.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # A valid polygon must have at least 3 points
            polygon = [(int(point[0][0]), int(point[0][1])) for point in contour]
            polygons.append(polygon)
    return polygons

def polygon_to_mask(polygons, shape):
    """
    Convert polygon representation back to a binary mask.
    Args:
        polygons (list): List of polygons represented as lists of (x, y) coordinates.
        shape (tuple): Shape of the output binary mask (height, width).
    Returns:
        numpy.ndarray: Binary mask with polygons filled.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask

def process_folders(aff_left_folder, aff_right_folder, obj_left_folder, obj_right_folder, output_file):
    """
    Process masks from four folders and save their polygon representations into a JSON file.
    """
    # Collect filenames from all four folders
    aff_left_files = set(os.listdir(aff_left_folder))
    aff_right_files = set(os.listdir(aff_right_folder))
    obj_left_files = set(os.listdir(obj_left_folder))
    obj_right_files = set(os.listdir(obj_right_folder))

    # Find shared filenames
    shared_files = aff_left_files & aff_right_files & obj_left_files & obj_right_files

    data = {}
    for idx, file_name in enumerate(shared_files):
        # Paths to the masks
        aff_left_path = os.path.join(aff_left_folder, file_name)
        aff_right_path = os.path.join(aff_right_folder, file_name)
        obj_left_path = os.path.join(obj_left_folder, file_name)
        obj_right_path = os.path.join(obj_right_folder, file_name)

        # Read masks
        aff_left_mask = cv2.imread(aff_left_path, cv2.IMREAD_GRAYSCALE)
        aff_right_mask = cv2.imread(aff_right_path, cv2.IMREAD_GRAYSCALE)
        obj_left_mask = cv2.imread(obj_left_path, cv2.IMREAD_GRAYSCALE)
        obj_right_mask = cv2.imread(obj_right_path, cv2.IMREAD_GRAYSCALE)

        if any(mask is None for mask in [aff_left_mask, aff_right_mask, obj_left_mask, obj_right_mask]):
            print(f"Skipping {file_name}: Could not read all masks.")
            continue

        # Convert to polygons
        aff_left_poly = mask_to_polygon(aff_left_mask)
        aff_right_poly = mask_to_polygon(aff_right_mask)
        obj_left_poly = mask_to_polygon(obj_left_mask)
        obj_right_poly = mask_to_polygon(obj_right_mask)

        # Store the polygon data
        data[str(idx)] = {
            "aff_left": aff_left_poly,
            "aff_right": aff_right_poly,
            "obj_left": obj_left_poly,
            "obj_right": obj_right_poly
        }

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Polygon data saved to {output_file}")

def json_entry_to_masks(json_entry, mask_shape):
    """
    Convert a JSON entry back to binary masks.
    Args:
        json_entry (dict): JSON entry containing polygon data.
        mask_shape (tuple): Shape of the output masks (height, width).
    Returns:
        dict: Dictionary containing the binary masks for "aff_left", "aff_right", "obj_left", and "obj_right".
    """
    masks = {
        "aff_left": polygon_to_mask(json_entry["aff_left"], mask_shape),
        "aff_right": polygon_to_mask(json_entry["aff_right"], mask_shape),
        "obj_left": polygon_to_mask(json_entry["obj_left"], mask_shape),
        "obj_right": polygon_to_mask(json_entry["obj_right"], mask_shape)
    }
    return masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mask images to JSON polygon representation.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    parser.add_argument("--aff_left_folder", help="Path to the folder containing aff_left masks.")
    parser.add_argument("--aff_right_folder", help="Path to the folder containing aff_right masks.")
    parser.add_argument("--obj_left_folder", help="Path to the folder containing obj_left masks.")
    parser.add_argument("--obj_right_folder", help="Path to the folder containing obj_right masks.")
    parser.add_argument("--mask_shape", type=int, nargs=2, help="Height and width of masks (required for reconstruction).", required=False)
    parser.add_argument("--reconstruct_index", type=str, help="Index of the JSON entry to reconstruct (optional).")
    args = parser.parse_args()

    if args.reconstruct_index and args.mask_shape:
        # Load JSON and reconstruct masks
        with open(args.output_file, 'r') as f:
            json_data = json.load(f)
        
        if args.reconstruct_index in json_data:
            masks = json_entry_to_masks(json_data[args.reconstruct_index], tuple(args.mask_shape))
            for key, mask in masks.items():
                cv2.imwrite(f"{key}_{args.reconstruct_index}.png", mask)
            print(f"Masks reconstructed for index {args.reconstruct_index} and saved.")
        else:
            print(f"Index {args.reconstruct_index} not found in JSON file.")
    else:
        # Process folders and save JSON file
        process_folders(args.aff_left_folder, args.aff_right_folder, args.obj_left_folder, args.obj_right_folder, args.output_file)
