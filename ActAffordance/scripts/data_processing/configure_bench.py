import argparse
import cv2
import os
import json
import numpy as np
import shutil
from PIL import Image

def rename_folders(directory):
    # List all the folders in the directory
    for folder in os.listdir(directory):
        # Extract the base name (remove the file extension)
        base_name = folder.split('.')[0]
        
        # Ensure the base name is numeric (only digits)
        if base_name.isdigit():
            # Convert to an integer to remove any excess leading zeros
            new_base_name = str(int(base_name)).zfill(7)  # Ensure it's exactly 7 digits
            new_name = new_base_name
            
            # Get the full path to the old and new folder names
            old_path = os.path.join(directory, folder)
            new_path = os.path.join(directory, new_name)
            
            # Rename the folder
            os.rename(old_path, new_path)
            
            print(f'Renamed: {old_path} -> {new_path}')

def add_inpainting(frame_directory, source_directory):
    # List all the .png files in the source directory
    for file in os.listdir(source_directory):
        if file.endswith('.png'):
            # Extract the base name (without extension)
            base_name = file.split('.')[0]
            
            # Construct the full path to the corresponding frame folder
            frame_folder_path = os.path.join(frame_directory, base_name)
            
            # Check if the frame folder exists
            if os.path.isdir(frame_folder_path):
                # Define the full path to the source file
                source_file = os.path.join(source_directory, file)
                
                # Read the image using OpenCV
                img = cv2.imread(source_file)
                
                if img is not None:
                    # Perform the color conversion (RGB -> BGR) and reverse the channels
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = img[:,:,::-1]  # Reverse the color channels

                    # Define the path to save the image as 'inpainting.png' in the frame folder
                    destination_file = os.path.join(frame_folder_path, 'inpainting.png')

                    # Save the processed image
                    cv2.imwrite(destination_file, img)
                    print(f'Processed and saved: {source_file} -> {destination_file}')
                else:
                    print(f'Failed to read image: {source_file}')
            else:
                print(f'No matching folder for {file} in {frame_directory}')

def pad_to_square(mask):
    """
    Pads the input image mask to make it square.
    The padding is added either to the top or left depending on which dimension is larger.

    Args:
    - mask (numpy array): A 2D numpy array representing the binary mask image.

    Returns:
    - padded_mask (numpy array): The square padded mask image.
    """
    # Get the current height and width of the mask
    height, width = mask.shape
    
    # Determine the margin to pad based on the larger dimension
    if height > width:
        # If height is greater, pad the left side to make it square
        padding = (height - width)  # This is the amount to pad on the left
        padded_mask = np.pad(mask, ((0, 0), (padding, 0)), mode='constant', constant_values=0)
    elif width > height:
        # If width is greater, pad the top side to make it square
        padding = (width - height)  # This is the amount to pad on the top
        padded_mask = np.pad(mask, ((padding, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        # If the mask is already square, no padding needed
        return mask

    return padded_mask

def copy_json_with_target_box(source_path, destination_path, target_box):
    """
    Copies a JSON file from source_path to destination_path and adds a 'target_box' field.

    Parameters:
        source_path (str): Path to the source JSON file.
        destination_path (str): Path to the destination JSON file.
        target_box (list or tuple): Bounding box coordinates, e.g., [x_min, y_min, x_max, y_max].

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load original JSON content
        with open(source_path, 'r') as f:
            data = json.load(f)

        # Add target_box field
        #import pdb;pdb.set_trace()
        target_box = list(target_box)
        target_box = [int(index) for index in target_box]
        data['target_box'] = target_box

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Save to new location with updated content
        with open(destination_path, 'w') as f:
            json.dump(data, f, indent=4)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def add_annotations(frame_directory, annotation_directory):
    # List all the annotation folders in the annotation directory
    for folder in os.listdir(annotation_directory):
        annotation_folder_path = os.path.join(annotation_directory, folder)
        
        # Check if it's a directory and contains the 'annotation.json' file
        if os.path.isdir(annotation_folder_path):
            annotation_file_path = os.path.join(annotation_folder_path, 'annotation.json')
            
            if os.path.isfile(annotation_file_path):
                frame_folder_path = os.path.join(frame_directory, folder)
                
                # Check if the corresponding frame folder exists in the frame directory
                if os.path.isdir(frame_folder_path):
                    destination_file_path = os.path.join(frame_folder_path, 'annotation.json')
                    
                    # Copy the annotation file to the respective frame folder
                    shutil.copy(annotation_file_path, destination_file_path)
                    print(f'Copied annotation.json from {annotation_file_path} to {destination_file_path}')
                else:
                    print(f'No matching frame folder for {folder} in {frame_directory}')
            else:
                print(f'No annotation.json found in {annotation_folder_path}')
        else:
            print(f'{annotation_folder_path} is not a directory')

def add_objects(frame_directory, object_directory):

    frames = os.listdir(frame_directory)

    object_left_dir = os.path.join(object_directory, 'left')

    for file in os.listdir(object_left_dir):
        basename = file.split('.')[0]
        if basename in frames:
            file_path = os.path.join(object_left_dir, file)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            mask = pad_to_square(mask)
            new_file_path = os.path.join(frame_directory, basename, 'obj_left.png')
            print(f"Saving {basename} to {new_file_path}")
            cv2.imwrite(new_file_path, mask)
    
    object_right_dir = os.path.join(object_directory, 'right')

    for file in os.listdir(object_right_dir):
        basename = file.split('.')[0]
        if basename in frames:
            file_path = os.path.join(object_right_dir, file)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            mask = pad_to_square(mask)
            new_file_path = os.path.join(frame_directory, basename, 'obj_right.png')
            print(f"Saving {basename} to {new_file_path}")
            cv2.imwrite(new_file_path, mask)


def get_bounding_box(mask_image_path):
    """
    Calculate the bounding box of the masked area in the image.
    Returns the bounding box as (min_x, min_y, max_x, max_y).
    """
    mask = Image.open(mask_image_path).convert("L")  # Convert to grayscale
    mask = np.array(mask)

    # Get coordinates of the non-zero pixels
    y, x = np.where(mask > 0)
    
    if len(x) == 0 or len(y) == 0:
        return None  # No non-zero pixels

    # Calculate the bounding box
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    return min_x, min_y, max_x, max_y

def get_shape(mask_image_path):
    mask = Image.open(mask_image_path).convert('L')
    mask = np.array(mask)

    return mask.shape

def merge_bounding_boxes(box1, box2):
    """
    Merge two bounding boxes and return the combined one.
    """
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2

    # Combine the bounding boxes
    min_x = min(min_x1, min_x2)
    min_y = min(min_y1, min_y2)
    max_x = max(max_x1, max_x2)
    max_y = max(max_y1, max_y2)

    return min_x, min_y, max_x, max_y

def apply_offset(bbox, image_size, offset=50):
    """
    Apply offset to the bounding box and make sure it stays within the image boundaries.
    """
    min_x, min_y, max_x, max_y = bbox
    img_width, img_height = image_size

    # Apply offset, ensuring the bounding box stays within the image bounds
    min_x = max(min_x - offset, 0)
    min_y = max(min_y - offset, 0)
    max_x = min(max_x + offset, img_width)
    max_y = min(max_y + offset, img_height)

    return min_x, min_y, max_x, max_y

def crop_and_save_images(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        # Check if we're in a "leaf" folder
        if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
            # Create the corresponding output folder
            relative_path = os.path.relpath(root, input_folder)
            output_leaf_folder = os.path.join(output_folder, relative_path)
            os.makedirs(output_leaf_folder, exist_ok=True)

            # Copy the annotation.json file
            annotation_file = os.path.join(root, 'annotation.json')
            """
            if os.path.exists(annotation_file):
                shutil.copy(annotation_file, os.path.join(output_leaf_folder, 'annotation.json'))
            """

            # Process the obj_left.png and obj_right.png files
            obj_left_path = os.path.join(root, 'obj_left.png')
            obj_right_path = os.path.join(root, 'obj_right.png')
            bbox_left = None
            bbox_right = None
            # Get bounding boxes for obj_left and obj_right (if they exist)
            if os.path.exists(obj_left_path):
                bbox_left = get_bounding_box(obj_left_path)
                shp = get_shape(obj_left_path)
            if os.path.exists(obj_right_path):
                bbox_right = get_bounding_box(obj_right_path)
                shp = get_shape(obj_right_path)
            print(f"Shape: {shp}")
            print(f"Bbox Left: {bbox_left} and Bbox Right: {bbox_right}")

            # Merge the bounding boxes
            bounding_box = merge_bounding_boxes(bbox_left, bbox_right)
            print(f"After Merge: {bounding_box}")

            if bounding_box:
                # Apply offset and ensure it's within bounds

                #copy_json_with_target_box(annotation_file, os.path.join(output_leaf_folder, 'annotation.json'), bounding_box)
                img = Image.open(os.path.join(root, 'inpainting.png'))
                print("Unmodifed Image Size: ", img.size)
                img = img.resize(shp)
                print("Shape: ", shp)
                print("Image Size: ", img.size)
                img_width, img_height = img.size
                bounding_box = apply_offset(bounding_box, (img_width, img_height))
                print(f"After Offset {bounding_box}")
                copy_json_with_target_box(annotation_file, os.path.join(output_leaf_folder, 'annotation.json'), bounding_box)
                """
                min_x, min_y, max_x, max_y = bounding_box

                # Crop and save all PNG files (except 'bench_frame_overlay.png')
                for file in files:
                    if file.endswith('.png') and file != 'bench_frame_overlay.png':
                        file_path = os.path.join(root, file)
                        img = Image.open(file_path)
                        img = img.resize(shp)


                        # Crop the image using the bounding box
                        cropped_img = img.crop((min_x, min_y, max_x, max_y))

                        # Save the cropped image to the output folder
                        output_file_path = os.path.join(output_leaf_folder, file)
                        cropped_img.save(output_file_path)

                print(f"Processed and saved images in {output_leaf_folder}")
                """
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument for renaming folders
    parser.add_argument('--rename-folders', default=None, help='Directory where frame folders are located')

    # Arguments for adding inpainting files
    parser.add_argument('--add-inpainting', nargs=2, default=None, metavar=('FRAME_DIRECTORY', 'SOURCE_DIRECTORY'),
                        help='Source directory containing .png files to be added to the frame folders')

    # Arguments for adding annotations    
    parser.add_argument('--add-annotations', nargs=2, default=None, metavar=('FRAME_DIRECTORY', 'ANNOTATION_DIRECTORY'),
                    help='Directory containing frame folders and annotation folders')
    
    parser.add_argument('--add-objects', nargs=2, default=None, metavar=('FRAME_DIRECTORY', 'OBJECT_DIRECTORY'))

    parser.add_argument('--create-cropped', nargs=2, default=None, metavar=('BENCHMARK_DIRECTORY', 'OUTPUT_DIRECTORY'))

    args = parser.parse_args()

    if args.rename_folders:
        rename_folders(args.rename_folders)

    if args.add_inpainting:
        frame_directory, source_directory = args.add_inpainting
        add_inpainting(frame_directory, source_directory)

    if args.add_annotations:
        frame_directory, annotation_directory = args.add_annotations
        add_annotations(frame_directory, annotation_directory)
    
    if args.add_objects:
        frame_directory, object_directory = args.add_objects
        add_objects(frame_directory, object_directory)
    
    if args.create_cropped:
        benchmark_directory, output_directory = args.create_cropped
        crop_and_save_images(benchmark_directory, output_directory)
