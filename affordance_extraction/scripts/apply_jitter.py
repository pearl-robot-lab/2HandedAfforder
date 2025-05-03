import os
import shutil
from PIL import Image, ImageOps, ImageEnhance
import random
import argparse
"""
def remove_padding(image):

    grayscale = image.convert("L")
    bbox = grayscale.getbbox()  # Get bounding box of the non-padded area
    return bbox

def crop_and_pad_image(image, bbox, target_size, padding_style):

    cropped_image = image.crop(bbox)  # Crop using the consistent bounding box

    # Determine padding based on the style
    width, height = cropped_image.size
    delta_w = target_size - width
    delta_h = target_size - height
    
    if padding_style == "equal":
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    elif padding_style == "left_top":
        padding = (delta_w, 0, 0, delta_h) if width < height else (0, delta_h, delta_w, 0)
    elif padding_style == "right_bottom":
        padding = (0, delta_h, delta_w, 0) if width < height else (delta_w, 0, 0, delta_h)
    
    return ImageOps.expand(cropped_image, padding, fill="black")

def apply_color_jitter(image):
    
    brightness_factor = random.uniform(0.4, 1.6)
    contrast_factor = random.uniform(0.4, 1.6)
    color_factor = random.uniform(0.4, 1.6)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image

def create_padded_and_jittered_copies(main_folder_path):
    for root, dirs, files in os.walk(main_folder_path):
        if any(file in files for file in ["obj_left.png", "obj_right.png", "aff_left.png", "aff_right.png", "inpainted_frame.png"]):
            # Determine the original folder name and create copies with padding styles and color jitter
            orig_folder_name = os.path.basename(root)
            #padding_styles = ["equal", "left_top", "right_bottom"]
            #padded_folder_paths = {
            #    style: os.path.join(os.path.dirname(root), f"{style}_padding_{orig_folder_name}")
            #    for style in padding_styles
            #}
            # Add fourth folder path for color jitter
            cj_folder_path = os.path.join(os.path.dirname(root), f"cj_{orig_folder_name}")
            
            # Copy the original folder structure to each padded and jittered folder
            #for style, path in padded_folder_paths.items():
            #    shutil.copytree(root, path)
            shutil.copytree(root, cj_folder_path)

            # Calculate bounding box from inpainted_frame.png
            inpainted_path = os.path.join(root, "inpainted_frame.png")
            if os.path.exists(inpainted_path):
                with Image.open(inpainted_path) as inpainted_image:
                    bbox = remove_padding(inpainted_image)  # Get the bounding box
                    if bbox:  # Proceed if a bounding box is found
                        target_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # Determine target size

                        # Process each image with the calculated bounding box
                        for image_file in ["obj_left.png", "obj_right.png", "aff_left.png", "aff_right.png", "inpainted_frame.png"]:
                            orig_image_path = os.path.join(root, image_file)
                            
                            if os.path.exists(orig_image_path):
                                with Image.open(orig_image_path) as image:
                                    # For each padding style, crop and pad, then save in the corresponding folder

                                    # For the cj folder, apply color jitter only to inpainted_frame.png
                                    if image_file == "inpainted_frame.png":
                                        print("jittering image: ", inpainted_path)
                                        jittered_image = apply_color_jitter(image)  # Apply jitter to original, unaltered image
                                        jittered_image.save(os.path.join(cj_folder_path, image_file))
"""

def remove_padding(image):
    """Remove padding from a square image by detecting the original non-padded region."""
    grayscale = image.convert("L")
    bbox = grayscale.getbbox()  # Get bounding box of the non-padded area
    return bbox

def crop_and_pad_image(image, bbox, target_size):
    """Crops the image to the specified bounding box and pads it to target size."""
    cropped_image = image.crop(bbox)  # Crop using the bounding box

    # Calculate padding dimensions to center the cropped image
    width, height = cropped_image.size
    delta_w = target_size - width
    delta_h = target_size - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    # Apply padding
    return ImageOps.expand(cropped_image, padding, fill="black")

def apply_color_jitter(image):
    """Apply random color jitter to the image by adjusting brightness, contrast, and color."""
    brightness_factor = random.uniform(0.4, 1.6)
    contrast_factor = random.uniform(0.4, 1.6)
    color_factor = random.uniform(0.4, 1.6)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image

def process_folders(main_folder_path):
    # Find all folders containing 'inpainted_frame.png'
    folders_with_inpainted = [
        root for root, _, files in os.walk(main_folder_path) if "inpainted_frame.png" in files
    ]
    
    # Select 1/4 of these folders to apply color jitter
    print("Folders with Inpainted Length: ", len(folders_with_inpainted))
    folders_to_jitter = random.sample(folders_with_inpainted, max(1, len(folders_with_inpainted) // 4))

    for root in folders_with_inpainted:
        inpainted_path = os.path.join(root, "inpainted_frame.png")
        
        # Open inpainted_frame.png and remove padding
        with Image.open(inpainted_path) as image:
            
            # Apply color jitter if the folder is selected for it
            if root in folders_to_jitter:
                print(f"Applying color jitter to {inpainted_path}")
                processed_image = apply_color_jitter(image)
            
                # Save the processed image back to the original file path
                processed_image.save(inpainted_path)

# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset by cropping, padding, and color jittering.")
    parser.add_argument("dataset", help="Path to the dataset folder")
    args = parser.parse_args()

    # Run the function with the provided dataset path
    process_folders(args.dataset)
"""
# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset with padding and color jitter.")
    parser.add_argument("dataset", help="Path to the dataset folder")
    args = parser.parse_args()

    # Run the function with the provided dataset path
    create_padded_and_jittered_copies(args.dataset)
"""