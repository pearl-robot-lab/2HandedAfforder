import os
import shutil
from PIL import Image, ImageOps, ImageEnhance
import random
import argparse

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