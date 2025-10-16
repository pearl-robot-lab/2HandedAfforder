import argparse
import cv2
import os
import numpy as np

def overlay_masks_on_image(inpainting_path, aff_left_path=None, aff_right_path=None, output_path=None):
    # Read the inpainting image
    inpainting_img = cv2.imread(inpainting_path)
    
    if aff_left_path:
        # Read the 'aff_left.png' mask
        aff_left_mask = cv2.imread(aff_left_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a red-colored mask for 'aff_left.png'
        red_mask = np.zeros_like(inpainting_img)
        red_mask[aff_left_mask == 255] = [0, 0, 255]  # Red color
        
        # Overlay the red mask using addWeighted
        inpainting_img = cv2.addWeighted(inpainting_img, 1, red_mask, 0.5, 0)
    
    if aff_right_path:
        # Read the 'aff_right.png' mask
        aff_right_mask = cv2.imread(aff_right_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a green-colored mask for 'aff_right.png'
        green_mask = np.zeros_like(inpainting_img)
        green_mask[aff_right_mask == 255] = [0, 255, 0]  # Green color
        
        # Overlay the green mask using addWeighted
        inpainting_img = cv2.addWeighted(inpainting_img, 1, green_mask, 0.5, 0)

    # Save the result
    if output_path:
        cv2.imwrite(output_path, inpainting_img)
        print(f"Saved overlay image to {output_path}")
    else:
        # Display the final image with overlays
        cv2.imshow('Overlayed Image', inpainting_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_folder(input_img_folder, input_mask_folder, output_folder):
    # Iterate through the subfolders in the input image folder
    for subfolder in os.listdir(input_img_folder):
        img_folder_path = os.path.join(input_img_folder, subfolder)
        mask_folder_path = os.path.join(input_mask_folder, subfolder)

        if not os.path.isdir(img_folder_path) or not os.path.isdir(mask_folder_path):
            continue

        # Check if the inpainting image exists in the image folder
        inpainting_img_path = os.path.join(img_folder_path, 'inpainting_cropped.png')
        if not os.path.exists(inpainting_img_path):
            print(f"Warning: inpainting.png not found in {img_folder_path}. Skipping.")
            continue

        # Initialize mask paths
        aff_left_path = os.path.join(mask_folder_path, 'aff_left.png') if os.path.exists(os.path.join(mask_folder_path, 'aff_left.png')) else None
        aff_right_path = os.path.join(mask_folder_path, 'aff_right.png') if os.path.exists(os.path.join(mask_folder_path, 'aff_right.png')) else None

        # Output path for the overlay image
        output_path = os.path.join(output_folder, subfolder, 'overlayed_image.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Overlay the masks and save the result
        overlay_masks_on_image(inpainting_img_path, aff_left_path, aff_right_path, output_path)
        img = cv2.imread(inpainting_img_path)
        heat_left = cv2.imread(os.path.join(mask_folder_path, 'aff_left_heat.png'))
        heat_left[heat_left[:,:,2] < 192] = [0, 0, 0]
        heat_left = cv2.addWeighted(img, 0.5, heat_left, 0.5, 0)
        cv2.imwrite(os.path.join(output_folder, subfolder, 'overlayed_aff_left.png'), heat_left)
        heat_right = cv2.imread(os.path.join(mask_folder_path, 'aff_right_heat.png'))
        heat_right = cv2.addWeighted(img, 0.5, heat_right, 0.5, 0)
        cv2.imwrite(os.path.join(output_folder, subfolder, 'overlayed_aff_right.png'), heat_right)

def main():
    # Set up argparse to handle input arguments
    parser = argparse.ArgumentParser(description="Overlay masks on inpainting images.")
    parser.add_argument('input_img_folder', type=str, help="Folder containing subfolders with inpainting.png images.")
    parser.add_argument('input_mask_folder', type=str, help="Folder containing subfolders with aff_left.png and aff_right.png masks.")
    parser.add_argument('output_folder', type=str, help="Folder where the result images will be saved.")

    args = parser.parse_args()

    # Process the given folders
    process_folder(args.input_img_folder, args.input_mask_folder, args.output_folder)

if __name__ == "__main__":
    main()
