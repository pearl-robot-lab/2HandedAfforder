import os
import cv2
from argparse import ArgumentParser

def process_images_in_folder(folder_path):
    # Define paths for inpainted_frame.png and list other .png files
    inpainted_frame_path = os.path.join(folder_path, 'inpainted_frame.png')
    
    # Verify inpainted_frame.png exists
    if not os.path.exists(inpainted_frame_path):
        print("inpainted_frame.png not found in the specified folder.")
        return
    
    # List all other .png files in the folder
    other_png_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f != 'inpainted_frame.png']
    
    # Exit if no other .png files are found
    if not other_png_files:
        print("No other .png files found in the specified folder.")
        return
    
    # Read the first other PNG file to get its dimensions
    sample_image_path = os.path.join(folder_path, other_png_files[0])
    sample_image = cv2.imread(sample_image_path)
    target_size = (sample_image.shape[1], sample_image.shape[0])  # Width, Height
    
    # Resize inpainted_frame.png to match the target size
    inpainted_frame = cv2.imread(inpainted_frame_path)
    resized_inpainted_frame = cv2.resize(inpainted_frame, target_size)
    cv2.imwrite(inpainted_frame_path, resized_inpainted_frame)
    
    # Convert each other .png file to grayscale and save
    for png_file in other_png_files:
        file_path = os.path.join(folder_path, png_file)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(file_path, gray_image)
    
    print("Processing complete. Resized inpainted_frame.png and converted other .png files to grayscale.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('video', default=None)
    args = parser.parse_args()
    vals = vars(args)
    if vals['video']:
        sequences = os.listdir(vals['video'])
        for sequence in sequences:
            sequence_path = os.path.join(vals['video'], sequence)
            process_images_in_folder(sequence_path)
