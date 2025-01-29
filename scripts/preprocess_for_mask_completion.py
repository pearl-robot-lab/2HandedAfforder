import os
import cv2
import numpy as np
import argparse
from shutil import copy2

def pad_image(image):
    h, w = image.shape[:2]
    if h > w:
        padding = (h - w, 0)
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding[0], 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padding = (w - h, 0)
        padded_image = cv2.copyMakeBorder(image, padding[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def process_folder(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = {os.path.splitext(f)[0]: f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {os.path.splitext(f)[0]: f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

    common_files = set(files1.keys()) & set(files2.keys())

    for filename in common_files:
        path1 = os.path.join(folder1, files1[filename])
        path2 = os.path.join(folder2, files2[filename])

        # Read and process images
        img1 = cv2.imread(path1)
        img1 = pad_image(img1)
        img2 = cv2.imread(path2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Create a subfolder in the output folder
        output_subfolder = os.path.join(output_folder, filename)
        os.makedirs(output_subfolder, exist_ok=True)

        # Save the processed images in the output folder
        output_img1 = os.path.join(output_subfolder, '1.png')
        output_img2 = os.path.join(output_subfolder, '2.png')

        cv2.imwrite(output_img1, img1)
        cv2.imwrite(output_img2, img2)

def main():
    parser = argparse.ArgumentParser(description="Process images from two folders and organize them in an output folder.")
    parser.add_argument('folder1', type=str, help="Path to frames")
    parser.add_argument('folder2', type=str, help="Path to inpainted")
    parser.add_argument('output_folder', type=str, help="Path to the output folder.")

    args = parser.parse_args()

    process_folder(args.folder1, args.folder2, args.output_folder)

if __name__ == "__main__":
    main()
