import cv2
import math
import numpy as np
import os
import shutil
from argparse import ArgumentParser
from functools import partial
from funcy import lmap
from PIL import Image

def rename_files_video(folder_path):
    #folder_path = os.path.join("../EPIC_DATA/frames", video_name.split("_")[0], video_name)
    for filename in os.listdir(folder_path):
        if filename[0] != "P" and filename[0] != "f" and len(filename) < 12:
            print("files already configured")
            return
        # Split filename by "_"
        parts = filename.split("_")
        
        # Search for the split string with at least three "0"
        for i, part in enumerate(parts):
            if part.count("0") >= 3:
                # Remove three "0" from the string
                new_part = part.replace("0", "", 3)
                # Merge parts back into filename
                new_filename = new_part
                
                # Rename the file
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed {filename} to {new_filename}")
                break

def recolor_masks_white(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        img[np.where(img != 0)] = 255
        Image.fromarray(img).save(os.path.join(folder_path, file))

def dilate_masks(folder_path, dilate_fac):
    files = os.listdir(folder_path)
    for file in files:
        mask = cv2.imread(os.path.join(folder_path, file))
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_fac, dilate_fac), np.uint8),
            iterations=1
        )
        img_mask = Image.fromarray(mask, 'RGB')
        img_mask.save(os.path.join(folder_path, file))

def pad_image(image):
    h, w = image.shape[:2]
    if h > w:
        padding = (h - w, 0)
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding[0], 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padding = (w - h, 0)
        padded_image = cv2.copyMakeBorder(image, padding[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def delete_empty_masks(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        mask = cv2.imread(os.path.join(folder_path, file))
        if np.all(mask == 0):
            os.remove(os.path.join(folder_path, file))

def recolor_whole_folder_structure(folder):
    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        new_path = os.path.join(folder, sub_folder, "masks")
        recolor_masks_white(new_path)

def restructure_folder_for_agent_inpaint(dirs, out):
	folders = os.listdir(dirs)
	folders.sort()
	max_len = int(math.floor(len(folders)/4))
	if not os.path.exists(out):
		os.makedirs(out)
	for i in range(max_len):
		for j in range(4):
			folder_name = folders[i * 4 + j]
			folder_path = os.path.join(dirs, folder_name)
			files = os.listdir(folder_path)
			files.sort()
			for k, file in enumerate(files):
				out_folder = os.path.join(out, str(i * 4 * len(files) + k).zfill(9))
				if not os.path.exists(out_folder):
					os.makedirs(out_folder) 
				file_path_in = os.path.join(folder_path, file)
				file_path_out = os.path.join(out_folder, file)
				shutil.move(file_path_in, file_path_out)

def delete_empty_folders(root_dir):
    # Walk the directory tree from bottom to top (topdown=False)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)
            # If the folder is empty, delete it
            if not os.listdir(folder_path):  # Check if the directory is empty
                os.rmdir(folder_path)
                print(f"Deleted empty folder: {folder_path}")

def fill_zeros_files(dirs):
    folders = os.listdir(dirs)
    for folder in folders:
        folder_path = os.path.join(dirs, folder)
        new_name = folder.zfill(7)
        new_path = os.path.join(dirs, new_name)
        os.rename(folder_path, new_path)

def add_raw_to_dir(dirs, raw_dir):
	folders = os.listdir(dirs)
	for folder in folders:
		folder_path = os.path.join(dirs, folder)
		files = os.listdir(folder_path)
		mask_dir = os.path.join(folder_path, "masks")
		new_raw_dir = os.path.join(folder_path, "raw")
		if not os.path.exists(mask_dir):
			os.mkdir(mask_dir)
		if not os.path.exists(new_raw_dir):
			os.mkdir(new_raw_dir)
		files = [f for f in files if f != "masks" and f != "raw"]
		for file in files:
			file_path = os.path.join(folder_path, file)
			mask_path = os.path.join(mask_dir, file)
			shutil.move(file_path, mask_path)
			raw_file_name = file.split('.')[0] + ".jpg"
			raw_path = os.path.join(raw_dir, raw_file_name)
			raw_path_agent = os.path.join(new_raw_dir, raw_file_name)
			shutil.copy(raw_path, raw_path_agent)

def batchify_agent_inp(dirs, out, batch_size):
    if not os.path.exists(out):
        os.makedirs(out)
    folders = os.listdir(dirs)
    counter = 0
    batch = 0
    for folder in folders:
        if counter >= batch_size:
            counter = 0
            batch += 1
        folder_path = os.path.join(dirs, folder)
        new_path = os.path.join(out, str(batch).zfill(7))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_folder_path = os.path.join(new_path, folder)
        shutil.move(folder_path, new_folder_path)
        counter += 1

def apply_padding(dirs, out):
    folders = os.listdir(dirs)
    if not os.path.exists(out):
        os.makedirs(out)
    for folder in folders:
        folder_path = os.path.join(dirs, folder)
        frames = os.listdir(os.path.join(folder_path, "raw"))
        masks = os.listdir(os.path.join(folder_path, "masks"))
        out_folder_path = os.path.join(out, folder)
        if not os.path.exists(out_folder_path):
            os.makedirs(os.path.join(out_folder_path, "raw"))
            os.makedirs(os.path.join(out_folder_path, "masks"))
        for frame in frames:
            frame_name = frame.split('.')[0]
            frame_path = os.path.join(folder_path, "raw", frame)
            mask_path = os.path.join(folder_path, "masks", frame_name + ".png")
            img = cv2.imread(frame_path)
            max_shape = max([img.shape[0], img.shape[1]])
            padded_img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
            Image.fromarray(padded_img).save(os.path.join(out_folder_path, "raw", frame))
            mask = cv2.imread(mask_path)
            max_shape_mask = max([mask.shape[0], mask.shape[1]])
            padded_mask = cv2.copyMakeBorder(mask, max_shape_mask - mask.shape[0], 0, max_shape_mask - mask.shape[1], 0, cv2.BORDER_CONSTANT)
            Image.fromarray(padded_mask).save(os.path.join(out_folder_path, "masks", frame_name + ".png"))

def apply_padding_single_folder(directory):
    category_list = ['left', 'right']
    for category in category_list:
        subdirectory = os.path.join(directory, category)
        frames = os.listdir(subdirectory)
        for frame in frames:
            frame_path = os.path.join(subdirectory, frame)
            img = cv2.imread(frame_path)
            max_shape = max([img.shape[0], img.shape[1]])
            padded_img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
            Image.fromarray(padded_img).save(frame_path)

def dilate_and_recolor(directory, dilation_fac):
    category_list = ['hand/both', 'hand/left', 'hand/right', 'object/left', 'object/right']
    dilate_masks(os.path.join(directory, category_list[0]), dilation_fac)
    for category in category_list:
        recolor_masks_white(os.path.join(directory, category))

def preprocess_for_agent_inpainting(directory, raw_directory, batchsize):
    fill_zeros_files(directory)
    restructure_folder_for_agent_inpaint(directory, directory)
    delete_empty_folders(directory)
    add_raw_to_dir(directory, raw_directory)
    recolor_whole_folder_structure(directory)
    apply_padding(directory, directory)
    batchify_agent_inp(directory, directory, batchsize)

def modify_folder_to_sequence(dir, ref_folder, out):
    category_list = ['left', 'right']
    for category in category_list:
        sub_out = os.path.join(out, category)
        sub_dir = os.path.join(dir, category)
        if not os.path.exists(sub_out):
            os.makedirs(sub_out)
        ref_files = os.listdir(ref_folder)
        files = os.listdir(sub_dir)
        for ref_file in ref_files:
            ref_file_number = int(ref_file.split('.')[0])
            for i in range(ref_file_number-9, ref_file_number+12):
                file = str(i).zfill(7) + ".png"
                if file in files:
                    old_path = os.path.join(sub_dir, file)
                    new_folder = os.path.join(sub_out, ref_file.split('.')[0])
                    new_path = os.path.join(new_folder, file)
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    shutil.copy(old_path, new_path)

def preprocess_for_mask_completion(folder1, folder2, output_folder):
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

def extract_affordances(completed_masks_directory, hand_masks_directory, aff_masks_directory):
    categories = ['left', 'right']
    for category in categories:
        completed_masks_subdirectory = os.path.join(completed_masks_directory, category)
        hand_masks_subdirectory = os.path.join(hand_masks_directory, category)
        aff_masks_subdirectory = os.path.join(aff_masks_directory, category)
        if not os.path.exists(aff_masks_subdirectory):
            os.makedirs(aff_masks_subdirectory)
        for file_name in os.listdir(completed_masks_subdirectory):
            # Generate file paths
            aff_mask_path = os.path.join(aff_masks_subdirectory, file_name)
            completed_mask_path = os.path.join(completed_masks_subdirectory, file_name)
            hand_mask_path = os.path.join(hand_masks_subdirectory, file_name)

            # Ensure the corresponding file exists in both folders
            if not os.path.isfile(hand_mask_path):
                print(f"Skipping {file_name}: No corresponding file in {hand_masks_subdirectory}")
                continue

            # Read the masks
            completed_mask = cv2.imread(completed_mask_path, cv2.IMREAD_GRAYSCALE)
            hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)


            if completed_mask is None or hand_mask is None:
                print(f"Skipping {file_name}: Could not read one or both masks.")
                continue

            hand_mask = pad_image(hand_mask)

            # Ensure the masks have the same dimensions
            if completed_mask.shape != hand_mask.shape:
                print(f"Resizing {file_name} to match dimensions of {hand_mask_path}.")
                hand_mask = cv2.resize(hand_mask, (completed_mask.shape[1], completed_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Calculate the overlapping region
            overlap = cv2.bitwise_and(completed_mask, hand_mask)

            # Save the updated mask back to the mask_differences folder
            cv2.imwrite(aff_mask_path, overlap)
            print(f"Updated mask saved: {aff_mask_path}")

def process_affordances(affordance_directory, dilation_fac):
    categories = ['left', 'right']
    for category in categories:
        affordance_subdirectory = os.path.join(affordance_directory, category)
        delete_empty_masks(affordance_subdirectory)
        dilate_masks(affordance_subdirectory, dilation_fac)
        recolor_masks_white(affordance_subdirectory)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rename_files_video', nargs=1, metavar=('PATH'), help='path to video')
    parser.add_argument('--dilate_and_recolor', nargs=2, metavar=('PATH', 'NUMBER'), help='path to propagated hand and object masks and dilation factor')
    parser.add_argument('--preprocess_for_agent_inpainting', nargs=3, metavar=('PATH', 'PATH', 'NUMBER'), help='path to hand masks, path to rgb images and batchsize')
    parser.add_argument('--preprocess_for_mask_completion', nargs=3, metavar=('PATH', 'PATH', 'PATH'), help='path to frames, path to inpainted hands and path to output directory')
    parser.add_argument('--extract_affordances', nargs=3, metavar=('PATH', 'PATH', 'PATH'), help='path to the completed masks, path to the hand masks as reference and out directory to store affordance masks')
    parser.add_argument('--process_affordances', nargs=2, metavar=('PATH', 'NUMBER'), help='path to affordance masks and dilation factor')
    parser.add_argument('--modify_folder_to_sequence', nargs=3, metavar=('PATH', 'PATH', 'PATH'), help='path to hand masks, reference json folder, out directory to store sequences')
    parser.add_argument('--apply_padding', nargs=1, metavar=('PATH'), help='path to the object masks')


args = parser.parse_args() 
vals = vars(args)
if args.rename_files_video:
    rename_files_video(*args.rename_files_video)
elif args.dilate_and_recolor:
    args.dilate_and_recolor[1] = int(args.dilate_and_recolor[1])
    dilate_and_recolor(*args.dilate_and_recolor)
elif args.preprocess_for_agent_inpainting:
    args.preprocess_for_agent_inpainting[2] = int(args.preprocess_for_agent_inpainting[2])
    preprocess_for_agent_inpainting(*args.preprocess_for_agent_inpainting)
elif args.preprocess_for_mask_completion:
    preprocess_for_mask_completion(*args.preprocess_for_mask_completion)
elif args.extract_affordances:
    extract_affordances(*args.extract_affordances)
elif args.process_affordances:
    args.process_affordances[1] = int(args.process_affordances[1])
    process_affordances(*args.process_affordances)
elif args.modify_folder_to_sequence:
    modify_folder_to_sequence(*args.modify_folder_to_sequence)
elif args.apply_padding:
    apply_padding_single_folder(*args.apply_padding)
else:
    print("Wrong Arguments!")
