import cv2
import math
import numpy as np
import os
import shutil
from argparse import ArgumentParser
from functools import partial
from funcy import lmap
from PIL import Image


def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename[0] != "P" and filename[0] != "f":
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

def rename_files_video(video_name):
    folder_path = os.path.join("../EPIC_DATA/frames", video_name.split("_")[0], video_name)
    for filename in os.listdir(folder_path):
        if filename[0] != "P" and filename[0] != "f":
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

def copy_prompt_files(folder_path, fac=10):
    files = os.listdir(folder_path)
    for file in files:
        filename = int(file.split('.')[0])
        for i in range(fac):
            new_file = str(filename - fac + i).zfill(7)
            shutil.copy(os.path.join(folder_path, file), os.path.join(folder_path, new_file + ".txt"))
            new_file = str(filename + fac - i).zfill(7)
            shutil.copy(os.path.join(folder_path, file), os.path.join(folder_path, new_file + ".txt"))

def move_ws_to_data_file(ws_path, dst_path):
    folders = os.listdir(ws_path)
    for folder in folders:
        path = os.path.join(ws_path, folder)
        sub_folders = os.listdir(path)
        for sub_folder in sub_folders:
            path_2 = os.path.join(path, sub_folder)
            sub_sub_folders = os.listdir(path_2)
            for sub_sub_folder in sub_sub_folders:
                path_3 = os.path.join(path_2, sub_sub_folder)
                content = os.listdir(path_3)
                if "masks" in content:
                    mask_path = os.path.join(path_3, "masks")
                    for mask in os.listdir(mask_path):
                        dst = os.path.join(dst_path, folder, sub_folder, sub_sub_folder)
                        if not os.path.exists(dst):
                            os.makedirs(dst)
                        shutil.copy(os.path.join(mask_path, mask), os.path.join(dst, mask))

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

def calculate_mask_difference(mask_prev, mask_post, out):
    mask_files_prev = os.listdir(mask_prev)
    mask_files_post = os.listdir(mask_post)
    if not os.path.exists(out):
        os.makedirs(out)
    for mask in mask_files_prev:
        if mask in mask_files_post:
            mask_1 = cv2.imread(os.path.join(mask_prev, mask), cv2.IMREAD_GRAYSCALE)
            mask_2 = cv2.imread(os.path.join(mask_post, mask), cv2.IMREAD_GRAYSCALE)
            result = mask_2 - mask_1
            result[np.where(result > 30)] = 255
            result[np.where(result <= 30)] = 0
            Image.fromarray(result).save(os.path.join(out, mask))

def regenerate_mask(folder_path, threshold):
    files = os.listdir(folder_path)
    for file in files:
        mask = cv2.imread(os.path.join(folder_path, file))
        mask[np.where(mask > threshold)] = 255
        mask[np.where(mask <= threshold)] = 0
        Image.fromarray(mask).save(os.path.join(folder_path, file))

def delete_empty_masks(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        mask = cv2.imread(os.path.join(folder_path, file))
        if np.all(mask == 0):
            os.remove(os.path.join(folder_path, file))


def extract_affordance_maximum(mask):
    indices = np.where(mask != 0)
    if len(indices[0]) == 0:
        return
    min_value = np.inf
    max_value = 0
    max_i = 0
    min_i = 0
    for i in range(len(indices[0])):
        prod = indices[0][i] * indices[1][i]
        if  prod < min_value:
            min_value = prod
            min_i = i
        if prod >= max_value:
            max_value = prod
            max_i = i
    index_1_min = indices[0][min_i]
    index_2_min = indices[1][min_i]
    index_1_max = indices[0][max_i]
    index_2_max = indices[1][max_i]
    index_1_av = (index_1_min + index_1_max)/2
    index_2_av = (index_2_min + index_2_max)/2
    perf_coords = np.array([index_1_av, index_2_av])
    dist = np.inf
    aff_point = 0
    for j in range(len(indices[0])):
        current_coords = np.array([indices[0][j], indices[1][j]])
        current_dist = np.linalg.norm(current_coords - perf_coords)
        if current_dist < dist:
            dist = current_dist
            aff_point = j
    mask[np.where(mask != 0)] = 0
    mask[indices[0][aff_point], indices[1][aff_point]] = 255
    return mask


def draw_affordances(image_path, mask_path, out):
    if not os.path.exists(out):
        os.makedirs(out)
    img_files = os.listdir(image_path)
    mask_files = os.listdir(mask_path)
    for img_file in img_files:
        frame = img_file.split('.')[0]
        mask_file = frame + ".png"
        if mask_file in mask_files:
            input_img = cv2.imread(os.path.join(image_path, img_file), 3)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            input_mask = cv2.imread(os.path.join(mask_path, mask_file))
            W = int(np.shape(input_img)[0] - np.shape(input_img)[0] % 8)
            H = int(np.shape(input_img)[1] - np.shape(input_img)[1] % 8)
            img = cv2.resize(input_img, (H, W))
            mask = cv2.resize(input_mask, (H, W))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = extract_affordance_maximum(gray_mask)
            output_path = os.path.join(out, frame + ".jpg")
            affordance = cv2.GaussianBlur(mask, (21, 21), 0)
            affordance = cv2.normalize(affordance, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_img = cv2.applyColorMap(affordance, cv2.COLORMAP_HOT)
            b, g, r = cv2.split(heatmap_img)
            heatmap_img = cv2.merge([r, g, b])
            vis = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
            Image.fromarray(vis).save(output_path)

def merge_affordances(image_path, mask_path_1, mask_path_2, out):
    if not os.path.exists(out):
        os.makedirs(out)
    img_files = os.listdir(image_path)
    mask_files_1 = os.listdir(mask_path_1)
    mask_files_2 = os.listdir(mask_path_2)
    for img_file in img_files:
        frame = img_file.split('.')[0]
        mask_file = frame + ".png"
        if mask_file in mask_files_1 and mask_file in mask_files_2:
            input_img = cv2.imread(os.path.join(image_path, img_file), 3)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            input_mask_1 = cv2.imread(os.path.join(mask_path_1, mask_file))
            input_mask_2 = cv2.imread(os.path.join(mask_path_2, mask_file))
            W = int(np.shape(input_img)[0] - np.shape(input_img)[0] % 8)
            H = int(np.shape(input_img)[1] - np.shape(input_img)[1] % 8)
            img = cv2.resize(input_img, (H, W))
            mask_1 = cv2.resize(input_mask_1, (H, W))
            mask_2 = cv2.resize(input_mask_2, (H, W))
            gray_mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
            gray_mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)
            aff_mask_1 = extract_affordance_maximum(gray_mask_1)
            aff_mask_2 = extract_affordance_maximum(gray_mask_2)
            mask = aff_mask_1 + aff_mask_2
            output_path = os.path.join(out, frame + ".jpg")
            affordance = cv2.GaussianBlur(mask, (21, 21), 0)
            affordance = cv2.normalize(affordance, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_img = cv2.applyColorMap(affordance, cv2.COLORMAP_HOT)
            b, g, r = cv2.split(heatmap_img)
            heatmap_img = cv2.merge([r, g, b])
            vis = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
            Image.fromarray(vis).save(output_path)
    

def draw_mask_affordance(images, masks_left, masks_right, out):
    image_files = os.listdir(images)
    mask_files_left = os.listdir(masks_left)
    mask_files_right = os.listdir(masks_right)
    if not os.path.exists(out):
        os.makedirs(out)
    for file in image_files:
        mask_file = file.split('.')[0] + '.png'
        if mask_file in mask_files_left and mask_file in mask_files_right:
            img = cv2.imread(os.path.join(images, file))
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            mask_left = cv2.imread(os.path.join(masks_left, file.split('.')[0] + '.png'))
            mask_right = cv2.imread(os.path.join(masks_right, file.split('.')[0] + '.png'))
            W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
            H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
            img = cv2.resize(img, (H, W))
            mask_left = cv2.resize(mask_left, (H, W))
            mask_right = cv2.resize(mask_right, (H, W))
            mask = mask_left + mask_right
            mask[np.where(mask != 0)] = 255
            vis = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            Image.fromarray(vis).save(os.path.join(out, file))

def dilate_whole_folder_structure(folder, dilate_fac):
    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        new_path = os.path.join(folder, sub_folder)
        dilate_masks(new_path, dilate_fac)

def recolor_whole_folder_structure(folder):
    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        new_path = os.path.join(folder, sub_folder, "masks")
        recolor_masks_white(new_path)
    
def copy_inpainted(folder_path, out):
    if not os.path.exists(out):
        os.makedirs(out)
    folders = os.listdir(folder_path)
    for folder in folders:
        file_folder = os.path.join(folder_path, folder, "out")
        files = os.listdir(file_folder)
        for file in files:
            shutil.copy(os.path.join(file_folder, file), os.path.join(out, file))
    
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
				out_folder = os.path.join(out, str(i * 4 * len(files) + k))
				if not os.path.exists(out_folder):
					os.makedirs(out_folder) 
				file_path_in = os.path.join(folder_path, file)
				file_path_out = os.path.join(out_folder, file)
				shutil.copy(file_path_in, file_path_out)

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
			print(file)
			file_path = os.path.join(folder_path, file)
			mask_path = os.path.join(mask_dir, file)
			shutil.move(file_path, mask_path)
			raw_file_name = file.split('.')[0] + ".jpg"
			raw_path = os.path.join(raw_dir, raw_file_name)
			raw_path_agent = os.path.join(new_raw_dir, raw_file_name)
			shutil.copy(raw_path, raw_path_agent)

def modify_folder_to_sequence(dir, ref_folder, out):
    if not os.path.exists(out):
        os.makedirs(out)
    ref_files = os.listdir(ref_folder)
    files = os.listdir(dir)
    for ref_file in ref_files:
        ref_file_number = int(ref_file.split('.')[0])
        for i in range(ref_file_number-9, ref_file_number+12):
            file = str(i).zfill(7) + ".png"
            if file in files:
                old_path = os.path.join(dir, file)
                new_folder = os.path.join(out, ref_file.split('.')[0])
                new_path = os.path.join(new_folder, file)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                shutil.copy(old_path, new_path)

def center_crop(img,shape):
    h,w = shape
    center = img.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    center_crop = img[int(y):int(y+h), int(x):int(x+w)]
    return center_crop

def maximal_crop_to_shape(image,shape,interpolation=cv2.INTER_AREA):
    target_aspect = shape[1]/shape[0]
    input_aspect = image.shape[1]/image.shape[0]
    if input_aspect > target_aspect:
        center_crop_shape = (image.shape[0],int(image.shape[0] * target_aspect))
    else:
        center_crop_shape = (int(image.shape[1] / target_aspect),image.shape[1])
    cropped = center_crop(image,center_crop_shape)
    resized = cv2.resize(cropped, (shape[1],shape[0]),interpolation=interpolation)
    return resized

def crop_masks(dir, out):
    if not os.path.exists(out):
        os.makedirs(out)
    files = os.listdir(dir)
    masks = np.ones((len(files), 480, 855, 3))
    for i, file in enumerate(files):
        mask_path = os.path.join(dir, file)
        mask = cv2.imread(mask_path)
        masks[i] = mask

    crop = lambda x: maximal_crop_to_shape(x,(256, 256))
    crop_ims = partial(lmap,crop)
    cropped_masks = crop_ims(masks)
    for i, ma in enumerate(cropped_masks):
        cv2.imwrite(os.path.join(out, files[i]), ma.astype(np.uint8))

def merge_mask_folders(dir, out):
    mask_path_left = os.path.join(dir, "left")
    mask_path_right = os.path.join(dir, "right")
    masks_left = os.listdir(mask_path_left)
    masks_right = os.listdir(mask_path_right)
    if not os.path.exists(out):
        os.makedirs(out)
    
    for mask in masks_left:
        if mask in masks_right:
            mask_left = cv2.imread(os.path.join(mask_path_left, mask), cv2.IMREAD_GRAYSCALE)
            mask_right = cv2.imread(os.path.join(mask_path_right, mask), cv2.IMREAD_GRAYSCALE)
            merged_mask = mask_left + mask_right
            merged_mask[np.where(merged_mask != 0)] = 255
            Image.fromarray(merged_mask).save(os.path.join(out, mask))
        else:
            mask_left = cv2.imread(os.path.join(mask_path_left, mask), cv2.IMREAD_GRAYSCALE)
            mask_left[np.where(mask_left != 0)] = 255
            Image.fromarray(mask_left).save(os.path.join(out, mask))
            pass
    
    for mask in masks_right:
        if not mask in masks_left:
            mask_right = cv2.imread(os.path.join(mask_path_right, mask), cv2.IMREAD_GRAYSCALE)
            mask_right[np.where(mask_right != 0)] = 255
            Image.fromarray(mask_right).save(os.path.join(out, mask))
            pass

def erode_masks(dir, erode_fac):
    files = os.listdir(dir)
    for file in files:
        mask = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.erode(
            mask,
            np.ones((erode_fac, erode_fac), np.uint8),
        )
        img_mask = Image.fromarray(mask)
        img_mask.save(os.path.join(dir, file))
"""
def modify_folder_for_batch(dir, out):
    if not os.path.exists(out):
        os.makedirs(out)
    
    folders = os.listdir(dir)
    batch = 0
    counter = 0
    for folder in folders:
        folder_path = os.path.join(dir, folder)
        new_path = os.path.join(out, str(batch), str(counter))
        shutil.copytree(folder_path, new_path)
        counter += 1
        if counter >= batch_size:
            batch += 1
            counter = 0
"""
def modify_folder_for_batch(dirs, raw_dir, out):
    if not os.path.exists(out):
        os.makedirs(out)
    folders = os.listdir(dirs)
    for folder in folders:
        folder_path = os.path.join(dirs, folder)
        files = os.listdir(folder_path)
        mask_dir = os.path.join(out, folder, "masks")
        new_raw_dir = os.path.join(out, folder, "raw")
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        if not os.path.exists(new_raw_dir):
            os.makedirs(new_raw_dir)
        files = [f for f in files if f != "masks" and f != "raw"]
        for file in files:
            print(file)
            file_path = os.path.join(folder_path, file)
            mask_path = os.path.join(mask_dir, file)
            shutil.copy(file_path, mask_path)
            raw_file_name = file.split('.')[0] + ".jpg"
            raw_path = os.path.join(raw_dir, raw_file_name)
            raw_path_agent = os.path.join(new_raw_dir, raw_file_name)
            shutil.copy(raw_path, raw_path_agent)

def batchify_migan(dir, batch_size, out):
    if not os.path.exists(out):
        os.makedirs(out)
    files = os.listdir(dir)
    counter = 0
    batch = 0
    for file in files:
        file_path = os.path.join(dir, file)
        new_path = os.path.join(out, str(batch))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_file_path = os.path.join(new_path, file)
        shutil.copy(file_path, new_file_path)
        if counter >= batch_size:
            batch += 1
            counter = 0
        counter += 1

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
        new_path = os.path.join(out, str(batch))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_folder_path = os.path.join(new_path, folder)
        shutil.copytree(folder_path, new_folder_path)
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

def apply_padding_single_folder(dir, out):
    frames = os.listdir(dir)
    if not os.path.exists(out):
        os.makedirs(out)
    for frame in frames:
        frame_path = os.path.join(dir, frame)
        img = cv2.imread(frame_path)
        max_shape = max([img.shape[0], img.shape[1]])
        padded_img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
        Image.fromarray(padded_img).save(os.path.join(out, frame))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rename_files', default=None)
    parser.add_argument('--rename_files_video', default=None)
    parser.add_argument('--recolor_masks_white', default=None)
    parser.add_argument('--copy_prompt_files', default=None)
    parser.add_argument('--move_ws_to_data_file_ws', default=None)
    parser.add_argument('--move_ws_to_data_file_dst', default=None)
    parser.add_argument('--dilate_masks_dir', default=None)
    parser.add_argument('--dilate_masks_fac', default=10, type=int)
    parser.add_argument('--calculate_mask_difference_pre', default=None)
    parser.add_argument('--calculate_mask_difference_post', default=None)
    parser.add_argument('--calculate_mask_difference_out', default=None)
    parser.add_argument('--regenerate_mask_src', default=None)
    parser.add_argument('--regenerate_mask_threshold', default=30, type=int)
    parser.add_argument('--draw_affordances_img', default=None)
    parser.add_argument('--draw_affordances_mask', default=None)
    parser.add_argument('--draw_affordances_dst', default=None)
    parser.add_argument('--delete_empty_masks', default=None)
    parser.add_argument('--merge_affordances_images', default=None)
    parser.add_argument('--merge_affordances_folder_left', default=None)
    parser.add_argument('--merge_affordances_folder_right', default=None)
    parser.add_argument('--merge_affordances_out', default=None)
    parser.add_argument('--draw_mask_affordance_images', default=None)
    parser.add_argument('--draw_mask_affordance_masks_left', default=None)
    parser.add_argument('--draw_mask_affordance_masks_right', default=None)
    parser.add_argument('--draw_mask_affordance_out', default=None)
    parser.add_argument('--dilate_whole_folder_structure', default=None)
    parser.add_argument('--copy_inpainted_dir', default=None)
    parser.add_argument('--copy_inpainted_out', default=None)
    parser.add_argument('--recolor_whole_folder_structure', default=None)
    parser.add_argument('--restructure_folder_for_agent_inpaint_folder', default=None)
    parser.add_argument('--restructure_folder_for_agent_inpaint_out', default=None)
    parser.add_argument('--fill_zeros_files', default=None)
    parser.add_argument('--add_raw_to_dir_dir', default=None)
    parser.add_argument('--add_raw_to_dir_raw', default=None)
    parser.add_argument('--modify_folder_to_sequence_dir', default=None)
    parser.add_argument('--modify_folder_to_sequence_ref', default=None)
    parser.add_argument('--modify_folder_to_sequence_out', default=None)
    parser.add_argument('--crop_masks_dir', default=None)
    parser.add_argument('--crop_masks_out', default=None)
    parser.add_argument('--merge_mask_folders_dir', default=None)
    parser.add_argument('--merge_mask_folders_out', default=None)
    parser.add_argument('--erode_masks_dir', default=None)
    parser.add_argument('--erode_masks_fac', default=5, type=int)
    parser.add_argument('--modify_folder_for_batch_out', default=None)
    parser.add_argument('--modify_folder_for_batch_dirs', default=None)
    parser.add_argument('--modify_folder_for_batch_raw_dir', default=None)
    parser.add_argument('--batchify_migan_dir', default=None)
    parser.add_argument('--batchify_migan_out', default=None)
    parser.add_argument('--batchify_migan_batchsize', default=None, type=int)
    parser.add_argument('--batchify_agent_inp_dirs', default=None)
    parser.add_argument('--batchify_agent_inp_out', default=None)
    parser.add_argument('--batchify_agent_inp_batchsize', default=16, type=int)
    parser.add_argument('--apply_padding_dir', default=None)
    parser.add_argument('--apply_padding_out', default=None)
    parser.add_argument('--apply_padding_single_folder_dir', default=None)
    parser.add_argument('--apply_padding_single_folder_out', default=None)

args = parser.parse_args()
vals = vars(args)
if vals["rename_files"] != None:
    print("Starting Renaming Process")
    folder_path = vals["rename_files"]
    rename_files(folder_path)
    print("Finished Renaming Proccess")
elif vals["rename_files_video"] != None:
    print("Starting Renaming Process")
    video_name = vals["rename_files_video"]
    rename_files_video(video_name)
    print("Finished Renaming Proccess")
elif vals["recolor_masks_white"] != None:
    print("Recoloring masks white at: ", vals["recolor_masks_white"])
    folder_path = vals["recolor_masks_white"]
    recolor_masks_white(folder_path)
    print("Done")
elif vals["copy_prompt_files"] != None:
    folder_path = vals["copy_prompt_files"]
    copy_prompt_files(folder_path)
elif vals["move_ws_to_data_file_ws"] != None and vals["move_ws_to_data_file_dst"] != None:
    ws_path = vals["move_ws_to_data_file_ws"]
    dst_path = vals["move_ws_to_data_file_dst"]
    move_ws_to_data_file(ws_path, dst_path)
elif vals["dilate_masks_dir"] != None:
    print("Dilating masks at: ", vals["dilate_masks_dir"])
    folder_path = vals["dilate_masks_dir"]
    dilate_fac = int(vals["dilate_masks_fac"])
    dilate_masks(folder_path, dilate_fac)
    print("Done")
elif vals["calculate_mask_difference_pre"] != None and vals["calculate_mask_difference_post"] != None and vals["calculate_mask_difference_out"] != None:
    print("Creating mask differences")
    mask_prev = vals["calculate_mask_difference_pre"]
    mask_post = vals["calculate_mask_difference_post"]
    out = vals["calculate_mask_difference_out"]
    calculate_mask_difference(mask_prev, mask_post, out)
    print("Done")
elif vals["regenerate_mask_src"] != None:
    print("Regenerating masks")
    folder_path = vals["regenerate_mask_src"]
    threshold = vals["regenerate_mask_threshold"]
    regenerate_mask(folder_path, threshold)
    print("Done")
elif vals["draw_affordances_img"] != None and vals["draw_affordances_mask"] != None and vals["draw_affordances_dst"] != None:
    print("Drawing affordances")
    image_path = vals["draw_affordances_img"]
    mask_path = vals["draw_affordances_mask"]
    out = vals["draw_affordances_dst"]
    draw_affordances(image_path, mask_path, out)
    print("Done")
elif vals["delete_empty_masks"] != None:
    print("Deleting empty masks")
    folder_path = vals["delete_empty_masks"]
    delete_empty_masks(folder_path)
    print("Done")
elif vals["merge_affordances_images"] != None and vals["merge_affordances_folder_left"] != None and vals["merge_affordances_folder_right"] != None and vals["merge_affordances_out"] != None:
    print("Start affordance merging process")
    image_path = vals["merge_affordances_images"]
    mask_path_1 = vals["merge_affordances_folder_left"]
    mask_path_2 = vals["merge_affordances_folder_right"]
    out = vals["merge_affordances_out"]
    merge_affordances(image_path, mask_path_1, mask_path_2, out)
    print("Done")
elif vals["draw_mask_affordance_images"] != None and vals["draw_mask_affordance_masks_left"] != None and vals["draw_mask_affordance_masks_right"] != None and vals["draw_mask_affordance_out"] != None:
    images = vals["draw_mask_affordance_images"]
    masks_left = vals["draw_mask_affordance_masks_left"]
    masks_right = vals["draw_mask_affordance_masks_right"]
    out = vals["draw_mask_affordance_out"]
    draw_mask_affordance(images, masks_left, masks_right, out)
elif vals["dilate_whole_folder_structure"] != None:
    folder = vals["dilate_whole_folder_structure"]
    dilate_fac = int(vals["dilate_masks_fac"])
    dilate_whole_folder_structure(folder)
elif vals["copy_inpainted_dir"] != None and vals["copy_inpainted_out"] != None:
    folder_path = vals["copy_inpainted_dir"] 
    out = vals["copy_inpainted_out"]
    copy_inpainted(folder_path, out)
elif vals["recolor_whole_folder_structure"] != None:
    folder = vals["recolor_whole_folder_structure"]
    recolor_whole_folder_structure(folder)
elif vals["restructure_folder_for_agent_inpaint_folder"] != None and vals["restructure_folder_for_agent_inpaint_out"] != None:
    dirs = vals["restructure_folder_for_agent_inpaint_folder"]
    out = vals["restructure_folder_for_agent_inpaint_out"]
    restructure_folder_for_agent_inpaint(dirs, out)
elif vals["fill_zeros_files"] != None:
    dirs = vals["fill_zeros_files"]
    fill_zeros_files(dirs) 
elif vals["add_raw_to_dir_dir"] != None and vals["add_raw_to_dir_raw"] != None:
    dirs = vals["add_raw_to_dir_dir"]
    raw_dir = vals["add_raw_to_dir_raw"]
    add_raw_to_dir(dirs, raw_dir)
elif vals["modify_folder_to_sequence_dir"] != None and vals["modify_folder_to_sequence_ref"] != None and vals["modify_folder_to_sequence_out"] != None:
    dir = vals["modify_folder_to_sequence_dir"]
    ref_folder = vals["modify_folder_to_sequence_ref"]
    out = vals["modify_folder_to_sequence_out"]
    modify_folder_to_sequence(dir, ref_folder, out)
elif vals["crop_masks_dir"] != None and vals["crop_masks_out"] != None:
    dir = vals["crop_masks_dir"]
    out = vals["crop_masks_out"]
    crop_masks(dir, out)
elif vals["merge_mask_folders_dir"] != None and vals["merge_mask_folders_out"] != None:
    dir = vals["merge_mask_folders_dir"]
    out = vals["merge_mask_folders_out"]
    merge_mask_folders(dir, out)
elif vals["erode_masks_dir"] != None:
    dir = vals["erode_masks_dir"]
    fac = vals["erode_masks_fac"]
    erode_masks(dir, fac)
    """
elif vals["modify_folder_for_batch_dir"] != None and vals["modify_folder_for_batch_out"] != None:
    dir = vals["modify_folder_for_batch_dir"]
    out = vals["modify_folder_for_batch_out"]
    batch_size = vals["modify_folder_for_batch_batch_size"]
    modify_folder_for_batch(dir, batch_size, out)
    """
elif vals["modify_folder_for_batch_dirs"] != None and vals["modify_folder_for_batch_raw_dir"] != None and vals["modify_folder_for_batch_out"] != None:
    dirs = vals["modify_folder_for_batch_dirs"]
    out = vals["modify_folder_for_batch_out"]
    raw_dir = vals["modify_folder_for_batch_raw_dir"]
    modify_folder_for_batch(dirs, raw_dir, out)
elif vals["batchify_migan_dir"] != None and vals["batchify_migan_out"] != None and vals["batchify_migan_batchsize"] != None:
    batchsize = vals["batchify_migan_batchsize"]
    dir = vals["batchify_migan_dir"]
    out = vals["batchify_migan_out"]
    batchify_migan(dir, batchsize, out)
elif vals["batchify_agent_inp_dirs"] != None and vals["batchify_agent_inp_out"] != None:
    dirs = vals["batchify_agent_inp_dirs"]
    out = vals["batchify_agent_inp_out"]
    batchsize = vals["batchify_agent_inp_batchsize"]
    batchify_agent_inp(dirs, out, batchsize)
elif vals["apply_padding_dir"] != None and vals["apply_padding_out"] != None:
    dir = vals["apply_padding_dir"]
    out = vals["apply_padding_out"]
    apply_padding(dir, out)
elif vals["apply_padding_single_folder_dir"] != None and vals["apply_padding_single_folder_out"] != None:
    dir = vals["apply_padding_single_folder_dir"]
    out = vals["apply_padding_single_folder_out"]
    apply_padding_single_folder(dir, out)
else:
    print("wrong arguments")
