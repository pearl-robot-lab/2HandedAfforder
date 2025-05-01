import cv2
import h5py
import matplotlib.pyplot as plt
import monai
import numpy as np 
import os
import time
import torch
import wandb

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from argparse import ArgumentParser
import json
from PIL import Image, ImageDraw, ImageFont
from statistics import mean
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from segment_anything_copy import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from segment_anything_copy.modeling import Sam
from transformers import SamProcessor
from typing import Optional, Tuple
from aff_model_new import TwoHandedAfforder
from dataset_new import AffDataset
from json_mask_handler import get_masks_and_overlay, recreate_mask_from_contours

def prepare_data(data_folder):
    #h5_folder = os.path.join(data_folder, 'h5')
    #json_folder = os.path.join(data_folder, 'jsons')
    #h5_files = sorted(os.listdir(h5_folder))
    #json_files = sorted(os.listdir(json_folder))
    h5_files = os.listdir(data_folder)
    images = []
    aff_masks_left = []
    aff_masks_right = []
    taxonomies = []
    text_prompts = []
    for file in h5_files:
        with h5py.File(os.path.join(data_folder, file), "r") as f:
            a_group_key = list(f.keys())[0]
            text = np.array(f[a_group_key]["narration"])
            imgs = np.array(f[a_group_key]["inpainted"])
            aff_left = np.array(f[a_group_key]["aff_left"])
            aff_right = np.array(f[a_group_key]["aff_right"])
            taxonomy = np.array(f[a_group_key]["taxonomy"])
            if imgs.size == 0:
                continue
            images.append(imgs)
            taxonomies.append(taxonomy)
            text_prompts.extend(text.tolist())
            aff_masks_left.append(aff_left)
            aff_masks_right.append(aff_right)
    """
    original_size = None
    for file in json_files:
      json_path = os.path.join(json_folder, file)
      with open(json_path, 'r') as f:           
        json_data = json.load(f)
        if original_size is None:
          original_size = json_data["0"]["original_size"]
        for key in json_data:
          entry = json_data[key]
          aff_masks_left.append(recreate_mask_from_contours(entry.get("aff_left", [])))
          aff_masks_right.append(recreate_mask_from_contours(entry.get("aff_right", [])))
    """
    images_arr = np.concatenate(images)
    aff_masks_left = np.concatenate(aff_masks_left)
    aff_masks_right = np.concatenate(aff_masks_right)
    taxonomies_arr = np.concatenate(taxonomies)
    return images_arr, aff_masks_left, aff_masks_right, text_prompts, taxonomies_arr


class InferenceHandler():
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([1, 1, 1])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def test(self, data_folder, out):
        self.model.to(self.device)
        images_arr, aff_masks_left, aff_masks_right, text_prompts, taxonomies_arr = prepare_data(data_folder)
        with torch.no_grad():
            for i in range(images_arr.shape[0]):
                img = images_arr[i]
                prompt = text_prompts[i]
                gt_mask_left = aff_masks_left[i]
                gt_mask_right = aff_masks_right[i]
                gt_taxonomy = taxonomies_arr[i]

                inputs = self.processor(img, input_boxes=None, return_tensors="pt").to(self.device)
                inputs["image"] = inputs["pixel_values"].squeeze(0)
                inputs["txt_prompt"] = [prompt]
                inputs["original_size"] = (inputs["original_sizes"][0][0], inputs["original_sizes"][0][1])
                self.model.eval()
                outputs = self.model(inputs)

                if not os.path.exists(os.path.join(out, "predictions")):
                    os.makedirs(os.path.join(out, "predictions"))
                if not os.path.exists(os.path.join(out, "gt")):
                    os.makedirs(os.path.join(out, "gt"))
                if not os.path.exists(os.path.join(out, "prompts")):
                    os.makedirs(os.path.join(out, "prompts"))

                if torch.argmax(outputs[0]["taxonomy"]) != 1:
                    sam_seg_prob_left = torch.sigmoid(outputs[0]["masks_left"].squeeze(1))
                    sam_seg_prob_left = sam_seg_prob_left.cpu().numpy().squeeze()
                    sam_seg_left = (sam_seg_prob_left > 0.5).astype(np.uint8)
                    if torch.argmax(outputs[0]["taxonomy"]) == 0:
                        sam_seg_right = np.zeros_like(sam_seg_left)
                if torch.argmax(outputs[0]["taxonomy"]) != 0:
                    sam_seg_prob_right = torch.sigmoid(outputs[0]["masks_right"].squeeze(1))
                    sam_seg_prob_right = sam_seg_prob_right.cpu().numpy().squeeze()
                    sam_seg_right = (sam_seg_prob_right > 0.5).astype(np.uint8)
                    if torch.argmax(outputs[0]["taxonomy"]) == 1:
                        sam_seg_left = np.zeros_like(sam_seg_right)

                sam_seg = np.logical_or(sam_seg_left, sam_seg_right).astype(np.uint8)

                if np.argmax(gt_taxonomy) == 0:
                    mask = gt_mask_left
                elif np.argmax(gt_taxonomy) == 1:
                    mask = gt_mask_right
                else:
                    mask = np.logical_or(gt_mask_left, gt_mask_right).astype(np.uint8)

                out_path_pred = os.path.join(out, "predictions", str(i) +  ".jpg")
                out_path_gt = os.path.join(out, "gt", str(i) + ".jpg")
                prompt_path = os.path.join(out, "prompts", str(i) + ".txt")
                aff_mask = sam_seg
                print("Aff Mask Shape: ", aff_mask.shape)
                aff_mask = cv2.cvtColor(sam_seg, cv2.COLOR_GRAY2BGR)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                self.save_result(img, aff_mask, prompt, out_path_pred)
                self.save_result(img, mask, prompt, out_path_gt)
                if not isinstance(prompt, str):
                    prompt = prompt.decode('utf-8')
                with open(prompt_path, 'w') as file:
                  file.write(prompt)

    def test_on_single(self, img, prompt, th=0.5):
        with torch.no_grad():
            self.model.to(self.device)
            inputs = self.processor(img, input_boxes=None, return_tensors="pt").to(self.device)
            inputs["image"] = inputs["pixel_values"].squeeze(0)
            inputs["txt_prompt"] = [prompt]
            inputs["original_size"] = (inputs["original_sizes"][0][0], inputs["original_sizes"][0][1])
            self.model.eval()
            outputs = self.model(inputs)

            if torch.argmax(outputs[0]["taxonomy"]) != 1:
                sam_seg_prob_left = torch.sigmoid(outputs[0]["masks_left"].squeeze(1))
                sam_seg_prob_left = sam_seg_prob_left.cpu().numpy().squeeze()
                sam_seg_left = (sam_seg_prob_left > th).astype(np.uint8)
                if torch.argmax(outputs[0]["taxonomy"]) == 0:
                    sam_seg_right = np.zeros_like(sam_seg_left)
            if torch.argmax(outputs[0]["taxonomy"]) != 0:
                sam_seg_prob_right = torch.sigmoid(outputs[0]["masks_right"].squeeze(1))
                sam_seg_prob_right = sam_seg_prob_right.cpu().numpy().squeeze()
                sam_seg_right = (sam_seg_prob_right > th).astype(np.uint8)
                if torch.argmax(outputs[0]["taxonomy"]) == 1:
                    sam_seg_left = np.zeros_like(sam_seg_right)

            sam_seg = np.logical_or(sam_seg_left, sam_seg_right).astype(np.uint8)
            aff_mask = cv2.cvtColor(sam_seg, cv2.COLOR_GRAY2BGR)

            return aff_mask, sam_seg_left, sam_seg_right
    
    def test_on_benchmark(self, benchmark_folder, out_images):
        #iou = 0
        #ctr = 0
        th_list = [0.1, 0.2, 0.3, 0.5, 0.7]
        if not os.path.exists(out_images):
            os.makedirs(out_images)
        dirs = os.listdir(benchmark_folder)
        for directory in dirs:
            dir_path = os.path.join(benchmark_folder, directory)
            folders = os.listdir(dir_path)
            for folder in folders:
                img_path = os.path.join(dir_path, folder, 'inpainting.png')
                #aff_left_path = os.path.join(benchmark_folder, folder, 'aff_left.png')
                #aff_right_path = os.path.join(benchmark_folder, folder, 'aff_right.png')
                prompt_path = os.path.join(dir_path, folder, 'annotation.json')
                with open(prompt_path, 'r') as file:
                    data = json.load(file)
                    prompt = data['narration']
                img = cv2.imread(img_path)
                #aff_left = cv2.imread(aff_left_path)
                #aff_right = cv2.imread(aff_right_path)
                #aff_gt = np.logical_or(aff_left, aff_right).astype(np.uint8)
                for th in th_list:
                    _, aff_left, aff_right = self.test_on_single(img, prompt, th)
                    output_path = os.path.join(out_images+str(th), directory, folder)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    if np.sum(aff_left) != 0:
                        aff_left = aff_left * 255
                        Image.fromarray(aff_left).save(os.path.join(output_path, 'aff_left.png'))
                    if np.sum(aff_right) != 0:
                        aff_right = aff_right * 255
                        Image.fromarray(aff_right).save(os.path.join(output_path, 'aff_right.png'))
                    if np.sum(aff_left) == np.sum(aff_right) == 0:
                        print(f"No successful aff mask prediction Saving null mask at location {os.path.join(output_path)}")
                        Image.fromarray(aff_left).save(os.path.join(output_path, 'aff_left.png'))
                #Image.fromarray(aff_pred).save()
                # img_pred = self.save_result(img, aff_pred, None, os.path.join(out_images, folder, 'pred.png'))
                # img_gt = self.save_result(img, aff_gt, None, os.path.join(out_images, folder, 'gt.png'))
                # self.concatenate_images_with_caption(img_pred, img_gt, prompt, os.path.join(out_images, folder, 'concatenated.png'))

    def save_result(self, img, mask, prompt, out):
        mask = cv2.resize(mask.astype(img.dtype), (img.shape[0], img.shape[1]))
        if max(mask.flatten()) == 1:
            mask = mask * 255
        print("Image Shape: ", img.shape)
        print("Mask Shape: ", mask.shape)
        overlay = cv2.addWeighted(mask, 0.5, img[:,:,::-1], 0.5, 0)
        print("saving result")
        Image.fromarray(overlay).save(out)
        return overlay

    def concatenate_images_with_caption(self, img1, img2, caption, output_path):
        # Open the two images
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        
        # Resize images to have the same height (optional)
        height = min(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * height / img1.height), height))
        img2 = img2.resize((int(img2.width * height / img2.height), height))
        
        # Create a new image with extra space at the top for the caption
        caption_height = 40  # Height for the caption text
        new_width = img1.width + img2.width
        new_height = height + caption_height
        concatenated_img = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))
        
        # Draw the caption on the top part of the image
        draw = ImageDraw.Draw(concatenated_img)
        try:
            # Load a font
            font = ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()
        
        # Get text size using textbbox (bounding box method)
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Width of the text
        text_height = text_bbox[3] - text_bbox[1]  # Height of the text

        # Position the text in the center of the available space
        text_position = ((new_width - text_width) // 2, (caption_height - text_height) // 2)  # Centered text
        draw.text(text_position, caption, font=font, fill=(0, 0, 0))  # Black text

        # Paste the images below the caption
        concatenated_img.paste(img1, (0, caption_height))
        concatenated_img.paste(img2, (img1.width, caption_height))
        
        # Save the concatenated image as PNG
        concatenated_img.save(output_path, format='PNG')

def main(data_folder, checkpoint_file, out_images, test_single, img_path, prompt, image_name, benchmark_folder):

    # Download pretrained model
    sam_model = build_sam_vit_b("pretrained/sam_vit_b_01ec64.pth", is_sam_pretrained=True)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base", is_sam_pretrained=True)
    config = {
    "in_dim": 512,
    "out_dim": 256
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_sam = TwoHandedAfforder(sam_model, config, device)
    state_dict = torch.load(checkpoint_file, map_location='cuda:0')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    clip_sam.load_state_dict(new_state_dict)

    # Instantiate affordance detection object
    haff_model = InferenceHandler(clip_sam, processor)


    if test_single:
        if not os.path.exists(out_images):
            os.makedirs(out_images)
        out = os.path.join(out_images, image_name)
        img = cv2.imread(img_path)
        #img_tensor = torch.tensor(img).to(device)
        max_shape = max([img.shape[0], img.shape[1]])
        img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
        pred, _, _ = haff_model.test_on_single(img, prompt)
        haff_model.save_result(img, pred, prompt, out)
    elif benchmark_folder != None:
        haff_model.test_on_benchmark(benchmark_folder, out_images)
    elif not test_single and benchmark_folder == None:
        haff_model.test(data_folder, out_images)


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-folder', default=None, help='directory where the images for training are located')
    parser.add_argument('--checkpoint', default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument('--out', default=None)
    parser.add_argument('--single', default=False)
    parser.add_argument('--img', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--image-name', default=None)
    parser.add_argument('--benchmark-folder', default=None)
    args = parser.parse_args()
    vals = vars(args)
    if vals["checkpoint"] != None and vals["out"] != None and (vals["benchmark_folder"] != None or vals["data_folder"] != None or (vals["single"] != None and vals["img"] != None and vals["prompt"] != None and vals["image_name"] != None)):
        data_folder = vals["data_folder"]
        benchmark_folder = vals["benchmark_folder"]
        checkpoint_file = vals["checkpoint"]
        out_images = vals["out"]
        test_single = vals["single"]
        img_path = vals["img"]
        prompt = vals["prompt"]
        image_name = vals["image_name"]
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        main(data_folder, checkpoint_file, out_images, test_single, img_path, prompt, image_name, benchmark_folder)
    else:
        print("Wrong Arguments!")
