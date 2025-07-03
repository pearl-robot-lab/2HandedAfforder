import glob
import h5py
import json
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        explanatory=-1,
        split_p=0.9,
        index_list=None
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.size = sum(h5py.File(os.path.join(self.base_image_dir, f), 'r')['data']['obj_mask_left'].shape[0] for f in os.listdir(self.base_image_dir) if f.endswith('.h5'))
        if index_list:
            self.index_list = index_list
        else:
            self.index_list = random.sample(range(self.size), int(self.size * split_p))
        """
        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)
        """

        #print("number of reason_seg samples: ", len(images))

    def __len__(self):
        return self.samples_per_epoch
    
    def get_missing_indices(self):
        return list(set(range(self.size)) - set(self.index_list))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        """
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        """
        return x

    def __getitem__(self, idx):
        idx = random.choice(self.index_list)
        _, _, ground_truth_mask_left, ground_truth_mask_right, sents, image, taxonomy = self.extract_index_from_h5(self.base_image_dir, idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        is_sentence = True
        sents = [sents]
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks_left = [
            ground_truth_mask_left
        ]
        sampled_masks_right = [
            ground_truth_mask_right
        ]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        #image_name = image_path.split("/")[-1]

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            #img_name = image_path.split("/")[-1]

            answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        #image_name = image_path.split("/")[-1]
        """
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
        """
        masks_left = np.stack(sampled_masks_left, axis=0)
        masks_right = np.stack(sampled_masks_right, axis=0)
        masks_left = torch.from_numpy(masks_left)
        masks_right = torch.from_numpy(masks_right)
        label = torch.ones(masks_left.shape[1], masks_left.shape[2]) #* self.ignore_label
        image_path = ""
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks_left,
            masks_right,
            label,
            resize,
            questions,
            sampled_sents,
            taxonomy,
        )
    def get_file_for_index(self, folder_path, index):
        # List all h5 files and find which file contains the target index based on the filename
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        
        # Regex to match the index range in the filename (e.g., "0-9_filename.h5")
        pattern = r"(\d+)-(\d+)_"
        
        for filename in h5_files:
            match = re.match(pattern, filename)
            if match:
                start_idx, end_idx = map(int, match.groups())
                if start_idx <= index <= end_idx:
                    return os.path.join(folder_path, filename), start_idx
        
        raise ValueError(f"Index {index} not found in any file in {folder_path}.")

    def extract_index_from_h5(self, folder_path, index):
        # Get the file and the start index of its range
        file_path, start_idx = self.get_file_for_index(folder_path, index)
        
        # Open the file and extract the specific entry at the adjusted index
        with h5py.File(file_path, 'r') as h5_file:
            data_mask_left = h5_file['data']['obj_mask_left']
            data_mask_right= h5_file['data']['obj_mask_right']
            data_aff_left = h5_file['data']['aff_left']
            data_aff_right = h5_file['data']['aff_right']
            data_narration = h5_file['data']['narration']
            data_inpainted = h5_file['data']['inpainted']
            data_taxonomy = h5_file['data']['taxonomy']
            adjusted_index = index - start_idx  # Calculate position within the file
            obj_mask_left = data_mask_left[adjusted_index]  # Extract the specific entry
            obj_mask_right = data_mask_right[adjusted_index]  # Extract the specific entry
            aff_left = data_aff_left[adjusted_index]  # Extract the specific entry
            aff_right = data_aff_right[adjusted_index]  # Extract the specific entry
            narration = data_narration[adjusted_index]  # Extract the specific entry
            inpainted = data_inpainted[adjusted_index]  # Extract the specific entry
            taxonomy = data_taxonomy[adjusted_index]  # Extract the specific entry

            
        return obj_mask_left, obj_mask_right, aff_left, aff_right, narration, inpainted, taxonomy
