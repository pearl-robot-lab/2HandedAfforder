import glob
import json
import os
import random
import re

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

#from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

DEFAULT_IMAGE_TOKEN = "<image>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you show me where I have to interact with the objects to perform the following task: {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the region to perform the action '{class_name}' in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "How can I perform the action '{class_name}' in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "How can I perform the action '{class_name}' in this image? Please output segmentation mask.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class AffDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        inference,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        #num_classes_per_sample: int = 3,
        #exclude_val=False,
        #sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
    ):
        #self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        #self.num_classes_per_sample = num_classes_per_sample
        self.inference = inference
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        # Check if base_image_dir is a HuggingFace dataset identifier
        self.use_hf_dataset = self._is_hf_dataset_identifier(base_image_dir)
        
        if self.use_hf_dataset:
            print(f"Loading dataset from HuggingFace: {base_image_dir}")
            self._load_from_huggingface(base_image_dir)
        else:
            print(f"Loading dataset from local directory: {base_image_dir}")
            self._load_from_local(base_image_dir)

    def _is_hf_dataset_identifier(self, path):
        """Check if the path is a HuggingFace dataset identifier (e.g., 'user/dataset')."""
        # HuggingFace identifiers typically have format "user/dataset" and don't exist as local paths
        has_slash = '/' in path
        path_exists = os.path.exists(path)
        
        print(f"DEBUG: Checking if '{path}' is HF dataset:")
        print(f"  - Has '/': {has_slash}")
        print(f"  - Path exists: {path_exists}")
        print(f"  - HF_DATASETS_AVAILABLE: {HF_DATASETS_AVAILABLE}")
        
        if has_slash and not path_exists and HF_DATASETS_AVAILABLE:
            return True
        
        if has_slash and not path_exists and not HF_DATASETS_AVAILABLE:
            raise ImportError(
                f"The path '{path}' appears to be a HuggingFace dataset identifier, "
                f"but the 'datasets' library is not available. "
                f"Please install it with: pip install datasets"
            )
        
        return False

    def _load_from_huggingface(self, dataset_name):
        """Load dataset from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library not available. Install with: pip install datasets")
        
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, split="train")
        
        # Extract data from the HuggingFace dataset
        self.hf_data = []
        self.aff_masks_left = []
        self.aff_masks_right = []
        self.original_size = None
        
        for idx, item in enumerate(dataset):
            # Extract the data based on the dataset structure
            # Assuming structure similar to the web search results
            if self.original_size is None and 'masks' in item and 'original_size' in item['masks']:
                self.original_size = item['masks']['original_size']
            
            # Store the item for later access
            self.hf_data.append(item)
            
            # Extract masks
            if 'masks' in item:
                masks = item['masks']
                self.aff_masks_left.append(masks.get('aff_left', []))
                self.aff_masks_right.append(masks.get('aff_right', []))
            else:
                self.aff_masks_left.append([])
                self.aff_masks_right.append([])
        
        self.size = len(self.hf_data)
        print(f"Loaded {self.size} samples from HuggingFace dataset")

    def _load_from_local(self, base_image_dir):
        """Load dataset from local directory."""
        #self.sem_seg_datas = sem_seg_data.split("||")
        self.image_dir = os.path.join(base_image_dir, "h5")
        self.json_dir = os.path.join(base_image_dir, "jsons")
        def extract_number(filename):
            # Use regular expression to extract the first number from the filename (before the first underscore or before the extension)
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else float('inf')
        self.h5_names = os.listdir(self.image_dir)
        self.json_names = sorted(os.listdir(self.json_dir), key=extract_number)
        self.size = sum(h5py.File(os.path.join(self.image_dir, f), 'r')['data']['inpainted'].shape[0] for f in self.h5_names if f.endswith('.h5'))

        self.original_size = None
        #self.index_ranges = []
        self.aff_masks_left = []
        self.aff_masks_right = []
        for filename in self.json_names:
            json_path = os.path.join(self.json_dir, filename)
            range_part = filename.split('_')[0]
            start_idx, end_idx = map(int, range_part.split('-'))
            #self.index_ranges.append([start_idx, end_idx])
            with open(json_path, 'r') as f:           
                json_data = json.load(f)
                if self.original_size is None:
                    self.original_size = json_data["0"]["original_size"]
                for key in json_data:
                    entry = json_data[key]
                    self.aff_masks_left.append(entry.get("aff_left", []))
                    self.aff_masks_right.append(entry.get("aff_right", []))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        idx = random.randint(0, self.size - 1)
        
        if self.use_hf_dataset:
            text_prompt, image, taxonomy = self._extract_from_hf_dataset(idx)
        else:
            text_prompt, image, taxonomy = self.extract_index_from_h5(self.image_dir, self.h5_names, idx)
        sampled_classes = [text_prompt]
        mask_data = {}
        mask_data['aff_left'] = self.aff_masks_left[idx]
        mask_data['aff_right'] = self.aff_masks_right[idx]
        masks = {}
        masks['aff_left'] = self.recreate_mask_from_contours(mask_data['aff_left'], shape=self.original_size)
        masks['aff_right'] = self.recreate_mask_from_contours(mask_data['aff_right'], shape=self.original_size)
        label_left = (masks['aff_left'] == 0).astype(int) * 255
        label_right = (masks['aff_right'] == 0).astype(int) * 255
        #label_left[label_left == 0] = 255
        #label_right[label_right == 0] = 255
        #label_left -= 1
        #label_right -= 1y
        #label_left[label_left == 254] = 255
        #label_right[label_right == 254] = 255
        label = {
            'left': torch.from_numpy(label_left).long(),
            'right': torch.from_numpy(label_right).long()
        }
        masks['aff_left'] = torch.from_numpy(masks['aff_left']).unsqueeze(0)
        masks['aff_right'] = torch.from_numpy(masks['aff_right']).unsqueeze(0)

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            text = sampled_cls
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        if self.inference:
            return (
                None,
                image,
                image_clip,
                conversations,
                masks['aff_left'],
                masks['aff_right'],
                taxonomy,
                label,
                resize,
                questions,
                sampled_classes,
                self.inference
            )
        else:
            return (
                None,
                image,
                image_clip,
                conversations,
                masks['aff_left'],
                masks['aff_right'],
                taxonomy,
                label,
                resize,
                questions,
                sampled_classes
            )
    
    def _extract_from_hf_dataset(self, idx):
        """Extract data from HuggingFace dataset at given index."""
        item = self.hf_data[idx]
        
        # Extract narration/text prompt
        text_prompt = item.get('narration', item.get('text', ''))
        if isinstance(text_prompt, bytes):
            text_prompt = text_prompt.decode('utf-8')
        
        # Extract image
        if 'image' in item:
            image = np.array(item['image'])
        elif 'inpainted' in item:
            image = np.array(item['inpainted'])
        else:
            # Fallback: create a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Extract taxonomy
        taxonomy = item.get('taxonomy', 2)  # Default to 2 (both hands)
        if isinstance(taxonomy, bytes):
            taxonomy = int(taxonomy.decode('utf-8'))
        
        return text_prompt, image, taxonomy
    
    def get_file_for_index(self, folder_path, folder_list, index):
        # List all h5 files and find which file contains the target index based on the filename
        h5_files = [f for f in folder_list if f.endswith('.h5')]
        
        # Regex to match the index range in the filename (e.g., "0-9_filename.h5")
        pattern = r"(\d+)-(\d+)_"
        
        for filename in h5_files:
            match = re.match(pattern, filename)
            if match:
                start_idx, end_idx = map(int, match.groups())
                if start_idx <= index <= end_idx:
                    return os.path.join(folder_path, filename), start_idx
        
        raise ValueError(f"Index {index} not found in any file in {folder_path}.")

    def extract_index_from_h5(self, folder_path, folder_list, index):
        # Get the file and the start index of its range
        file_path, start_idx = self.get_file_for_index(folder_path, folder_list, index)
        
        # Open the file and extract the specific entry at the adjusted index
        with h5py.File(file_path, 'r') as h5_file:
            data_narration = h5_file['data']['narration']
            data_inpainted = h5_file['data']['inpainted']
            data_taxonomy = h5_file['data']['taxonomy']
            adjusted_index = index - start_idx  # Calculate position within the file
            narration = data_narration[adjusted_index]  # Extract the specific entry
            inpainted = data_inpainted[adjusted_index]  # Extract the specific entry
            taxonomy = data_taxonomy[adjusted_index]  # Extract the specific entry
            
        #return obj_mask_left, obj_mask_right, aff_left, aff_right, narration, inpainted, taxonomy
        return narration, inpainted, taxonomy
    
    def recreate_mask_from_contours(self, contours, shape):
        """Reconstruct a binary mask from OpenCV contours."""
        mask = np.zeros(shape, dtype=np.uint8)  # Create a blank mask with the same size as original
        for contour in contours:
            # Draw the contour on the mask
            cv2.drawContours(mask, [np.array(contour, dtype=np.int32)], -1, (1), thickness=cv2.FILLED)
        return mask
    


class AffDatasetVal(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        precision: str = "fp32",
        image_size: int = 224,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.data2list = {}
        self.data2classes = {}

        self.images, self.affs_left, self.affs_right, self.narrations, self.taxonomies = self.load_data_from_nested_folders(self.base_image_dir)
        print(len(self.images))
        print(len(self.affs_left))
        print(len(self.affs_right))
        print(len(self.narrations))
        print(len(self.taxonomies))
        self.size = len(self.images)


    def __len__(self):
        return self.size

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        idx = random.randint(0, self.size - 1)
        text_prompt = self.narrations[idx]
        image = self.images[idx]
        taxonomy = self.taxonomies[idx]
        masks = {}
        masks['aff_left'] = self.affs_left[idx]
        masks['aff_right'] = self.affs_right[idx]
        sampled_classes = [text_prompt]
        label_left = (masks['aff_left'] == 0).astype(int) * 255
        label_right = (masks['aff_right'] == 0).astype(int) * 255
        # label_left[label_left == 0] = 255
        # label_right[label_right == 0] = 255
        # label_left -= 1
        # label_right -= 1
        # label_left[label_left == 254] = 255
        # label_right[label_right == 254] = 255
        label = {
            'left': torch.from_numpy(label_left).long(),
            'right': torch.from_numpy(label_right).long()
        }
        masks['aff_left'] = torch.from_numpy(masks['aff_left']).unsqueeze(0)
        masks['aff_right'] = torch.from_numpy(masks['aff_right']).unsqueeze(0)
        
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        
        if masks['aff_left'].shape != image.shape:
            print(f"Image Shape: {image.shape}, Mask Shape: {masks['aff_left'].shape}")
            cv2.resize(image, masks['aff_left'].shape[1:])
        resize = image.shape[:2]

        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            text = sampled_cls
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        return (
            None,
            image,
            image_clip,
            conversations,
            masks['aff_left'],
            masks['aff_right'],
            taxonomy,
            label,
            resize,
            questions,
            sampled_classes,
            True
        )
    
    def load_data_from_nested_folders(self, root_folder):
        # Initialize lists to store the content
        inpaintings = []
        affs_left = []
        affs_right = []
        narrations = []
        taxonomies = []

        # Walk through the first level of directories (level_1_folder)
        for level_1_folder in os.listdir(root_folder):
            level_1_folder_path = os.path.join(root_folder, level_1_folder)

            if os.path.isdir(level_1_folder_path):
                # Walk through the second level of directories (level_2_folder)
                for level_2_folder in os.listdir(level_1_folder_path):
                    level_2_folder_path = os.path.join(level_1_folder_path, level_2_folder)

                    if os.path.isdir(level_2_folder_path):
                        # Initialize variables for the current level_2 folder
                        inpainting_path = None
                        aff_left_path = None
                        aff_right_path = None
                        annotation_path = None

                        # Find the files in the level_2 folder
                        for file in os.listdir(level_2_folder_path):
                            file_path = os.path.join(level_2_folder_path, file)
                            
                            if file == "inpainting.png":
                                inpainting_path = file_path
                            elif file == "aff_left.png":
                                aff_left_path = file_path
                            elif file == "aff_right.png":
                                aff_right_path = file_path
                            elif file == "annotation.json":
                                annotation_path = file_path

                        # Load inpainting image (make sure it exists)
                        if not inpainting_path or not annotation_path or (not aff_left_path and not aff_right_path):
                            continue
                        inpainting = np.array(Image.open(inpainting_path))
                        inpaintings.append(inpainting)

                        # Load aff_left and aff_right, handling missing files
                        aff_left = None
                        aff_right = None

                        # If both are missing, handle that case
                        if aff_left_path and aff_right_path:
                            aff_left = cv2.imread(aff_left_path, cv2.IMREAD_GRAYSCALE)
                            aff_right = cv2.imread(aff_right_path, cv2.IMREAD_GRAYSCALE)
                        elif aff_left_path:
                            aff_left = cv2.imread(aff_left_path, cv2.IMREAD_GRAYSCALE)
                            # If aff_left exists but aff_right is missing, create a zero-like array with the same shape as aff_left
                            aff_right = np.zeros_like(aff_left)
                        elif aff_right_path:
                            aff_right = cv2.imread(aff_right_path, cv2.IMREAD_GRAYSCALE)
                            # If aff_right exists but aff_left is missing, create a zero-like array with the same shape as aff_right
                            aff_left = np.zeros_like(aff_right)
                        else:
                            continue
                        # Add the images to the lists
                        affs_left.append(aff_left)
                        affs_right.append(aff_right)

                        # Load annotation
                        if annotation_path:
                            with open(annotation_path, 'r') as json_file:
                                annotation_data = json.load(json_file)
                                narration = annotation_data.get('narration', "")
                                narrations.append(narration)
                                taxonomy = annotation_data.get('taxonomy', "")
                                taxonomies.append(taxonomy)
                        else:
                            print(f"Warning: annotation.json missing in {level_2_folder_path}")

        return inpaintings, affs_left, affs_right, narrations, taxonomies

