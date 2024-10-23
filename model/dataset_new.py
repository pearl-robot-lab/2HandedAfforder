import numpy as np
import torch

from torch.utils.data import Dataset

class AffDataset(Dataset):
  def __init__(self, img, aff_mask_left, aff_mask_right, txt_prompt, taxonomies, processor):
    self.img = img
    self.aff_mask_left = aff_mask_left
    self.aff_mask_right = aff_mask_right
    self.txt_prompt = txt_prompt
    self.taxonomies = taxonomies
    self.processor = processor

  def __len__(self):
    return self.img.size()[0]

  def __getitem__(self, idx):
    image = self.img[idx]
    ground_truth_mask_left = self.aff_mask_left[idx]
    ground_truth_mask_right = self.aff_mask_right[idx]
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=None, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask_left"] = ground_truth_mask_left
    inputs["ground_truth_mask_right"] = ground_truth_mask_right
    inputs["image"] = inputs["pixel_values"]
    inputs["txt_prompt"] = self.txt_prompt[idx]
    inputs["taxonomies_gt"] = self.taxonomies[idx] 
    return inputs