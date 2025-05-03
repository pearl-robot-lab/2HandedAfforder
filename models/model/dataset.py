import numpy as np
import torch

from torch.utils.data import Dataset

class AffDataset(Dataset):
  def __init__(self, img, aff_mask, txt_prompt, processor):
    self.img = img
    self.aff_mask = aff_mask
    self.txt_prompt = txt_prompt
    self.processor = processor

  def __len__(self):
    return self.img.size()[0]

  def __getitem__(self, idx):
    image = self.img[idx]
    ground_truth_mask = self.aff_mask[idx]
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=None, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    inputs["image"] = inputs["pixel_values"]
    inputs["txt_prompt"] = self.txt_prompt[idx]
    return inputs