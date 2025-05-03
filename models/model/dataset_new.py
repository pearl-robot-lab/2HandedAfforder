import h5py
import ujson as json
import numpy as np
import os
import re
import time
import torch

from torch.utils.data import Dataset
from json_mask_handler import get_masks_and_overlay, recreate_mask_from_contours

"""
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
    #inputs["image"] = torch.permute(image, (1, 2, 0)).unsqueeze(0)
    #inputs["image"] = torch.nn.functional.interpolate(image, size=(1024, 1024), mode='bicubic', align_corners=False)
    inputs["image"] = inputs["pixel_values"]
    inputs["txt_prompt"] = self.txt_prompt[idx]
    inputs["taxonomies_gt"] = self.taxonomies[idx] 
    return inputs
"""

class AffDataset(Dataset): 
  def __init__(self, folder_path, processor):
    self.folder_path_h5 = os.path.join(folder_path, 'h5')
    self.folder_path_json = os.path.join(folder_path, 'jsons')
    self.processor = processor
    self.h5_names = os.listdir(self.folder_path_h5)
    self.size = sum(h5py.File(os.path.join(self.folder_path_h5, f), 'r')['data']['inpainted'].shape[0] for f in self.h5_names if f.endswith('.h5'))
    def extract_number(filename):
      # Use regular expression to extract the first number from the filename (before the first underscore or before the extension)
      match = re.search(r'(\d+)', filename)
      return int(match.group(1)) if match else float('inf')
    self.json_names = sorted(os.listdir(self.folder_path_json), key=extract_number)
    self.index_ranges = []
    self.mask_strings = []
  
    self.original_size = None
    self.aff_masks_left = []
    self.aff_masks_right = []
    for filename in self.json_names:
      json_path = os.path.join(self.folder_path_json, filename)
      range_part = filename.split('_')[0]
      start_idx, end_idx = map(int, range_part.split('-'))
      self.index_ranges.append([start_idx, end_idx])
      with open(json_path, 'r') as f:           
        json_data = json.load(f)
        if self.original_size is None:
          self.original_size = json_data["0"]["original_size"]
        #self.mask_strings.append(json_data)
        for key in json_data:
          #print("Key: ", key)
          entry = json_data[key]
          self.aff_masks_left.append(entry.get("aff_left", []))
          #print("Entry Aff Left: ", entry.get("aff_left", []))
          self.aff_masks_right.append(entry.get("aff_right", []))
    #print(len(self.aff_masks_left))
    assert len(self.aff_masks_left) == len(self.aff_masks_right)


  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    #print("Getting Item: ", str(idx))
    #start_time = time.time()
    #_, _, ground_truth_mask_left, ground_truth_mask_right, txt_prompt, image, taxonomy = self.extract_index_from_h5(self.folder_path, idx)
    """
    json_index = -1
    target_start = -1
    for i, (start, end) in enumerate(self.index_ranges):
      if start <= idx <= end:
        target_start = start
        json_index = i
        break
    if json_index == -1 or target_start == -1:
      print("Failed to retrieve the correct json file")
      return
    """
    #mask_data = self.mask_strings[json_index].get(str(idx - target_start))
    mask_data = {}
    mask_data['aff_left'] = self.aff_masks_left[idx]
    mask_data['aff_right'] = self.aff_masks_right[idx]
    """
    if mask_data is None:
        raise IndexError(f"Invalid index {local_index} for masks in {json_data['original_size']}.")
    """
    # Recreate masks as numpy arrays from contours for all 4 masks
    # original_size = mask_data['original_size']
    masks = {}

    for mask_key in ['aff_left', 'aff_right']:
        contours = mask_data[mask_key]
        #print("Contours: ", contours)
        masks[mask_key] = recreate_mask_from_contours(contours, shape=self.original_size)
        #print("Masks: ", masks[mask_key])


    txt_prompt, image, taxonomy = self.extract_index_from_h5(self.folder_path_h5, self.h5_names, idx)
    #masks, _  = get_masks_and_overlay(idx, self.folder_path_json, self.folder_path_h5)
    #image = self.img[idx]
    #ground_truth_mask_left = self.aff_mask_left[idx]
    #ground_truth_mask_right = self.aff_mask_right[idx]
    ground_truth_mask_left = masks['aff_left'] * 255
    ground_truth_mask_right = masks['aff_right'] * 255
    #print("Mask not empty: ", str(ground_truth_mask_left[np.where(ground_truth_mask_left != 0)].shape[0] != 0))
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=None, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask_left"] = ground_truth_mask_left
    inputs["ground_truth_mask_right"] = ground_truth_mask_right
    #inputs["image"] = torch.permute(image, (1, 2, 0)).unsqueeze(0)
    #inputs["image"] = torch.nn.functional.interpolate(image, size=(1024, 1024), mode='bicubic', align_corners=False)
    inputs["image"] = inputs["pixel_values"]
    inputs["txt_prompt"] = txt_prompt
    inputs["taxonomies_gt"] = taxonomy
    #end_time = time.time()

    #print("Get Item Time: ", end_time - start_time)
    return inputs

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
          #data_mask_left = h5_file['data']['obj_mask_left']
          #data_mask_right= h5_file['data']['obj_mask_right']
          #data_aff_left = h5_file['data']['aff_left']
          #data_aff_right = h5_file['data']['aff_right']
          data_narration = h5_file['data']['narration']
          data_inpainted = h5_file['data']['inpainted']
          data_taxonomy = h5_file['data']['taxonomy']
          adjusted_index = index - start_idx  # Calculate position within the file
          #obj_mask_left = data_mask_left[adjusted_index]  # Extract the specific entry
          #obj_mask_right = data_mask_right[adjusted_index]  # Extract the specific entry
          #aff_left = data_aff_left[adjusted_index]  # Extract the specific entry
          #aff_right = data_aff_right[adjusted_index]  # Extract the specific entry
          narration = data_narration[adjusted_index]  # Extract the specific entry
          inpainted = data_inpainted[adjusted_index]  # Extract the specific entry
          taxonomy = data_taxonomy[adjusted_index]  # Extract the specific entry

          
      #return obj_mask_left, obj_mask_right, aff_left, aff_right, narration, inpainted, taxonomy
      return narration, inpainted, taxonomy

  # Example usage
  # folder_path = "/path/to/your/h5_files"
  # index = 12
  # data = extract_index_from_h5(folder_path, index)
  # print(data)
