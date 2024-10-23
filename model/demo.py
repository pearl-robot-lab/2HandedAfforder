import cv2
import h5py
import matplotlib.pyplot as plt
import monai
import numpy as np 
import os
import torch
import wandb

from argparse import ArgumentParser
from PIL import Image
from statistics import mean
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from segment_anything import build_sam_vit_b
from segment_anything.modeling import Sam
from transformers import SamProcessor
from typing import Optional, Tuple
from aff_model import TwoHandedAfforder
from dataset import AffDataset

class InferenceHandler():
  def __init__(self, model, processor):
    self.model = model
    self.processor = processor

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

  def is_valid(self, aff_left, aff_right):
    assert aff_left.shape == aff_right.shape
    if (not np.any(aff_left) and np.any(aff_right)) or (not np.any(aff_right) and np.any(aff_left)):
      return True, "right" if np.any(aff_right) else "left"
    else:
      print("found invalid datapoint")
      return False, ""
      
  def extract_tensor(self, aff_left, aff_right, bbox_left=None, bbox_right=None):
    assert aff_left.shape[0] == aff_right.shape[0]
    aff = []
    for i in range(aff_left.shape[0]):
      valid, side = self.is_valid(aff_left[i], aff_right[i])
      if valid:
        if side == "left":
            aff.append(cv2.cvtColor(aff_left[i], cv2.COLOR_BGR2GRAY))
        else:
            aff.append(cv2.cvtColor(aff_right[i], cv2.COLOR_BGR2GRAY))
      else:
          aff.append(cv2.cvtColor(aff_left[i], cv2.COLOR_BGR2GRAY))
    aff = torch.tensor(np.array(aff))
    return aff

  def prepare_mask(self, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 1, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

  def prepare_data(self, data_folder):
    files = os.listdir(data_folder)
    affs = []
    images = []
    text_prompts = []
    for file in files:
      with h5py.File(os.path.join(data_folder, file), "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        aff_left = np.array(f[a_group_key]["aff_left"])
        aff_right = np.array(f[a_group_key]["aff_right"])
        obj_left = np.array(f[a_group_key]["obj_mask_left"])
        obj_right = np.array(f[a_group_key]["obj_mask_right"])
        text = np.array(f[a_group_key]["narration"])
        imgs = np.array(f[a_group_key]["inpainted"])
        if imgs.size == 0:
          continue
        assert obj_left.shape == obj_right.shape
        resized_imgs = []
        imgs = imgs[:,:,:,::-1]
        for img in imgs:
            resized_imgs.append(cv2.resize(img, (aff_left.shape[1], aff_left.shape[2])))
        images.append(np.array(resized_imgs))
        aff = self.extract_tensor(aff_left, aff_right)
        affs.append(aff)
        text_prompts.extend(text.tolist())
    aff_all = torch.cat(affs)
    aff_arr = aff_all.cpu().detach().numpy()
    images_arr = np.concatenate(images)
    return affs, images_arr, text_prompts

  def test_on_single_image(self, img, prompt):
    self.model.to(self.device)
    inputs = self.processor(img, input_boxes=None, return_tensors="pt").to(self.device)
    inputs["image"] = inputs["pixel_values"].squeeze(0)
    inputs["txt_prompt"] = [prompt]
    inputs["original_size"] = (inputs["original_sizes"][0][0], inputs["original_sizes"][0][1])
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(inputs)
    sam_seg_prob = torch.sigmoid(outputs[0]["masks"].squeeze(1))
    sam_seg_prob = sam_seg_prob.cpu().numpy().squeeze()
    sam_seg = (sam_seg_prob > 0.5).astype(np.uint8)
    mask = self.prepare_mask(sam_seg)
    return mask

  def preprocess_image_for_inference(self, img, out):
    if not os.path.exists(out):
      os.makedirs(out)
    shp = img.shape
    max_shp = max(shp)
    pad_img = cv2.copyMakeBorder(img, max_shp - shp[0], 0, max_shp - shp[1], 0, cv2.BORDER_CONSTANT)
    img = cv2.resize(pad_img, (256, 256))
    img_tensor = torch.tensor(img)
    return img_tensor
  
  def save_result(self, img, mask, prompt, out):
    mask = cv2.resize(mask.astype(img.dtype), (img.shape[0], img.shape[1]))
    if max(mask.flatten()) == 1:
      mask = mask * 255
    overlay = cv2.addWeighted(mask, 0.5, img[:,:,::-1], 0.5, 0)
    print("saving result")
    Image.fromarray(overlay).save(out)

  def test_on_dataset(self, data_folder, out):
    if not os.path.exists(out):
      os.makedirs(out)
    aff_all, images_arr, text_prompts = self.prepare_data(data_folder)
    affs = torch.cat(aff_all)
    images = torch.tensor(images_arr)

    if not os.path.exists(os.path.join(out, "predictions")):
      os.makedirs(os.path.join(out, "predictions"))
    if not os.path.exists(os.path.join(out, "gt")):
      os.makedirs(os.path.join(out, "gt"))
    if not os.path.exists(os.path.join(out, "prompts")):
      os.makedirs(os.path.join(out, "prompts"))

    for i in range(images.size()[0]):
      img = images[i]
      prompt = text_prompts[i]
      mask = self.test_on_single_image(img, prompt)
      out_path_pred = os.path.join(out, "predictions", str(i) +  ".jpg")
      out_path_gt = os.path.join(out, "gt", str(i) + ".jpg")
      prompt_path = os.path.join(out, "prompts", str(i) + ".txt")
      aff_mask = cv2.cvtColor(affs[i].cpu().detach().numpy().astype(images_arr[i].dtype), cv2.COLOR_GRAY2BGR)
      self.save_result(images_arr[i], mask, prompt, out_path_pred)
      self.save_result(images_arr[i], aff_mask, prompt, out_path_gt)
      if not isinstance(prompt, str):
          prompt = prompt.decode('utf-8')
      with open(prompt_path, 'w') as file:
        file.write(prompt)

  def load_pretrained_weights(self, checkpoint_file):
    self.model = torch.jit.load(checkpoint_file)

def main(data_folder, checkpoint_file, out_images, test_single, img_path, prompt, image_name):

  # Download pretrained model
  sam_model = build_sam_vit_b("pretrained/sam_vit_b_01ec64.pth")
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  config = {
      "in_dim": 512,
      "out_dim": 256
  }
  device = "cuda" if torch.cuda.is_available() else "cpu"
  clip_sam = TwoHandedAfforder(sam_model, config, device)
  clip_sam.load_state_dict(torch.load(checkpoint_file, map_location='cuda:0'))

  # Instantiate affordance detection object
  haff_model = InferenceHandler(clip_sam, processor)

  if not test_single:
    # Prepare data data
    aff_all, images_arr, text_prompts = haff_model.prepare_data(data_folder)

    # Tensors
    affs = torch.cat(aff_all)
    images = torch.tensor(images_arr)
    haff_model.test_on_dataset(data_folder, out_images)

  else:
    if not os.path.exists(out_images):
      os.makedirs(out_images)
    out = os.path.join(out_images, image_name)
    img = cv2.imread(img_path)
    img_tensor = torch.tensor(img)
    max_shape = max([img.shape[0], img.shape[1]])
    img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
    pred = haff_model.test_on_single_image(img_tensor, prompt)
    haff_model.save_result(img, pred, prompt, out)

       
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--data-folder', default=None, help='directory where the images for training are located')
  parser.add_argument('--checkpoint', default="pretrained/sam_vit_b_01ec64.pth")
  parser.add_argument('--out', default=None)
  parser.add_argument('--test-single', default=False, type=bool)
  parser.add_argument('--img', default=None)
  parser.add_argument('--prompt', default=None)
  parser.add_argument('--image-name', default=None)
  args = parser.parse_args()
  vals = vars(args)
  if (vals["data_folder"] != None and vals["checkpoint"] != None and vals["out"] != None) or (vals["test_single"] and vals["checkpoint"] != None and vals["img"] != None and vals["prompt"] != None and vals["out"] != None and vals["image_name"] != None):
    data_folder = vals["data_folder"]
    checkpoint_file = vals["checkpoint"]
    out_images = vals["out"]
    img_path = vals["img"]
    prompt = vals["prompt"]
    test_single = vals["test_single"]
    image_name = vals["image_name"]
    main(data_folder, checkpoint_file, out_images, test_single, img_path, prompt, image_name)
  else:
    print("Wrong Arguments!")
