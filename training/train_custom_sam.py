import cv2
import h5py
import matplotlib.pyplot as plt
import monai
import numpy as np 
import os
import torch

from argparse import ArgumentParser
from PIL import Image
# from segment_anything import SamPredictor, sam_model_registry
from statistics import mean
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from transformers import SamModel, SamProcessor


class SAMDataset(Dataset):
  def __init__(self, img, aff_mask, bbox, processor):
    self.img = img
    self.aff_mask = aff_mask
    self.bbox = bbox
    self.processor = processor

  def __len__(self):
    return self.img.size()[0]

  def __getitem__(self, idx):
    #item = self.dataset[idx]
    #image = item["image"]
    #ground_truth_mask = np.array(item["label"])
    image = self.img[idx]
    ground_truth_mask = self.aff_mask[idx]

    # get bounding box prompt
    #prompt = get_bounding_box(ground_truth_mask)
    bbox = self.bbox[idx]
    if bbox.shape == (2, 2):
      prompt = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
    else:
      prompt = bbox.tolist()
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def is_valid(aff_left, aff_right):
    assert aff_left.shape == aff_right.shape
    if (not np.any(aff_left) and np.any(aff_right)) or (not np.any(aff_right) and np.any(aff_left)):
      return True, "right" if np.any(aff_right) else "left"
    else:
      print("found invalid datapoint")
      return False, ""
      
def extract_tensor(aff_left, aff_right, bbox_left=None, bbox_right=None):
    assert aff_left.shape[0] == aff_right.shape[0]
    aff = []
    for i in range(aff_left.shape[0]):
      valid, side = is_valid(aff_left[i], aff_right[i])
      if valid:
        if side == "left":
            aff.append(cv2.cvtColor(aff_left[i], cv2.COLOR_BGR2GRAY))
        else:
            aff.append(cv2.cvtColor(aff_right[i], cv2.COLOR_BGR2GRAY))
      else:
          continue
    aff = torch.tensor(np.array(aff))
    return aff

def visualize_datapoint(img_arr, bbox_arr, aff_arr):
  y_min = int(bbox_arr[0][0])
  x_min = int(bbox_arr[0][1])
  y_max = int(bbox_arr[1][0])
  x_max = int(bbox_arr[1][1])
  if len(aff_arr.shape) == 2:
    aff_arr = cv2.cvtColor(aff_arr, cv2.COLOR_GRAY2BGR)
  overlay = cv2.addWeighted(aff_arr, 1, img_arr, 1, 0)
  cv2.imshow("", cv2.rectangle(overlay, (x_min, y_min),(x_max, y_max), color=(255, 0, 0), thickness=2))
  cv2.waitKey(0)

def show_mask(mask, random_color=False):
  if random_color:
      color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
  else:
      color = np.array([1, 1, 1])
  print("mask shape: ", mask.shape)
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  print("mask image shape: ", mask_image.shape)
  return mask_image

def prepare_data(data_folder, bbox_eps, visualization_active):
  files = os.listdir(data_folder)
  affs = []
  bboxes = []
  images = []
  if not os.path.exists(model_folder):
    os.makedirs(model_folder)
  for file in files:
    with h5py.File(os.path.join(data_folder, file), "r") as f:
      print("Keys: %s" % f.keys())
      a_group_key = list(f.keys())[0]
      print(type(f[a_group_key]))
      print(type(a_group_key))
      data = list(f[a_group_key])
      print(data)
      aff_left = np.array(f[a_group_key]["aff_left"])
      aff_right = np.array(f[a_group_key]["aff_right"])
      obj_left = np.array(f[a_group_key]["obj_mask_left"])
      obj_right = np.array(f[a_group_key]["obj_mask_right"])
      imgs = np.array(f[a_group_key]["inpainted"])
      if imgs.size == 0:
        continue
      assert obj_left.shape == obj_right.shape
      for i in range(obj_left.shape[0]):
        if np.any(obj_left[i]):
          x_indices, y_indices, _ = np.where(obj_left[i] > 0)
          x_min, x_max = np.min(x_indices), np.max(x_indices)
          y_min, y_max = np.min(y_indices), np.max(y_indices)
          bboxes.append(np.array([[x_min - bbox_eps, y_min - bbox_eps], [x_max + bbox_eps, y_max + bbox_eps]]))
        elif np.any(obj_right[i]):
          x_indices, y_indices, _ = np.where(obj_right[i] > 0)
          x_min, x_max = np.min(x_indices), np.max(x_indices)
          y_min, y_max = np.min(y_indices), np.max(y_indices)
          bboxes.append(np.array([[x_min - bbox_eps, y_min - bbox_eps], [x_max + bbox_eps, y_max + bbox_eps]]))
      resized_imgs = []
      imgs = imgs[:,:,:,::-1]
      for img in imgs:
          resized_imgs.append(cv2.resize(img, (aff_left.shape[1], aff_left.shape[2])))
      images.append(np.array(resized_imgs))
      aff = extract_tensor(aff_left, aff_right)
      affs.append(aff)
  aff_all = torch.cat(affs)
  aff_arr = aff_all.cpu().detach().numpy()
  bbox_arr = np.array(bboxes)
  images_arr = np.concatenate(images)
  if visualization_active:
    for i in range(aff_arr.shape[0]):
      visualize_datapoint(images_arr[i], bbox_arr[i], aff_arr[i])
  return affs, bbox_arr, images_arr

def train(batchsize, model_folder, images, affs, bboxes, num_epochs):
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  dset = SAMDataset(img=images, aff_mask=affs, bbox=bboxes, processor=processor)
  validation_split = .2
  shuffle_dataset = True
  random_seed = 42

  dataset_size = len(dset)
  indices =list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  
  train_loader = DataLoader(dset, batch_size=batchsize, sampler=train_sampler)
  validation_loader = DataLoader(dset, batch_size=batchsize, sampler=valid_sampler)
  model = SamModel.from_pretrained("facebook/sam-vit-base")
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)
  
  optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
  seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

  # num_epochs = 100
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  model.train()
  for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_loader):
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      ground_truth_masks = ground_truth_masks/255
      predicted_masks = nn.functional.interpolate(predicted_masks, 
                                                  size=(ground_truth_masks.size()[1], ground_truth_masks.size()[2]),
                                                  mode='bilinear',
                                                  align_corners=False)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(loss.item())
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
  torch.save(model.state_dict(), os.path.join(model_folder, "bowl_model.pth"))
  return model, processor

def validate(img, prompt, processor, device, model):
  inputs = processor(img, input_boxes=[[prompt]], return_tensors="pt").to(device)
  model.eval()
  with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)
  medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
  medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
  medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
  # Image.fromarray(images_arr[i][:,:,::-1]).save(os.path.join(out_images, str(i), "image.jpg"))
  mask = show_mask(medsam_seg)
  return mask

def main(data_folder, data_folder_validation, bbox_eps, visualization_active, batchsize, model_folder, out_images, num_epochs, single_image):

  aff_all, bbox_arr, images_arr = prepare_data(data_folder, bbox_eps, visualization_active)
  # Tensors
  affs = torch.cat(aff_all)
  bboxes = torch.tensor(bbox_arr)
  images = torch.tensor(images_arr)
  model, processor = train(batchsize, model_folder, images, affs, bboxes, num_epochs)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if not os.path.exists(out_images):
    os.makedirs(out_images)
  if single_image == "False":
    aff_all, bbox_arr, images_arr = prepare_data(data_folder_validation, bbox_eps, visualization_active)
    affs = torch.cat(aff_all)
    bboxes = torch.tensor(bbox_arr)
    images = torch.tensor(images_arr)
    if not os.path.exists(os.path.join(out_images, "predictions")):
      os.makedirs(os.path.join(out_images, "predictions"))
    if not os.path.exists(os.path.join(out_images, "gt")):
      os.makedirs(os.path.join(out_images, "gt"))

    for i in range(images.size()[0]):
      img = images[i]
      bbox = bboxes[i]
      x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
      prompt = [x_min, y_min, x_max, y_max]
      mask = validate(img, prompt, processor, device, model)
      mask = cv2.resize(mask.astype(images_arr[i].dtype), (images_arr[i].shape[0], images_arr[i].shape[1]))
      mask = mask * 255
      Image.fromarray(cv2.addWeighted(mask, 0.5, images_arr[i][:,:,::-1], 0.5, 0)).save(os.path.join(out_images, "predictions", str(i) +  ".jpg"))
      aff_mask = affs[i].cpu().detach().numpy().astype(images_arr[i].dtype)
      print("Aff mask shape: ", aff_mask.shape)
      print("Aff mask dtype: ", aff_mask.dtype)
      Image.fromarray(cv2.addWeighted(cv2.cvtColor(aff_mask, cv2.COLOR_GRAY2BGR), 0.5, images_arr[i][:,:,::-1], 0.5, 0)).save(os.path.join(out_images, "gt", str(i) + ".jpg"))
  else:
    img = cv2.imread(data_folder_validation)
    shp = img.shape
    max_shp = max(shp)
    pad_img = cv2.copyMakeBorder(img, max_shp - shp[0], 0, max_shp - shp[1], 0, cv2.BORDER_CONSTANT)
    img = cv2.resize(pad_img, (256, 256))
    img_tensor = torch.tensor(img)
    prompt = [50, 90, 210, 250]
    print("Image Data:")
    print(img.shape)
    print(img.dtype)
    mask = validate(img_tensor, prompt, processor, device, model)
    mask = cv2.resize(mask.astype(img.dtype), (img.shape[0], img.shape[1]))
    mask = mask * 255
    Image.fromarray(cv2.addWeighted(mask, 0.5, img[:,:,::-1], 0.5, 0)).save(os.path.join(out_images, single_image))
      
       
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--bbox-eps', default=20, type=int, help='padding added on top of the bounding box')
  parser.add_argument('--data-folder', default=None, help='directory where the images for training are located')
  parser.add_argument('--visualization-active', default=False, type=bool, help='parameter to indicate whether the data points should be visualized or not')
  parser.add_argument('--model-folder', default=None, help='path to the model folder')
  parser.add_argument('--batchsize', default=1, type=int)
  parser.add_argument('--data-folder-validation', default=None, help='directory where the images are located for validation/ image file if single-image != False')
  parser.add_argument('--out-images', default=None, help='ouput directory')
  parser.add_argument('--num-epochs', default=100, type=int)
  parser.add_argument('--single-image', default="", help='False if multiple images are being infered, else the image name for the output should be provided')
  args = parser.parse_args()
  vals = vars(args)
  if vals["data_folder"] != None and vals["model_folder"] != None and vals["data_folder_validation"] != None and vals["out_images"] != None:
    data_folder = vals["data_folder"]
    bbox_eps = vals["bbox_eps"]
    visualization_active = vals["visualization_active"]
    model_folder = vals["model_folder"]
    batchsize = vals["batchsize"]
    data_folder_validation = vals["data_folder_validation"]
    out_images = vals["out_images"]
    num_epochs = vals["num_epochs"]
    single_image = vals["single_image"]
    main(data_folder, data_folder_validation, bbox_eps, visualization_active, batchsize, model_folder, out_images, num_epochs, single_image)
  else:
    print("Wrong Arguments!")
