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
    image = self.img[idx]
    ground_truth_mask = self.aff_mask[idx]

    bbox = self.bbox[idx]
    if bbox.shape == (2, 2):
      prompt = [bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]]
    else:
      prompt = bbox.tolist()

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


class AffordanceDetection():
  def __init__(self, model, processor, bbox_eps, batchsize, model_folder, model_name, num_epochs, visualization_active=False):
    self.model = model
    self.processor = processor
    self.bbox_eps = bbox_eps
    self.batchsize = batchsize
    self.model_folder = model_folder
    self.model_name = model_name
    self.num_epochs = num_epochs
    self.visualization_active = visualization_active

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    self.optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    self.train_loader = None
    self.validation_loader = None

    

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
          continue
    aff = torch.tensor(np.array(aff))
    return aff

  def visualize_datapoint(self, img_arr, bbox_arr, aff_arr):
    y_min = int(bbox_arr[0][0])
    x_min = int(bbox_arr[0][1])
    y_max = int(bbox_arr[1][0])
    x_max = int(bbox_arr[1][1])
    if len(aff_arr.shape) == 2:
      aff_arr = cv2.cvtColor(aff_arr, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(aff_arr, 1, img_arr, 1, 0)
    cv2.imshow("", cv2.rectangle(overlay, (x_min, y_min),(x_max, y_max), color=(255, 0, 0), thickness=2))
    cv2.waitKey(0)

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
    bboxes = []
    images = []
    if not os.path.exists(self.model_folder):
      os.makedirs(self.model_folder)
    for file in files:
      with h5py.File(os.path.join(data_folder, file), "r") as f:
        #print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        #print(type(f[a_group_key]))
        #print(type(a_group_key))
        data = list(f[a_group_key])
        #print(data)
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
            bboxes.append(np.array([[x_min - self.bbox_eps, y_min - self.bbox_eps], [x_max + self.bbox_eps, y_max + self.bbox_eps]]))
          elif np.any(obj_right[i]):
            x_indices, y_indices, _ = np.where(obj_right[i] > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bboxes.append(np.array([[x_min - self.bbox_eps, y_min - self.bbox_eps], [x_max + self.bbox_eps, y_max + self.bbox_eps]]))
        resized_imgs = []
        imgs = imgs[:,:,:,::-1]
        for img in imgs:
            resized_imgs.append(cv2.resize(img, (aff_left.shape[1], aff_left.shape[2])))
        images.append(np.array(resized_imgs))
        aff = self.extract_tensor(aff_left, aff_right)
        affs.append(aff)
    aff_all = torch.cat(affs)
    aff_arr = aff_all.cpu().detach().numpy()
    bbox_arr = np.array(bboxes)
    images_arr = np.concatenate(images)
    if self.visualization_active:
      for i in range(aff_arr.shape[0]):
        self.visualize_datapoint(images_arr[i], bbox_arr[i], aff_arr[i])
    return affs, bbox_arr, images_arr

  def train_one_epoch(self, epoch_index):
    epoch_losses = []
    for batch in tqdm(self.train_loader):
      outputs = self.model(pixel_values=batch["pixel_values"].to(self.device),
                      input_boxes=batch["input_boxes"].to(self.device),
                      multimask_output=False)
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
      ground_truth_masks = ground_truth_masks/255
      predicted_masks = nn.functional.interpolate(predicted_masks, 
                                                  size=(ground_truth_masks.size()[1], ground_truth_masks.size()[2]),
                                                  mode='bilinear',
                                                  align_corners=False)
      loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      epoch_losses.append(loss.item())
    return epoch_losses

  def validate_one_epoch(self, epoch_index):
    epoch_vlosses = []
    accuracies = []
    with torch.no_grad():
      for batch in tqdm(self.validation_loader):
        outputs = self.model(pixel_values=batch["pixel_values"].to(self.device),
                      input_boxes=batch["input_boxes"].to(self.device),
                      multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        ground_truth_masks = ground_truth_masks/255
        predicted_masks = nn.functional.interpolate(predicted_masks, 
                                                    size=(ground_truth_masks.size()[1], ground_truth_masks.size()[2]),
                                                    mode='bilinear',
                                                    align_corners=False)
        vloss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        epoch_vlosses.append(vloss.item())
        acc = monai.metrics.compute_iou(predicted_masks, ground_truth_masks.unsqueeze(1))
        accuracies.append(acc.item())
    return epoch_vlosses, accuracies

  def train(self, images, affs, bboxes):

    
    dset = SAMDataset(img=images, aff_mask=affs, bbox=bboxes, processor=self.processor)
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    wandb.init(
      project="test",
      config={
        "dataset": "Bowl_Dataset (P02_01, P04_02, P14_05, P20_03)",
        "epochs": self.num_epochs,
        "batchsize": self.batchsize
      }
    )
    dataset_size = len(dset)
    indices =list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
      np.random.seed(random_seed)
      np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    self.train_loader = DataLoader(dset, batch_size=self.batchsize, sampler=train_sampler)
    self.validation_loader = DataLoader(dset, batch_size=self.batchsize, sampler=valid_sampler)
    for name, param in self.model.named_parameters():
      if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

    self.model.to(self.device)
    for epoch in range(self.num_epochs):
      self.model.train()
      epoch_losses = self.train_one_epoch(epoch)
      self.model.eval()
      epoch_vlosses, accuracies = self.validate_one_epoch(epoch)
      print(f'EPOCH: {epoch}')
      print(f'Mean loss training: {mean(epoch_losses)}')
      print(f'Mean loss validation: {mean(epoch_vlosses)}')
      #wandb.log({'train_loss': mean(epoch_losses)})
      wandb.log({'loss': mean(epoch_vlosses)})
      wandb.log({'acc': mean(accuracies)})
    torch.save(self.model.state_dict(), os.path.join(self.model_folder, self.model_name))
    wandb.finish()

  def test_on_single_image(self, img, prompt):
    self.model.to(self.device)
    inputs = self.processor(img, input_boxes=[[prompt]], return_tensors="pt").to(self.device)
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(**inputs, multimask_output=False)
    sam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
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
    x_min, y_min, x_max, y_max = prompt
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    overlay = cv2.addWeighted(mask, 0.5, img[:,:,::-1], 0.5, 0)
    Image.fromarray(cv2.rectangle(overlay, (x_min, y_min),(x_max, y_max), color=(255, 0, 0), thickness=2)).save(os.path.join(out))

  def test_on_dataset(self, data_folder, out):
    if not os.path.exists(out):
      os.makedirs(out)
    aff_all, bbox_arr, images_arr = self.prepare_data(data_folder)
    affs = torch.cat(aff_all)
    bboxes = torch.tensor(bbox_arr)
    images = torch.tensor(images_arr)

    if not os.path.exists(os.path.join(out, "predictions")):
      os.makedirs(os.path.join(out, "predictions"))
    if not os.path.exists(os.path.join(out, "gt")):
      os.makedirs(os.path.join(out, "gt"))

    for i in range(images.size()[0]):
      img = images[i]
      bbox = bboxes[i]
      x_min, y_min, x_max, y_max = bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]
      prompt = [x_min, y_min, x_max, y_max]
      mask = self.test_on_single_image(img, prompt)
      out_path_pred = os.path.join(out, "predictions", str(i) +  ".jpg")
      out_path_gt = os.path.join(out, "gt", str(i) + ".jpg")
      aff_mask = cv2.cvtColor(affs[i].cpu().detach().numpy().astype(images_arr[i].dtype), cv2.COLOR_GRAY2BGR)
      self.save_result(images_arr[i], mask, prompt, out_path_pred)
      self.save_result(images_arr[i], aff_mask, prompt, out_path_gt)

  def load_pretrained_weights(self, checkpoint_file):
    self.model = torch.jit.load(checkpoint_file)

def get_bbox(mask, bbox_eps):
  y_indices, x_indices, _ = np.where(mask > 0)
  x_max = max(x_indices) + bbox_eps
  x_min = min(x_indices) - bbox_eps
  y_max = max(y_indices) + bbox_eps
  y_min = min(y_indices) - bbox_eps
  return [x_min, y_min, x_max, y_max]

def main(data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, mask_path):

  # Download pretrained model
  model = SamModel.from_pretrained("facebook/sam-vit-base")
  if checkpoint_file != "":
    checkpoint = torch.load(checkpoint_file, weights_only=True)
    model.load_state_dict(checkpoint)

  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

  # Instantiate affordance detection object
  haff_model = AffordanceDetection(model, processor, bbox_eps, batchsize, model_folder, model_name, num_epochs, visualization_active)

  if not test_single:
    # Prepare data data
    aff_all, bbox_arr, images_arr = haff_model.prepare_data(data_folder)

    # Tensors
    affs = torch.cat(aff_all)
    bboxes = torch.tensor(bbox_arr)
    images = torch.tensor(images_arr)

    # Training
    if not testing:
      haff_model.train(images, affs, bboxes)
    else:
      haff_model.test_on_dataset(data_folder, out_images)
  else:
    if not os.path.exists(out_images):
      os.makedirs(out_images)
    out = os.path.join(out_images, "out.png")
    img = cv2.imread(img_path)
    img_tensor = torch.tensor(img)
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    max_shape = max([img.shape[0], img.shape[1]])
    img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
    mask = cv2.copyMakeBorder(mask, max_shape - mask.shape[0], 0, max_shape - mask.shape[1], 0, cv2.BORDER_CONSTANT)
    prompt = get_bbox(mask, bbox_eps)
    pred = haff_model.test_on_single_image(img_tensor, prompt)
    #print("Image Shape: ", img.shape)
    #print("Mask Shape: ", mask.shape)
    #print("Pred Shape: ", pred.shape)
    print(np.where(pred != 0))
    haff_model.save_result(img, pred, prompt, out)

       
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--bbox-eps', default=20, type=int, help='padding added on top of the bounding box')
  parser.add_argument('--data-folder', default=None, help='directory where the images for training are located')
  parser.add_argument('--visualization-active', default=False, type=bool, help='parameter to indicate whether the data points should be visualized or not')
  parser.add_argument('--model-folder', default=None, help='path to the model folder')
  parser.add_argument('--batchsize', default=16, type=int)
  parser.add_argument('--checkpoint', default="")
  parser.add_argument('--out-images', default=None, help='ouput directory')
  parser.add_argument('--num-epochs', default=100, type=int)
  parser.add_argument('--test', default=False, type=bool)
  parser.add_argument('--out', default=None)
  parser.add_argument('--model-name', default="bowl_model.pth")
  parser.add_argument('--test-single', default=False, type=bool)
  parser.add_argument('--img', default=None)
  parser.add_argument('--prompt', default=None)
  args = parser.parse_args()
  vals = vars(args)
  if (vals["test"] and vals["data_folder"] != None and vals["checkpoint"] != None and vals["out"] != None) or (vals["test_single"] and vals["checkpoint"] != None and vals["img"] != None and vals["prompt"] != None and vals["out"] != None) or (vals["data_folder"] != None and vals["model_folder"] != None):
    data_folder = vals["data_folder"]
    bbox_eps = vals["bbox_eps"]
    visualization_active = vals["visualization_active"]
    model_folder = vals["model_folder"]
    batchsize = vals["batchsize"]
    checkpoint_file = vals["checkpoint"]
    num_epochs = vals["num_epochs"]
    testing = vals["test"]
    out_images = vals["out"]
    model_name = vals["model_name"]
    img_path = vals["img"]
    mask_path = vals["prompt"]
    test_single = vals["test_single"]
    main(data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, mask_path)
  else:
    print("Wrong Arguments!")
