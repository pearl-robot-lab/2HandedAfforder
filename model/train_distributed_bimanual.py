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
from PIL import Image
from statistics import mean
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from segment_anything_copy import build_sam_vit_b
from segment_anything_copy.modeling import Sam
from transformers import SamProcessor
from typing import Optional, Tuple
from aff_model_new import TwoHandedAfforder
from dataset_new import AffDataset

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def prepare_data(data_folder, model_folder, bbox_eps):
    files = os.listdir(data_folder)
    affs_left = []
    affs_right = []
    bboxes = []
    images = []
    taxonomies = []
    text_prompts = []
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
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
            text = np.array(f[a_group_key]["narration"])
            imgs = np.array(f[a_group_key]["inpainted"])
            taxonomy = np.array(f[a_group_key]["taxonomy"])
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
        for i in range(aff_left.shape[0]):
            if is_valid(aff_left[i], aff_right[i]) == 'left':
                taxonomies.append(np.array([1, 0, 0, 0]))
            elif is_valid(aff_left[i], aff_right[i]) == 'right':
                taxonomies.append(np.array([0, 1, 0, 0]))
            else:
                if np.array_equal(taxonomy[i], np.array([0, 1, 0])):
                    taxonomies.append(np.array([0, 0, 1, 0]))
                else:
                    taxonomies.append(np.array([0, 0, 0, 1]))

        #aff = extract_tensor(aff_left, aff_right)
        for i in range(aff_left.shape[0]):
            affs_left.append(cv2.cvtColor(aff_left[i], cv2.COLOR_BGR2GRAY))
            affs_right.append(cv2.cvtColor(aff_right[i], cv2.COLOR_BGR2GRAY))
        #affs.append(aff)
        text_prompts.extend(text.tolist())
    # print("Number of Text Prompts: ", text_prompts)
    #aff_all = torch.cat(affs)
    #aff_arr = aff_all.cpu().detach().numpy()
    affs_left_arr = np.array(affs_left)
    affs_right_arr = np.array(affs_right)
    bbox_arr = np.array(bboxes)
    images_arr = np.concatenate(images)
    taxonomies_arr = np.array(taxonomies)
    #print("Affs Left: ", affs_left_arr.shape)
    #print("Affs Right: ", affs_right_arr.shape)
    #print("Taxonomies: ", taxonomies_arr.shape)
    #print("Images: ", images_arr.shape)
    """
    if self.visualization_active:
        for i in range(aff_arr.shape[0]):
        self.visualize_datapoint(images_arr[i], bbox_arr[i], aff_arr[i])
    """
    return affs_left_arr, affs_right_arr, bbox_arr, images_arr, text_prompts, taxonomies_arr

def is_valid(aff_left, aff_right):
    assert aff_left.shape == aff_right.shape
    if (not np.any(aff_left) and np.any(aff_right)) or (not np.any(aff_right) and np.any(aff_left)):
        return "right" if np.any(aff_right) else "left"
    elif not np.any(aff_left) and not np.any(aff_right):
        print("found invalid datapoint")
    return "both"
    
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
            aff.append(cv2.cvtColor(aff_left[i], cv2.COLOR_BGR2GRAY))
    aff = torch.tensor(np.array(aff))
    return aff

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss

class AffordanceDetection():
    def __init__(self, model, processor, bbox_eps, batchsize, model_folder, model_name, num_epochs, device, train, visualization_active=False):
        self.model = model
        self.processor = processor
        self.bbox_eps = bbox_eps
        self.batchsize = batchsize
        self.model_folder = model_folder
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.visualization_active = visualization_active
        self.invalid_counter = 0
        self.trace_created = False

        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = Adam([{'params': self.model.module.sam.mask_decoder_left.parameters()}, {'params':self.model.module.sam.mask_decoder_right.parameters()}, {'params': self.model.module.text_hidden_fcs.parameters()}], lr=1e-5, weight_decay=0)
        self.train_loader = None
        self.validation_loader = None
        self.scaler = torch.cuda.amp.GradScaler()

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

    def train_one_epoch(self, epoch_index):
        #print("Train one epoch")
        epoch_losses = []
        #print("Before Loading")
        self.train_loader.sampler.set_epoch(epoch_index)
        #print("After Loading")
        for batch in tqdm(self.train_loader):
            #print("Start training")
            """
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
                with record_function("model_inference"):
                    outputs = self.model(batch)
                prof.step()
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            if not os.path.exists("debug"):
                os.makedirs("debug")
            prof.export_chrome_trace("debug/" + str(self.device) + "_trace.json")
            """
            with torch.cuda.amp.autocast():
                if self.device == 0:
                    start = time.time()
                outputs = self.model(batch)
                if self.device == 0:
                    forward_time = time.time() - start
                #print("After inference")
                pred_masks_left = torch.cat([x["masks_left"] for x in outputs]).float()
                pred_masks_right = torch.cat([x["masks_right"] for x in outputs]).float()
                #print("After pred_mask calc")
                gt_masks_left = batch["ground_truth_mask_left"].unsqueeze(1).float() / 255
                gt_masks_right = batch["ground_truth_mask_right"].unsqueeze(1).float() / 255
                #print("After gt_mask calc")
                #loss_left = self.seg_loss(pred_masks_left, gt_masks_left.to(self.device))
                #loss_right = self.seg_loss(pred_masks_right, gt_masks_right.to(self.device))
                #loss_left = sigmoid_focal_loss(pred_masks_left, gt_masks_left.to(self.device)).mean((1,2,3))
                #loss_right = sigmoid_focal_loss(pred_masks_right, gt_masks_right.to(self.device)).mean((1,2,3))
                pred_taxonomy = torch.cat([x["taxonomy"] for x in outputs]).float()
                #print("Pred_Taxonomy Size: ", pred_taxonomy.size())
                gt_taxonomy = batch["taxonomies_gt"].float().to(self.device)
                loss_taxonomy = self.ce_loss(pred_taxonomy, gt_taxonomy.to(self.device))
                #print("Loss taxonomy: ", loss_taxonomy.size())
                #print("Loss Left: ", loss_left.size())
                
                weight_decoder_left = gt_taxonomy[:, 0] #.view(-1, 1, 1, 1)
                #print("weight decoder: ", weight_decoder_left.size())
                weight_decoder_right = gt_taxonomy[:, 1] #.view(-1, 1, 1, 1)
                weight_decoder_both = (gt_taxonomy[:, 2] + gt_taxonomy[:, 3]) #.view(-1, 1, 1, 1)
                #loss = weight_decoder_left * loss_left + weight_decoder_right * loss_right + weight_decoder_both * (loss_left + loss_right) #+ loss_taxonomy
                print("Weight decoder shape: ", weight_decoder_left.size())
                print("Pred Masks Left SHape: ", pred_masks_left.size())
                filtered_pred_left = (weight_decoder_left.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_left
                filtered_pred_right = (weight_decoder_right.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_right
                loss_left = self.seg_loss(filtered_pred_left, gt_masks_left.to(self.device))
                loss_right = self.seg_loss(filtered_pred_right, gt_masks_right.to(self.device))
                loss = loss_left + loss_right + loss_taxonomy
                loss = loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_losses.append(loss.item())
        return epoch_losses

    def validate_one_epoch(self, epoch_index):
        epoch_vlosses = []
        self.validation_loader.sampler.set_epoch(epoch_index)
        accuracies = []
        with torch.no_grad():
            for batch in tqdm(self.validation_loader):
                outputs = self.model(batch)
                pred_masks_left = torch.cat([x["masks_left"] for x in outputs]).float()
                pred_masks_right = torch.cat([x["masks_right"] for x in outputs]).float()
                gt_masks_left = batch["ground_truth_mask_left"].unsqueeze(1).float() / 255
                gt_masks_right = batch["ground_truth_mask_right"].unsqueeze(1).float() / 255
                #vloss = self.seg_loss(pred_masks, gt_masks.to(self.device))
                #vloss_left = sigmoid_focal_loss(pred_masks_left, gt_masks_left.to(self.device)).mean((1,2,3))
                #vloss_right = sigmoid_focal_loss(pred_masks_left, gt_masks_left.to(self.device)).mean((1,2,3))

                pred_taxonomy = torch.cat([x["taxonomy"] for x in outputs]).float()
                gt_taxonomy = batch["taxonomies_gt"].float().to(self.device)
                vloss_taxonomy = self.ce_loss(pred_taxonomy, gt_taxonomy.to(self.device))

                weight_decoder_left = gt_taxonomy[:, 0]
                weight_decoder_right = gt_taxonomy[:, 1]
                weight_decoder_both = gt_taxonomy[:, 2] + gt_taxonomy[:, 3]

                #vloss = weight_decoder_left * vloss_left + weight_decoder_right * vloss_right + weight_decoder_both * (vloss_left + vloss_right) + vloss_taxonomy
                #vloss = vloss.mean()
                filtered_pred_left = (weight_decoder_left.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_left
                filtered_pred_right = (weight_decoder_right.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_right
                #acc = monai.metrics.compute_iou(pred_masks, gt_masks.to(self.device)).mean()
                #print("Pred mask shape: ", pred_masks_left.shape)
                #print("GT Mask Shape: ", gt_masks_left.shape)
                #print("pred_masks_left min: ", torch.min(pred_masks_left))
                #print("Pred masks left max: ", torch.max(pred_masks_left))
                pred_masks_left = torch.sigmoid(pred_masks_left)
                pred_masks_right = torch.sigmoid(pred_masks_right)
                vloss_left = self.seg_loss(filtered_pred_left, gt_masks_left.to(self.device))
                vloss_right = self.seg_loss(filtered_pred_right, gt_masks_right.to(self.device))
                vloss = vloss_left + vloss_right + vloss_taxonomy
                vloss = vloss.mean()
                #pred_masks_left = (pred_masks_left > 0.5).float()
                #pred_masks_right = (pred_masks_right > 0.5).float()
                #print("pred_masks_left min: ", torch.min(pred_masks_left))
                #print("Pred masks left max: ", torch.max(pred_masks_left))
                acc_left = monai.metrics.compute_iou(pred_masks_left, gt_masks_left.to(self.device)).nan_to_num()
                acc_right = monai.metrics.compute_iou(pred_masks_right, gt_masks_right.to(self.device)).nan_to_num()
                acc = weight_decoder_left * acc_left + weight_decoder_right * acc_right + weight_decoder_both * (acc_left + acc_right)
                acc = acc.mean()
                #acc_left = compute_iou(pred_masks_left, gt_masks_left.to(self.device))
                #acc_right = compute_iou(pred_masks_right, gt_masks_right.to(self.device))
                #print("ACc Left: ", acc_left)
                #print("Acc Right: ", acc_right)
                epoch_vlosses.append(vloss.item())
                #acc = monai.metric.compute_iou(pred_masks, ground_truth_masks.unsqueeze(1))
                accuracies.append(acc.item())
        return epoch_vlosses, accuracies


    def train(self, dset):
        torch.manual_seed(0)
        validation_split = .2
        shuffle_dataset = True
        random_seed = 42
        if self.device == 0:
            wandb.init(
                    project="2HAff_Bimanual",
                    config={
                        "dataset": "EPIC_Affordance",
                        "epochs": self.num_epochs,
                        "batchsize": self.batchsize
                    }
                )
        
        """
        dataset_size = len(dset)
        indices =list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        """
        train_set, val_set = torch.utils.data.random_split(dset, [0.9, 0.1])
        #print("Before train and val loader")
        self.train_loader = DataLoader(train_set, batch_size=self.batchsize, shuffle=False, sampler=DistributedSampler(train_set), num_workers=8, pin_memory=False, multiprocessing_context="fork")
        self.validation_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False, sampler=DistributedSampler(val_set), num_workers=8, pin_memory=False, multiprocessing_context="fork")
        #print("After train val loader")
        self.model.to(self.device)
        #print("AFter model to device")
        for epoch in range(self.num_epochs):
            #print("In for loop")
            self.model.train()
            #print("Call train one epoch")
            epoch_losses = self.train_one_epoch(epoch)
            self.model.eval()
            epoch_vlosses, accuracies = self.validate_one_epoch(epoch)
            #print(f'EPOCH: {epoch}')
            #print(f'Mean loss training: {mean(epoch_losses)}')
            #print(f'Mean loss validation: {mean(epoch_vlosses)}')
            #print(f'Mean IoU: {mean(accuracies)}')
            if self.device == 0:
                wandb.log({'train_loss': mean(epoch_losses)})
                wandb.log({'loss': mean(epoch_vlosses)})
                wandb.log({'acc (IoU)': mean(accuracies)})
            #wandb.log({'acc': mean(accuracies)})
            if self.device == 0:
                if epoch % 10 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.model_folder, str(epoch) + '_' + self.model_name))
        torch.save(self.model.module.state_dict(), os.path.join(self.model_folder, self.model_name))
        if self.device == 0:
            wandb.finish()

    def test_on_single_image(self, img, prompt):
        self.model.to(self.device)
        inputs = self.processor(img, input_boxes=None, return_tensors="pt").to(self.device)
        inputs["image"] = inputs["pixel_values"].squeeze(0)
        inputs["txt_prompt"] = [prompt]
        print("original size shape: ", inputs["original_sizes"].size())
        print("original size: ", inputs["original_sizes"])
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
        aff_all, bbox_arr, images_arr, text_prompts = self.prepare_data(data_folder)
        affs = torch.cat(aff_all)
        bboxes = torch.tensor(bbox_arr)
        images = torch.tensor(images_arr)

        if not os.path.exists(os.path.join(out, "predictions")):
            os.makedirs(os.path.join(out, "predictions"))
        if not os.path.exists(os.path.join(out, "gt")):
            os.makedirs(os.path.join(out, "gt"))

        for i in range(images.size()[0]):
            img = images[i]
            """
            bbox = bboxes[i]
            x_min, y_min, x_max, y_max = bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]
            prompt = [x_min, y_min, x_max, y_max]
            """
            prompt = text_prompts[i]
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

def main(rank, world_size, data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, prompt, image_name, dataset):

    # Download pretrained model
    if not testing and not test_single:
        ddp_setup(rank, world_size)
        sam_model = build_sam_vit_b(checkpoint="pretrained/sam_vit_b_01ec64.pth", is_sam_pretrained=True)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        config = {
        "in_dim": 512,
        "out_dim": 256
        }
        clip_sam = TwoHandedAfforder(sam_model, config, rank)
        clip_sam.to(rank)
        clip_sam = DDP(clip_sam, device_ids=[rank], find_unused_parameters=True)
    else:
        sam_model = build_sam_vit_b("pretrained/sam_vit_b_01ec64.pth")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base", is_sam_pretrained=True)
        config = {
        "in_dim": 512,
        "out_dim": 256
        }
        clip_sam =  TwoHandedAfforder(sam_model, config)
        clip_sam.load_state_dict(torch.load(checkpoint_file))
        #clip_sam = nn.DataParallel(clip_sam, device_ids=[rank])

    # Instantiate affordance detection object
    haff_model = AffordanceDetection(clip_sam, processor, bbox_eps, batchsize, model_folder, model_name, num_epochs, rank, not (not testing and not test_single), visualization_active)

    if not test_single:
        # Prepare data data
        # aff_all, bbox_arr, images_arr, text_prompts = haff_model.prepare_data(data_folder)

        # Tensors
        #affs = torch.cat(aff_all)
        #images = torch.tensor(images_arr)

        # Training
        if not testing:
            haff_model.train(dataset)
            destroy_process_group()
        else:
            haff_model.test_on_dataset(data_folder, out_images)
    else:
        if not os.path.exists(out_images):
            os.makedirs(out_images)
        out = os.path.join(out_images, image_name)
        img = cv2.imread(img_path)
        img_tensor = torch.tensor(img)
        #mask = cv2.imread(mask_path)
        #mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        max_shape = max([img.shape[0], img.shape[1]])
        img = cv2.copyMakeBorder(img, max_shape - img.shape[0], 0, max_shape - img.shape[1], 0, cv2.BORDER_CONSTANT)
        #mask = cv2.copyMakeBorder(mask, max_shape - mask.shape[0], 0, max_shape - mask.shape[1], 0, cv2.BORDER_CONSTANT)
        #prompt = get_bbox(mask, bbox_eps)
        pred = haff_model.test_on_single_image(img_tensor, prompt)
        #print("Image Shape: ", img.shape)
        #print("Mask Shape: ", mask.shape)
        #print("Pred Shape: ", pred.shape)
        #print(np.where(pred != 0))
        haff_model.save_result(img, pred, prompt, out)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--bbox-eps', default=20, type=int, help='padding added on top of the bounding box')
    parser.add_argument('--data-folder', default=None, help='directory where the images for training are located')
    parser.add_argument('--visualization-active', default=False, type=bool, help='parameter to indicate whether the data points should be visualized or not')
    parser.add_argument('--model-folder', default=None, help='path to the model folder')
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--checkpoint', default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument('--out-images', default=None, help='ouput directory')
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--out', default=None)
    parser.add_argument('--model-name', default="bowl_model.pth")
    parser.add_argument('--test-single', default=False, type=bool)
    parser.add_argument('--img', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--image-name', default=None)
    args = parser.parse_args()
    vals = vars(args)
    if (vals["test"] and vals["data_folder"] != None and vals["checkpoint"] != None and vals["out"] != None) or (vals["test_single"] and vals["checkpoint"] != None and vals["img"] != None and vals["prompt"] != None and vals["out"] != None and vals["image_name"] != None) or (vals["data_folder"] != None and vals["model_folder"] != None):
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
        prompt = vals["prompt"]
        test_single = vals["test_single"]
        image_name = vals["image_name"]
        world_size = torch.cuda.device_count()
        affs_left, affs_right, bbox, images, text_prompts, taxonomies = prepare_data(data_folder, model_folder, bbox_eps)
        images = torch.tensor(images)
        affs_left = torch.tensor(affs_left)
        affs_right = torch.tensor(affs_right)
        taxonomies = torch.tensor(taxonomies)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        dset = AffDataset(img=images, aff_mask_left=affs_left, aff_mask_right=affs_right, txt_prompt=text_prompts, taxonomies=taxonomies, processor=processor)
        mp.spawn(main, args=(world_size, data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, prompt, image_name, dset, ), nprocs=world_size)
        #main(data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, prompt, image_name)
    else:
        print("Wrong Arguments!")
