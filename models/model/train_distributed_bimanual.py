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
from segment_anything_copy import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from segment_anything_copy.modeling import Sam
from transformers import SamProcessor
from typing import Optional, Tuple
from aff_model_new import TwoHandedAfforder
from dataset_new import AffDataset
from json_mask_handler import get_masks_and_overlay

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
    images = []
    taxonomies = []
    text_prompts = []
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
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
            taxonomy = np.array(f[a_group_key]["taxonomy"])
            if imgs.size == 0:
                continue
            assert obj_left.shape == obj_right.shape
            affs_left.append(aff_left)
            affs_right.append(aff_right)
            images.append(imgs)
            taxonomies.append(taxonomy)
            text_prompts.extend(text.tolist())
    affs_left_arr = np.concatenate(affs_left)
    affs_right_arr = np.concatenate(affs_right)
    images_arr = np.concatenate(images)
    taxonomies_arr = np.concatenate(taxonomies)
    return affs_left_arr, affs_right_arr, images_arr, text_prompts, taxonomies_arr

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

        self.device = device
        self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = Adam([{'params': self.model.module.sam.mask_decoder_left.parameters()}, {'params':self.model.module.sam.mask_decoder_right.parameters()}, {'params': self.model.module.text_hidden_fcs.parameters()}], lr=1e-4, weight_decay=0)
        self.train_loader = None
        self.validation_loader = None
        self.scaler = torch.amp.GradScaler('cuda')

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
        epoch_losses = []
        self.train_loader.sampler.set_epoch(epoch_index)
        for batch in tqdm(self.train_loader):
            with torch.amp.autocast('cuda'):

                outputs = self.model(batch)

                pred_masks_left = torch.cat([x["masks_left"] for x in outputs]).float()
                pred_masks_right = torch.cat([x["masks_right"] for x in outputs]).float()

                gt_masks_left = batch["ground_truth_mask_left"].unsqueeze(1).float() / 255
                gt_masks_right = batch["ground_truth_mask_right"].unsqueeze(1).float() / 255

                pred_taxonomy = torch.cat([x["taxonomy"] for x in outputs]).float()

                gt_taxonomy = batch["taxonomies_gt"].float().to(self.device)
                loss_taxonomy = self.ce_loss(pred_taxonomy, gt_taxonomy.to(self.device))
                
                weight_decoder_left = gt_taxonomy[:, 0]
                weight_decoder_right = gt_taxonomy[:, 1]
                weight_decoder_both = (gt_taxonomy[:, 2] + gt_taxonomy[:, 3])

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

                pred_taxonomy = torch.cat([x["taxonomy"] for x in outputs]).float()
                gt_taxonomy = batch["taxonomies_gt"].float().to(self.device)
                vloss_taxonomy = self.ce_loss(pred_taxonomy, gt_taxonomy.to(self.device))

                weight_decoder_left = gt_taxonomy[:, 0]
                weight_decoder_right = gt_taxonomy[:, 1]
                weight_decoder_both = gt_taxonomy[:, 2] + gt_taxonomy[:, 3]

                filtered_pred_left = (weight_decoder_left.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_left
                filtered_pred_right = (weight_decoder_right.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_right

                pred_masks_left = torch.sigmoid(pred_masks_left)
                pred_masks_right = torch.sigmoid(pred_masks_right)

                vloss_left = self.seg_loss(filtered_pred_left, gt_masks_left.to(self.device))
                vloss_right = self.seg_loss(filtered_pred_right, gt_masks_right.to(self.device))
                vloss = vloss_left + vloss_right + vloss_taxonomy
                vloss = vloss.mean()

                acc_left = monai.metrics.compute_iou(pred_masks_left, gt_masks_left.to(self.device)).nan_to_num()
                acc_right = monai.metrics.compute_iou(pred_masks_right, gt_masks_right.to(self.device)).nan_to_num()
                acc = weight_decoder_left * acc_left + weight_decoder_right * acc_right + weight_decoder_both * (acc_left + acc_right)
                acc = acc.mean()

                epoch_vlosses.append(vloss.item())
                accuracies.append(acc.item())
        return epoch_vlosses, accuracies

    def log_test_image(self, epoch_index):
        self.test_loader.sampler.set_epoch(epoch_index)
        first_batch = True
        results = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                outputs = self.model(batch)
                if first_batch:
                    first_batch = False
                    for i in range(min(5, len(outputs))):
                        if torch.argmax(outputs[i]["taxonomy"]) != 1:
                            sam_seg_prob_left = torch.sigmoid(outputs[i]["masks_left"].squeeze(1))
                            sam_seg_prob_left = sam_seg_prob_left.cpu().numpy().squeeze()
                            sam_seg_left = (sam_seg_prob_left > 0.5).astype(np.uint8)
                            if torch.argmax(outputs[i]["taxonomy"]) == 0:
                                sam_seg_right = np.zeros_like(sam_seg_left)
                        if torch.argmax(outputs[i]["taxonomy"]) != 0:
                            sam_seg_prob_right = torch.sigmoid(outputs[i]["masks_right"].squeeze(1))
                            sam_seg_prob_right = sam_seg_prob_right.cpu().numpy().squeeze()
                            sam_seg_right = (sam_seg_prob_right > 0.5).astype(np.uint8)
                            if torch.argmax(outputs[i]["taxonomy"]) == 1:
                                sam_seg_left = np.zeros_like(sam_seg_right)
                        sam_seg = sam_seg_left + sam_seg_right
                        #print("Batch: ", batch.shape)
                        img = batch["image"][i]
                        img = img.permute((1, 2, 0)).cpu().numpy()  # Change to HWC format
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        def denormalize_image(normalized_image, mean, std):
                            original_image = (normalized_image * std + mean) * 255  # Reverse normalization
                            original_image = np.clip(original_image, 0, 255)  # Clip to valid range
                            original_image = original_image[:,:,::-1]
                            return original_image.astype(np.uint8)
                        img = denormalize_image(img, mean, std)
                        #print("Max Image Torch: ", np.max(img))
                        # Check if img is in the range [0, 1]
                        if img.max() <= 1.0:  
                            img = (img * 255).clip(0, 255)  # Scale to [0, 255]
                            
                        img = img.astype(np.uint8)  # Convert to uint8
                        Image.fromarray(img).save("example_pre.png")  # Save the image
                        mask = self.prepare_mask(sam_seg)
                        mask = cv2.resize(mask.astype(img.dtype), (img.shape[0], img.shape[1]))
                        if max(mask.flatten()) == 1:
                            mask = mask * 255
                        overlay = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
                        Image.fromarray(overlay).save("example.png")
                        results.append(Image.fromarray(overlay))   
        wandb.log({"examples": [wandb.Image(image) for image in results]})
                


    def train(self, dset, use_wandb):
        #print("Starting Training: ", str(self.device))
        torch.manual_seed(0)
        validation_split = .2
        shuffle_dataset = True
        random_seed = 42
        if self.device == 0 and use_wandb:
            wandb.init(
                    project="2HAff_Bimanual",
                    config={
                        "dataset": "EPIC_Affordance",
                        "epochs": self.num_epochs,
                        "batchsize": self.batchsize,
                        "name": "Test"
                    }
                )
        
        train_set, val_set, test_set = torch.utils.data.random_split(dset, [0.85, 0.1, 0.05])
        self.train_loader = DataLoader(train_set, batch_size=self.batchsize, shuffle=False, sampler=DistributedSampler(train_set), num_workers=8, pin_memory=False, multiprocessing_context="fork")
        self.validation_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False, sampler=DistributedSampler(val_set), num_workers=8, pin_memory=False, multiprocessing_context="fork")
        self.test_loader = DataLoader(test_set, batch_size=self.batchsize, shuffle=False, sampler=DistributedSampler(test_set), num_workers=8, pin_memory=False, multiprocessing_context="fork")
        self.model.to(self.device)
        """
        def trace_handler(profiler):
            if self.device == 0:
                profiler.export_chrome_trace("chrome_trace/run_2/trace.json")
                print("Saved the trace")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=False,
            with_stack=False
        ) as profiler:
            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_losses = self.train_one_epoch(epoch)
                self.model.eval()
                epoch_vlosses, accuracies = self.validate_one_epoch(epoch)
                if self.device == 0 and epoch % 20 == 0 and use_wandb:
                    self.log_test_image(epoch)
                if self.device == 0 and use_wandb:
                    wandb.log({'train_loss': mean(epoch_losses)})
                    wandb.log({'val loss': mean(epoch_vlosses)})
                    wandb.log({'IoU': mean(accuracies)})
                if self.device == 0:
                    if epoch % 10 == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.model_folder, str(epoch) + '_' + self.model_name))
                profiler.step()
            """
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = self.train_one_epoch(epoch)
            self.model.eval()
            epoch_vlosses, accuracies = self.validate_one_epoch(epoch)
            if self.device == 0 and epoch % 20 == 0 and use_wandb:
                self.log_test_image(epoch)
            if self.device == 0 and use_wandb:
                wandb.log({'train_loss': mean(epoch_losses)})
                wandb.log({'val loss': mean(epoch_vlosses)})
                wandb.log({'IoU': mean(accuracies)})
            if self.device == 0:
                if epoch % 10 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.model_folder, str(epoch) + '_' + self.model_name))
        torch.save(self.model.module.state_dict(), os.path.join(self.model_folder, self.model_name))
        if self.device == 0 and use_wandb:
            wandb.finish()

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

        for i in range(images.size()[0]):
            img = images[i]
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

def main(rank, world_size, data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, prompt, image_name, dataset, resume, use_wandb):

    # Download pretrained model
    if not testing and not test_single:
        ddp_setup(rank, world_size)
        sam_model = build_sam_vit_b(checkpoint="pretrained/sam_vit_b_01ec64.pth", is_sam_pretrained=True)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        #sam_model = build_sam_vit_h(checkpoint="pretrained/sam_vit_h_4b8939.pth", is_sam_pretrained=True)
        #processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        #print("Loading checkpoint pretrained/sam_vit_l_0b3195.pth")
        #sam_model = build_sam_vit_l(checkpoint="pretrained/sam_vit_l_0b3195.pth", is_sam_pretrained=True)
        #processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        config = {
        "in_dim": 512,
        "out_dim": 256
        }
        clip_sam = TwoHandedAfforder(sam_model, config, rank)
        clip_sam.to(rank)
        if resume:
            clip_sam.load_state_dict(torch.load(resume, weights_only=True), strict=False)
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

    # Instantiate affordance detection object
    haff_model = AffordanceDetection(clip_sam, processor, bbox_eps, batchsize, model_folder, model_name, num_epochs, rank, not (not testing and not test_single), visualization_active)

    if not test_single:
        # Training
        if not testing:
            haff_model.train(dataset, use_wandb)
            destroy_process_group()
        else:
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
    parser.add_argument('--resume', default=None)
    parser.add_argument('--wandb', default=True)
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
        resume = vals["resume"]
        use_wandb = vals["wandb"]
        world_size = torch.cuda.device_count()
        #affs_left, affs_right, images, text_prompts, taxonomies = prepare_data(data_folder, model_folder, bbox_eps)
        #images = torch.tensor(images)
        #affs_left = torch.tensor(affs_left)
        #affs_right = torch.tensor(affs_right)
        #taxonomies = torch.tensor(taxonomies)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        #dset = AffDataset(img=images, aff_mask_left=affs_left, aff_mask_right=affs_right, txt_prompt=text_prompts, taxonomies=taxonomies, processor=processor)
        dset = AffDataset(processor=processor, folder_path=data_folder)
        mp.spawn(main, args=(world_size, data_folder, bbox_eps, checkpoint_file, visualization_active, batchsize, model_folder, model_name, num_epochs, out_images, testing, test_single, img_path, prompt, image_name, dset, resume, use_wandb,), nprocs=world_size)
    else:
        print("Wrong Arguments!")
