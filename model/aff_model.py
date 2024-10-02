import clip
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
from segment_anything import build_sam_vit_b, SamPredictor
from segment_anything.modeling import Sam
from transformers import SamModel, SamProcessor
from typing import Optional, Tuple

class TwoHandedAfforder(nn.Module):
    def __init__(self, sam_model, config):
        super().__init__()
        
        in_dim = config["in_dim"]
        out_dim = config["out_dim"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.Sequential(*text_fc)
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self.sam = sam_model
        self.sam.train()
        for name, param in self.sam.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def convert_dict_to_list(self, tensor_dict):
        # Get the batch size B from one of the tensors in the dictionary
        B = next(iter(tensor_dict.values())).shape[0]  # Get B from the first tensor

        # Initialize a list to store dictionaries for each batch element
        batch_list = []

        # Loop over the batch dimension
        for i in range(B):
            # Create a new dictionary for each batch index
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #print("Tensor Dict Items: ", tensor_dict.items())
            batch_dict = {key: value[i].to(device) if key != "txt_prompt" else value[i] for key, value in tensor_dict.items()}
            # Append the dictionary to the list
            batch_list.append(batch_dict)
        if "original_sizes" in batch_list[0].keys():
            for j in range(len(batch_list)):
                batch_list[j]["original_size"] = (batch_list[j]["original_sizes"][0], batch_list[j]["original_sizes"][1])
        return batch_list
        
    def forward(self, batched_input):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        if not isinstance(batched_input["txt_prompt"][0], str):
            decoded_text = [x.decode("utf-8") for x in batched_input["txt_prompt"]]
        else:
            decoded_text = batched_input["txt_prompt"]
        text = clip.tokenize(decoded_text).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).type(torch.float32)
        text_embeds = self.text_hidden_fcs(text_features)
        batched_input["text_embeds"] = text_embeds
        batched_input = self.convert_dict_to_list(batched_input)
        outputs = self.sam(batched_input, multimask_output=False)

        return outputs
