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
from segment_anything_copy import build_sam_vit_b
from segment_anything_copy.modeling import Sam
from transformers import SamProcessor
from typing import Optional, Tuple
from aff_model_copy import TwoHandedAfforder
from dataset import AffDataset

if __name__ == '__main__':
    sam_model = build_sam_vit_b(checkpoint='pretrained/sam_vit_b_01ec64.pth', is_sam_pretrained=True)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    config = {
        "in_dim": 512,
        "out_dim": 256
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_sam = TwoHandedAfforder(sam_model, config, device)