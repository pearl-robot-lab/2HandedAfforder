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

if __name__ == '__main__':
    prompt = ["Hello World! How are you doing?", "I am doing fine.", "Yes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    print("Features: ", text_features)
    print("Features Size: ", text_features.size())