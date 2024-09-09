import argparse
import logging
import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pickle
import PIL.Image
import torch
from PIL import Image
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN
from lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)

warnings.filterwarnings("ignore")


def read_mask(mask_path, invert=False):
    mask = Image.open(mask_path)
    #mask = resize(mask, max_size=512, interpolation=Image.NEAREST)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")


def resize(image, max_size, interpolation=Image.BICUBIC):
    w, h = image.size
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
    return image


def preprocess(img: Image, mask: Image, resolution: int) -> torch.Tensor:
    img = img.resize((resolution, resolution), Image.BICUBIC)
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    img = np.array(img)
    mask = np.array(mask)[:, :, np.newaxis] // 255
    img = torch.Tensor(img).float() * 2 / 255 - 1
    mask = torch.Tensor(mask).float()
    print("img tensor shape: ", img.size())
    print("mask tensor shape: ", mask.size())
    img = img.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)
    x = torch.cat([mask - 0.5, img * mask], dim=1)
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="One of [migan-256, migan-512, comodgan-256, comodgan-512]", required=True)
    parser.add_argument("--model-path", type=str, help="Saved model path.", required=True)
    parser.add_argument("--images-dir", type=Path, help="Path to images directory.", required=True)
    parser.add_argument("--masks-dir", type=Path, help="Path to masks directory.", required=True)
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask? (make 0-known, 1-hole)")
    parser.add_argument("--output-dir", type=Path, help="Output directory.", required=True)
    parser.add_argument("--device", type=str, help="Device.", default="cuda")
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cuda = False
    if args.device == "cuda":
        cuda = True

    if args.model_name == "migan-256":
        resolution = 256
        model = MIGAN(resolution=256)
    elif args.model_name == "migan-512":
        resolution = 512
        model = MIGAN(resolution=512)
    elif args.model_name == "comodgan-256":
        resolution = 256
        comodgan_mapping = CoModGANMapping(num_ws=14)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    elif args.model_name == "comodgan-512":
        resolution = 512
        comodgan_mapping = CoModGANMapping(num_ws=16)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    else:
        raise Exception("Unsupported model name.")

    model.load_state_dict(torch.load(args.model_path))
    if cuda:
        model = model.to("cuda")
    model.eval()

    mask_dirs = os.listdir(args.masks_dir)
    for mask_dir in mask_dirs:
        
        xs = []
        #img_extensions = {".jpg", ".jpeg", ".png"}
        #img_paths = []
        mask_extensions = {".png"}
        mask_paths = []
        imgs = []
        masks = []
        for mask_extension in mask_extensions:
            mask_paths += glob(os.path.join(args.masks_dir, mask_dir, "**", f"*{mask_extension}"), recursive=True)
        print("mask paths: ", mask_paths)
        mask_paths = sorted(mask_paths)
        for mask_path in tqdm(mask_paths):
            img_path = os.path.join(args.images_dir, "".join(os.path.basename(mask_path).split('.')[:-1]) + ".jpg")
            print("mask path: ", mask_path)
            print("image path: ", img_path)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            #img_resized = resize(img, max_size=resolution)
            img_resized = img
            mask = read_mask(mask_path, invert=args.invert_mask)
            print("mask before: ", np.array(mask).shape)
            print("img before: ", np.array(img).shape)
            #mask_resized = resize(mask, max_size=resolution, interpolation=Image.NEAREST)
            #mask_cvt = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)
            # print("mask after recolor: ", mask_cvt.shape)
            mask_resized = cv2.resize(np.array(mask), (np.array(img_resized).shape[1], np.array(img_resized).shape[0]))
            imgs.append(np.array(img_resized))
            masks.append(np.array(mask_resized))
            #print(mask_resized.shape)
            print(resolution)
            print("mask after resize: ", mask_resized.shape)
            x = preprocess(img_resized, Image.fromarray(mask_resized), resolution)
            xs.append(x)
        if len(xs) == 0:
            continue
        x = torch.cat(xs)
        #print("masks shape ", np.array(masks).shape)
        print("input shape ", x.shape)
        if cuda:
            x = x.to("cuda")
        with torch.no_grad():
            result_images = model(x)
            print("result_images shape before clamping ", result_images.shape)
        result_images = (result_images * 0.5 + 0.5).clamp(0, 1) * 255
        print("result_images shape before permutation ", result_images.shape)
        result_images = result_images.to(torch.uint8).permute(0, 2, 3, 1).detach().to("cpu").numpy()
        print("result_images shape ", result_images.shape)
        for i, result_image in enumerate(result_images):
            print("result_image size ", result_image.shape)
            result_image = cv2.resize(result_image, dsize=img_resized.size[:2], interpolation=cv2.INTER_CUBIC)
            print("result_image size after resize ", result_image.shape)
            mask_resized_new = np.array(masks[i])[:, :, np.newaxis] // 255
            composed_img = imgs[i] * mask_resized_new + result_image * (1 - mask_resized_new)
            composed_img = Image.fromarray(composed_img)
            composed_img.save(args.output_dir / f"{Path(mask_paths[i]).stem}.png")


if __name__ == '__main__':
    main()
