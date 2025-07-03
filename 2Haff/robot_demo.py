import argparse
import os
import signal
import sys

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from PIL import Image
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

th = 0

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="aff_weights")
    parser.add_argument("--vis_save_path", default="./robot_demo/out", type=str)
    parser.add_argument("--force_left", action="store_true", default=False)
    parser.add_argument("--force_right", action="store_true", default=False)
    parser.add_argument("--force_both", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument("--zed2_img_path", default='robot_demo/in', type=str, help='directory containing subfolders of benchmark examples')
    parser.add_argument('--th', default=-5, type=int)
    return parser.parse_args(args)

def create_heatmap(probability_array, kernel_size=(5, 5), sigma=1):
    # Convert the list of probabilities into a numpy array
    data = np.array(probability_array)

    # Normalize the data to the range 0-255 for image representation
    normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized data to an 8-bit format
    heatmap = np.uint8(normalized_data)

    # Apply a colormap (e.g., JET, which is often used for heatmaps)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Apply Gaussian smoothing to the heatmap
    heatmap_smoothed = cv2.GaussianBlur(heatmap_color, kernel_size, sigma)

    return heatmap_smoothed

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    th = args.th

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    print("Ready")
    while True:
        # confirmation = input("Do you want to change the th? (y/n)")
        # if confirmation.lower() == 'y':
        #     th = int(input("Set the threshold: "))
        image_path = os.path.join(args.zed2_img_path, 'img.png')
        #image_path = os.path.join(folder_path, 'inpainting.png')
        prompt_path = os.path.join(args.zed2_img_path, 'prompt.txt')
        margins_path = os.path.join(args.zed2_img_path, 'margins.txt')
        mask_right_path = os.path.join(args.zed2_img_path, 'mask_right.png')
        mask_left_path = os.path.join(args.zed2_img_path, 'mask_left.png')
        
        
        if not os.path.exists(image_path) or not os.path.exists(prompt_path) or not os.path.exists(margins_path):
            print("Files not found, continuing")
            continue  # Skip non-directory files
        mask_left = None
        mask_right = None
        if os.path.exists(mask_left_path):
            mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(mask_right_path):
            mask_right = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE)
        if mask_left is None and mask_right is None:
            print("Masks not found")
            continue
        # Load the narration from the annotation.json
        with open(prompt_path, 'r') as f:
            prompt = f.readline()
        
        with open(margins_path, 'r') as f:
            margins = f.readline()
            margins = margins.split(',')
            left = int(margins[0])
            top = int(margins[1])
            right = int(margins[2])
            bottom = int(margins[3])

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + "Where would you interact with the object to perform action " + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
        # Load the image
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"]
            [0].unsqueeze(0).cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0).cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks_left, pred_masks_right, taxonomies = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        taxonomy = taxonomies[0]
        if taxonomy.numel() != 0:
            if args.force_left or args.force_both:
            # if torch.argmax(taxonomy) != 1:
                for i, pred_mask in enumerate(pred_masks_left):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    heatmap = create_heatmap(pred_mask)
                    heatmap_save_path = os.path.join(args.vis_save_path, 'aff_left_heat.png')
                    cv2.imwrite(heatmap_save_path, heatmap)
                    pred_mask[pred_mask > th] = 1
                    pred_mask[pred_mask <= th] = 0
                    # Save the mask
                    mask_save_path = os.path.join(args.vis_save_path, 'aff_left.png')
                    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                    pred_mask = Image.fromarray(pred_mask)
                    orig_width, orig_height = pred_mask.size
                    new_width = orig_width + left + right
                    new_height = orig_height + top + bottom
                    padded_mask = Image.new("L", (new_width, new_height), color=0)
                    padded_mask.paste(pred_mask, (left, top))
                    pred_mask = np.array(padded_mask)
                    if mask_left is not None:
                        pred_mask = (cv2.bitwise_and(pred_mask, mask_left)*255).astype(np.uint8)
                    else:
                        pred_mask = (cv2.bitwise_and(pred_mask, mask_right)*255).astype(np.uint8)
                    padded_mask = Image.fromarray(pred_mask)
                    padded_mask.save(mask_save_path)
                    print(f"{mask_save_path} has been saved.")
                    
            if args.force_right or args.force_both:
            # if torch.argmax(taxonomy) != 0:
                for i, pred_mask in enumerate(pred_masks_right):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    heatmap = create_heatmap(pred_mask)
                    heatmap_save_path = os.path.join(args.vis_save_path, 'aff_right_heat.png')
                    cv2.imwrite(heatmap_save_path, heatmap)
                    pred_mask[pred_mask > th] = 1
                    pred_mask[pred_mask <= th] = 0


                    # Save the mask
                    mask_save_path = os.path.join(args.vis_save_path, 'aff_right.png')
                    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                    pred_mask = Image.fromarray(pred_mask)
                    orig_width, orig_height = pred_mask.size
                    new_width = orig_width + left + right
                    new_height = orig_height + top + bottom
                    padded_mask = Image.new("L", (new_width, new_height), color=0)
                    padded_mask.paste(pred_mask, (left, top))
                    pred_mask = np.array(padded_mask)
                    if mask_right is not None:
                        pred_mask = (cv2.bitwise_and(pred_mask, mask_right)*255).astype(np.uint8)
                    else:
                        pred_mask = (cv2.bitwise_and(pred_mask, mask_left)*255).astype(np.uint8)
                    padded_mask = Image.fromarray(pred_mask)
                    padded_mask.save(mask_save_path)
                    print(f"{mask_save_path} has been saved.")
        else:
            print("No taxonomy found!!")
        # Save the cropped input image image_np as well in the out folder
        cropped_img = image_np
        cropped_img_path = os.path.join(args.vis_save_path, 'cropped_img.png')
        cv2.imwrite(cropped_img_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        os.remove(image_path)
        os.remove(prompt_path)
        os.remove(margins_path)


if __name__ == "__main__":
    main(sys.argv[1:])
