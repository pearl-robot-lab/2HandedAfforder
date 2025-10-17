import argparse
import os
import sys

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="sjauhri/2HAff")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
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

    parser.add_argument("--benchmark-dir", default=None, type=str, help='directory containing subfolders of benchmark examples')
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

def min_max_normalize_2d(data):
    """
    Perform Min-Max normalization on a 2D image mask (NumPy array).
    
    Parameters:
    - data: 2D numpy array representing the image mask to be normalized.
    
    Returns:
    - normalized_data: 2D numpy array with Min-Max normalized values.
    """
    # Ensure the data is a numpy array
    data = np.array(data)  
    
    # Get the minimum and maximum values in the 2D mask
    min_value = data.min()
    max_value = data.max()
    
    # Apply the Min-Max normalization formula
    normalized_data = (data - min_value) / (max_value - min_value)
    
    return normalized_data

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(args):
    args = parse_args(args)
    #os.makedirs(args.vis_save_path, exist_ok=True)

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


    th_list = [0.1, 0.2, 0.3, 0.5, 0.7]
    model.eval()
    for dir_name in os.listdir(args.benchmark_dir):
        dir_path = os.path.join(args.benchmark_dir, dir_name)
        folders = os.listdir(dir_path)
        for folder_name in folders:
            folder_path = os.path.join(dir_path, folder_name)
            
            if not os.path.isdir(folder_path):
                continue  # Skip non-directory files

            image_path = os.path.join(folder_path, 'inpainting.png')
            #image_path = os.path.join(folder_path, 'inpainting.png')
            annotation_path = os.path.join(folder_path, 'annotation.json')
            
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                print(f"Required files not found in {folder_path}, skipping...")
                continue

            # Load the narration from the annotation.json
            with open(annotation_path, 'r') as annotation_file:
                annotation_data = json.load(annotation_file)
                prompt = annotation_data.get('narration', '')

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

            #text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            #text_output = text_output.replace("\n", "").replace("  ", " ")
            #print("text_output: ", text_output)
            
            taxonomy = taxonomies[0]
            if taxonomy.numel() != 0:
                if torch.argmax(taxonomy) != 1:
                    for i, pred_mask in enumerate(pred_masks_left):
                        if pred_mask.shape[0] == 0:
                            continue
                        
                        pred_mask = torch.sigmoid(pred_mask)
                        pred_mask = pred_mask.detach().cpu().numpy()[0]
                        
                        #pred_mask[pred_mask < -0.5] = 0
                        #heatmap = create_heatmap(pred_mask)
                        #heatmap_save_path = os.path.join(args.vis_save_path, dir_name, folder_name, 'aff_left_heat.png')
                        #os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                        #cv2.imwrite(heatmap_save_path, heatmap)
                        #pred_mask = min_max_normalize_2d(pred_mask)
                        #pred_mask[pred_mask > 0.5] = 255
                        #pred_mask[pred_mask <= 0.5] = 0
                        for th in th_list:
                            th_pred = np.zeros_like(pred_mask)
                            th_pred[pred_mask > th] = 255

                            # Save the mask
                            mask_save_path = os.path.join(args.vis_save_path + str(th), dir_name, folder_name, 'aff_left.png')
                            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                            cv2.imwrite(mask_save_path, th_pred)
                            print(f"{mask_save_path} has been saved.")

                        # Save the masked image
                        # masked_img_save_path = os.path.join(args.vis_save_path, folder_name, 'aff_left_masked_img_{}.jpg'.format(i))
                        # save_img = image_np.copy()
                        # save_img[pred_mask] = (
                        #     image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                        # )[pred_mask]
                        # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(masked_img_save_path, save_img)
                        # print(f"{masked_img_save_path} has been saved.")
                if torch.argmax(taxonomy) != 0:
                    for i, pred_mask in enumerate(pred_masks_right):
                        if pred_mask.shape[0] == 0:
                            continue
                        pred_mask = torch.sigmoid(pred_mask)
                        pred_mask = pred_mask.detach().cpu().numpy()[0]
                        #pred_mask[pred_mask < 0] = 0
                        #heatmap = create_heatmap(pred_mask)
                        #heatmap_save_path = os.path.join(args.vis_save_path, dir_name, folder_name, 'aff_right_heat.png')
                        #os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                        #cv2.imwrite(heatmap_save_path, heatmap)
                        #pred_mask = min_max_normalize_2d(pred_mask)
                        for th in th_list:
                            th_pred = np.zeros_like(pred_mask)
                            th_pred[pred_mask > th] = 255
                            #pred_mask[pred_mask <= 0] = 0

                            # Save the mask
                            mask_save_path = os.path.join(args.vis_save_path + str(th), dir_name, folder_name, 'aff_right.png')
                            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                            cv2.imwrite(mask_save_path, th_pred)
                            print(f"{mask_save_path} has been saved.")


if __name__ == "__main__":
    main(sys.argv[1:])
