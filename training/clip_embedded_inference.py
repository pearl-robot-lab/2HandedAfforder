import clip
import cv2
import numpy as np
import torch

from argparse import ArgumentParser
from PIL import Image
from segment_anything import build_sam_vit_l, sam_model_registry, SamAutomaticMaskGenerator
from transformers import pipeline, SamModel, SamProcessor

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

@torch.no_grad()
def retriev(elements, search_text, model, preprocess) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def main(checkpoint, img_path, prompt, out):
    #sam = SamModel.from_pretrained("facebook/sam-vit-base")
    #weights = torch.load(checkpoint, weights_only=True)
    #sam.load_state_dict(weights)
    #mask_generator = SamAutomaticMaskGenerator(sam)
    
    #mask_generator = SamAutomaticMaskGenerator(build_sam_vit_l(checkpoint=checkpoint))

    device = "cuda"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)   
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    # Cut out all masks
    image = Image.open(image_path)
    cropped_boxes = []

    for mask in masks:
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

    # Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    scores = retriev(cropped_boxes, prompt, model, preprocesss)
    indices = get_indices_of_values_above_threshold(scores, 0.05)

    segmentation_masks = []

    for seg_idx in indices:
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        segmentation_masks.append(segmentation_mask_image)

    original_image = Image.open(image_path)
    overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_color = (255, 0, 0, 200)

    draw = ImageDraw.Draw(overlay_image)
    for segmentation_mask_image in segmentation_masks:
        draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

    dir_path = os.path.dirname(out)
    if not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(dir_path))
    result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image).save(out)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--img', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--out', default=None)

    args = parser.parse_args()
    vals = vars(args)

    if vals['checkpoint'] != None and vals['img'] != None and vals['prompt'] != None and vals['out'] != None:
        checkpoint = vals['checkpoint']
        img_path = vals['img']
        prompt = vals['prompt']
        out = vals['out']
        main(checkpoint, img_path, prompt, out)