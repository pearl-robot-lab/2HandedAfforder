import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# Define paths
input_image = './example_0053376.png'
annotation_file = 'example_export_coco_0053376.json'
output_dir = '.'
output_prefix = './masked_'

# Load COCO annotations
coco = COCO(annotation_file)

# Load image
image = cv2.imread(input_image)
height, width = image.shape[:2]

# Create a blank mask
mask = np.zeros((height, width), dtype=np.uint8)

# Get all category ids
cat_ids = coco.getCatIds()
# Iterate over all categories
for cat_id in cat_ids:
    # Get category name
    cat_name = coco.loadCats(cat_id)[0]['name']
    
    # Get all annotation ids for the current category
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    
    # Create a blank mask for the current category
    cat_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate over all annotations
    for ann in anns:
        # Get the segmentation mask using annToMask
        m = coco.annToMask(ann)
        
        # Combine the mask with the current annotation mask
        cat_mask = np.maximum(cat_mask, m * 255)
    
    # Save the category mask image
    output_path = os.path.join(output_dir, f"{output_prefix}cat_{cat_name}_{os.path.basename(input_image)}")
    cv2.imwrite(output_path, cat_mask)
