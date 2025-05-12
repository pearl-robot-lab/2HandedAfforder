import os
import json
import argparse
import random
from PIL import Image
import numpy as np

# Utility function to load JSON file
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Utility function to crop images with padding and resizing
def crop_and_pad(image, bbox, offset, original_size):
    min_x, min_y, max_x, max_y = bbox
    width, height = image.size

    # Apply the offset
    min_x = max(min_x - offset, 0)
    min_y = max(min_y - offset, 0)
    max_x = min(max_x + offset, width)
    max_y = min(max_y + offset, height)
    #print("Crop And Pad")
    print("Min X: ", min_x)
    print("Min Y: ", min_y)
    print("Max X: ", max_x)
    print("Max Y: ", max_y)

    # Crop the image
    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    # Calculate padding to make it square
    cropped_width, cropped_height = cropped_image.size
    max_dim = max(cropped_width, cropped_height)
    padded_image = Image.new("RGB", (max_dim, max_dim))
    padded_image.paste(cropped_image, ((max_dim - cropped_width) // 2, (max_dim - cropped_height) // 2))

    # Resize back to original dimensions
    return padded_image.resize(original_size)

# Function to process unimanual images
def process_unimanual(sequence_path, original_size):
    obj_right_path = os.path.join(sequence_path, 'obj_right.png')
    obj_left_path = os.path.join(sequence_path, 'obj_left.png')
    aff_right_path = os.path.join(sequence_path, 'aff_right.png')
    aff_left_path = os.path.join(sequence_path, 'aff_left.png')
    inpainted_frame_path = os.path.join(sequence_path, 'inpainted_frame.png')

    obj_img, aff_img, side = None, None, None
    if os.path.exists(obj_right_path) and os.path.exists(aff_right_path):
        obj_img = Image.open(obj_right_path)
        aff_img = Image.open(aff_right_path)
        aff_img = aff_img.resize(original_size)
        side = 'right'
    elif os.path.exists(obj_left_path) and os.path.exists(aff_left_path):
        obj_img = Image.open(obj_left_path)
        aff_img = Image.open(aff_left_path)
        aff_img = aff_img.resize(original_size)
        side = 'left'
    
    if obj_img and aff_img:
        # Convert obj image to numpy array for bounding box calculation
        obj_np = np.array(obj_img)
        non_zero_coords = np.nonzero(obj_np)
        #print("Obj Shape: ", obj_np.shape)
        #print("Aff Shape: ", np.array(aff_img).shape)
        
        #print("Non Zero Coords Shape: ", non_zero_coords)
        min_y, max_y = min(non_zero_coords[0]), max(non_zero_coords[0])
        min_x, max_x = min(non_zero_coords[1]), max(non_zero_coords[1])
        #print(min_x)
        #print(min_y)
        #print(max_x)
        #print(max_y)

        bbox = (min_x, min_y, max_x, max_y)
        offset = 50

        # Load inpainted frame
        inpainted_img = Image.open(inpainted_frame_path)
        inpainted_img = inpainted_img.resize(original_size)

        print("Inpainted_Img Shape: ", np.array(inpainted_img).shape)
        print("Obj Shape: ", obj_np.shape)
        print("Aff Shape: ", np.array(aff_img).shape)

        # Crop, pad and resize the images
        obj_cropped = crop_and_pad(obj_img, bbox, offset, original_size)
        aff_cropped = crop_and_pad(aff_img, bbox, offset, original_size)
        inpainted_cropped = crop_and_pad(inpainted_img, bbox, offset, original_size)

        # Save the images
        obj_cropped.save(obj_right_path if side == 'right' else obj_left_path)
        aff_cropped.save(aff_right_path if side == 'right' else aff_left_path)
        inpainted_cropped.save(inpainted_frame_path)

# Function to process bimanual images
def process_bimanual(sequence_path, original_size):
    obj_left_path = os.path.join(sequence_path, 'obj_left.png')
    obj_right_path = os.path.join(sequence_path, 'obj_right.png')
    aff_left_path = os.path.join(sequence_path, 'aff_left.png')
    aff_right_path = os.path.join(sequence_path, 'aff_right.png')
    inpainted_frame_path = os.path.join(sequence_path, 'inpainted_frame.png')

    if os.path.exists(obj_left_path) and os.path.exists(obj_right_path):
        # Load left and right images
        obj_left_img = Image.open(obj_left_path)
        obj_right_img = Image.open(obj_right_path)

        # Convert both to numpy arrays to get bounding boxes
        obj_left_np = np.array(obj_left_img)
        obj_right_np = np.array(obj_right_img)

        left_non_zero = np.nonzero(obj_left_np)
        right_non_zero = np.nonzero(obj_right_np)

        min_y = min(min(left_non_zero[0]), min(right_non_zero[0]))
        max_y = max(max(left_non_zero[0]), max(right_non_zero[0]))
        min_x = min(min(left_non_zero[1]), min(right_non_zero[1]))
        max_x = max(max(left_non_zero[1]), max(right_non_zero[1]))

        bbox = (min_x, min_y, max_x, max_y)
        offset = 50

        # Load inpainted frame
        inpainted_img = Image.open(inpainted_frame_path)
        inpainted_img = inpainted_img.resize(original_size)
        aff_left_img = Image.open(aff_left_path)
        aff_left_img = aff_left_img.resize(original_size)
        aff_right_img = Image.open(aff_right_path)
        aff_right_img = aff_right_img.resize(original_size)

        # Crop, pad and resize all images
        obj_left_cropped = crop_and_pad(obj_left_img, bbox, offset, original_size)
        obj_right_cropped = crop_and_pad(obj_right_img, bbox, offset, original_size)
        aff_left_cropped = crop_and_pad(aff_left_img, bbox, offset, original_size)
        aff_right_cropped = crop_and_pad(aff_right_img, bbox, offset, original_size)
        inpainted_cropped = crop_and_pad(inpainted_img, bbox, offset, original_size)

        # Save the images
        obj_left_cropped.save(obj_left_path)
        obj_right_cropped.save(obj_right_path)
        aff_left_cropped.save(aff_left_path)
        aff_right_cropped.save(aff_right_path)
        inpainted_cropped.save(inpainted_frame_path)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process folders with cropped unimanual and bimanual sequences.")
    parser.add_argument('input_folder', type=str, help='Input folder path')
    args = parser.parse_args()
    sequences = []
    ambiguous_sequences = []
    input_folder = args.input_folder
    # Traverse the folder structure
    for sequence in os.listdir(input_folder):
        sequence_path = os.path.join(input_folder, sequence)
        annotation_path = os.path.join(sequence_path, 'annotation.json')

        if os.path.exists(annotation_path):
            # Load the annotation and check the taxonomy
            annotation_data = load_json(annotation_path)
            taxonomy_value = max(annotation_data["taxonomy"][0], annotation_data["taxonomy"][1])  # Get first index value
            if "something" in annotation_data["narration"] or "things" in annotation_data["narration"]:
                ambiguous_sequences.append((sequence_path, taxonomy_value)) 
            else:
                # Add to the sequences list for splitting later
                sequences.append((sequence_path, taxonomy_value))

    # Split the sequences into two subsets by taxonomy (0 or 1)
    unimanual_sequences = [s for s in sequences if s[1] == 1]
    bimanual_sequences = [s for s in sequences if s[1] == 0]

    # Randomly select 2/3 of the sequences for each subset
    cropped_unimanual = random.sample(unimanual_sequences, int(len(unimanual_sequences) * 2 / 3))
    cropped_bimanual = random.sample(bimanual_sequences, int(len(bimanual_sequences) * 2 / 3))

    # Process cropped_unimanual
    for sequence_path, _ in cropped_unimanual:
        # Load an image to get the original size
        obj_left_path = os.path.join(sequence_path, 'obj_left.png')
        obj_right_path = os.path.join(sequence_path, 'obj_right.png')
        if os.path.exists(obj_left_path):
            mask = Image.open(obj_left_path)
        elif os.path.exists(obj_right_path):
            mask = Image.open(obj_right_path)
        original_size = mask.size
        print("Cropping: ", sequence_path)
        process_unimanual(sequence_path, original_size)

    # Process cropped_bimanual
    for sequence_path, _ in cropped_bimanual:
        # Load an image to get the original size
        mask = Image.open(os.path.join(sequence_path, 'obj_left.png'))
        original_size = mask.size
        print("Cropping: ", sequence_path)
        process_bimanual(sequence_path, original_size)
    
    for sequence_path, taxonomy in ambiguous_sequences:
        if taxonomy == 1:
            obj_left_path = os.path.join(sequence_path, 'obj_left.png')
            obj_right_path = os.path.join(sequence_path, 'obj_right.png')
            if os.path.exists(obj_left_path):
                mask = Image.open(obj_left_path)
            elif os.path.exists(obj_right_path):
                mask = Image.open(obj_right_path)
            original_size = mask.size
            print("Cropping: ", sequence_path)
            process_unimanual(sequence_path, original_size)
        else:
            # Load an image to get the original size
            mask = Image.open(os.path.join(sequence_path, 'obj_left.png'))
            original_size = mask.size
            print("Cropping: ", sequence_path)
            process_bimanual(sequence_path, original_size)
    
    for sequence_path, _ in unimanual_sequences:
        obj_left_path = os.path.join(sequence_path, 'obj_left.png')
        obj_right_path = os.path.join(sequence_path, 'obj_right.png')
        aff_path = ''
        if os.path.exists(obj_left_path):
            mask = Image.open(obj_left_path)
            aff_path = os.path.join(sequence_path, 'aff_left.png')
            aff = Image.open(aff_path)
        elif os.path.exists(obj_right_path):
            mask = Image.open(obj_right_path)
            aff_path = os.path.join(sequence_path, 'aff_right.png')
            aff = Image.open(aff_path) 
        original_size = mask.size
        if aff_path != '':
            aff = aff.resize(original_size)
            aff.save(aff_path)
    
    for sequence_path, _ in bimanual_sequences:
        obj_left_path = os.path.join(sequence_path, 'obj_left.png')
        obj_left = Image.open(obj_left_path)
        original_size = obj_left.size
        aff_left_path = os.path.join(sequence_path, 'aff_left.png')
        aff_right_path = os.path.join(sequence_path, 'aff_right.png')
        aff_left = Image.open(aff_left_path)
        aff_right = Image.open(aff_right_path)
        aff_left = aff_left.resize(original_size)
        aff_right = aff_right.resize(original_size)
        aff_left.save(aff_left_path)
        aff_right.save(aff_right_path)

if __name__ == "__main__":
    main()
