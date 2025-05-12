import os
import json
from PIL import Image
import shutil
from argparse import ArgumentParser

def create_flipped_copy(main_folder_path):
    print(os.path.dirname(main_folder_path))
    new_folder_name = 'flipped_' + os.path.basename(main_folder_path)
    print(f"New Folder Name: {new_folder_name}")
    new_folder_path = os.path.join(os.path.dirname(main_folder_path), new_folder_name)
    shutil.copytree(main_folder_path, new_folder_path)
    sequences = os.listdir(new_folder_path)
    for sequence in sequences:
        sequence_path = os.path.join(new_folder_path, sequence)
        files = os.listdir(sequence_path)
        if 'annotation.json' in files:
            print(f"Flipping folder: {sequence_path}")
            # Check for existence and flip images in the new folder
            left_img_path = os.path.join(sequence_path, "obj_left.png")
            right_img_path = os.path.join(sequence_path, "obj_right.png")
            aff_left_path = os.path.join(sequence_path, "aff_left.png")
            aff_right_path = os.path.join(sequence_path, "aff_right.png")
            inpainted_path = os.path.join(sequence_path, "inpainted_frame.png")
            is_left = False
            is_right = False
            if os.path.exists(left_img_path):
                is_left = True
                aff_left = Image.open(aff_left_path)
                left_img = Image.open(left_img_path)
                os.remove(aff_left_path)
                os.remove(left_img_path)
            if os.path.exists(right_img_path):
                is_right = True
                right_img = Image.open(right_img_path)
                aff_right = Image.open(aff_right_path)
                os.remove(aff_right_path)
                os.remove(right_img_path)
            if is_left:
                left_img_flipped = left_img.transpose(Image.FLIP_LEFT_RIGHT)
                aff_left_flipped = aff_left.transpose(Image.FLIP_LEFT_RIGHT)
                left_img_flipped.save(right_img_path)
                aff_left_flipped.save(aff_right_path)
            if is_right:
                right_img_flipped = right_img.transpose(Image.FLIP_LEFT_RIGHT)
                aff_right_flipped = aff_right.transpose(Image.FLIP_LEFT_RIGHT)
                right_img_flipped.save(left_img_path)
                aff_right_flipped.save(aff_left_path)

            inpainted = Image.open(inpainted_path)
            inpainted = inpainted.transpose(Image.FLIP_LEFT_RIGHT)
            inpainted.save(inpainted_path)

            # Load and modify annotations.json
            annotations_path = os.path.join(sequence_path, 'annotation.json')
            with open(annotations_path, 'r') as json_file:
                data = json.load(json_file)
            
            # Flip taxonomy array if it has at least 2 dimensions
            if 'taxonomy' in data and len(data['taxonomy']) >= 2:
                taxonomy = data['taxonomy']
                taxonomy[0], taxonomy[1] = taxonomy[1], taxonomy[0]  # Swap the first two indices
                data['taxonomy'] = taxonomy

            # Swap contents of obj_left and obj_right fields if they exist
            if 'obj_left' in data and 'obj_right' in data:
                data['obj_left'], data['obj_right'] = data['obj_right'], data['obj_left']

            # Save modified annotations.json
            with open(annotations_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

# Run the function with the path to your main folder
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', default=None)
    args = parser.parse_args()
    vals = vars(args)
    if vals['dataset']:
        create_flipped_copy(vals['dataset'])
