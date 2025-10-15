import csv
import cv2
import h5py
import json
import numpy as np
import os
import shutil

from argparse import ArgumentParser

def check_threshold(img, limit):

    # RGB channels
    if len(img.shape) > 2:
        number_of_white = np.sum(img)/765

    # Grayscale
    else:
        number_of_white = np.sum(img)

    return number_of_white < limit and number_of_white > 0, number_of_white

def extract_verb_class_dict(verb_class_file):
    classes = []
    with open(verb_class_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            classes.append(row)
    return classes

def map_verb_to_class(verb, verb_classes):
    for verb_class in verb_classes:
        key = verb_class['key']
        instances = verb_class['instances']
        if verb in instances:
            return key
    print("Verb not found")
    return ''

def is_valid(folder, limit, categories, verb_classes):

    files = os.listdir(folder)

    # Check for completeness of files that should exist always
    if not "annotation.json" in files or not "inpainted_frame.png" in files:
        return False
    
    # Process the annotation.json file
    with open(os.path.join(folder, "annotation.json")) as json_file:
        data = json.load(json_file)
        taxonomy = data["taxonomy"]

        # Check if narration annotations are complete
        if data["noun"] == None or data["verb"] == None or data["narration"] == None:
            return False

        # Check if verb actually represents affordances
        invalid_verb_classes = ['eat', 'look', 'search', 'feel', 'transition', 'wait', 'smell', 'finish', 'unfreeze']
        verb_class = map_verb_to_class(data["verb"], verb_classes)
        if verb_class == '' or verb_class in invalid_verb_classes:
            return False
        
        # Check bimanual case
        if taxonomy[0] == 0:

            # Check for completeness
            if not "aff_left.png" in files or not "aff_right.png" in files or not "obj_left.png" in files or not "obj_right.png" in files:
                return False

            # Check if object is within the specified categories
            if not data["obj_right"] or not data["obj_left"]:
                return False
            if not data["obj_left"] in categories and not data["obj_right"] in categories and not 'all' in categories:
                return False
            
            # Check for empty or too large masks
            aff_left = cv2.imread(os.path.join(folder, "aff_left.png"))
            aff_right = cv2.imread(os.path.join(folder, "aff_right.png"))
            is_valid_left, number_of_white_left = check_threshold(aff_left, limit)
            is_valid_right, number_of_white_right = check_threshold(aff_right, limit)
            return is_valid_left and is_valid_right
        
        # Check unimanual case
        else:

            # Check for completeness
            if not (("aff_left.png" in files and "obj_left.png" in files) or("aff_right.png" in files and "obj_right.png" in files)):
                return False
            
            else:
                # Left Hand
                if "aff_left.png" in files and "obj_left.png" in files:

                    # Check if object is within specified categories
                    if not data["obj_left"]:
                        return False
                    if not data["obj_left"] in categories and not 'all' in categories:
                        return False
                    
                    # Check for empty or too large masks
                    aff_left = cv2.imread(os.path.join(folder, "aff_left.png"))
                    is_valid, number_of_white = check_threshold(aff_left, limit)
                    return is_valid
                
                # Right Hand
                else:

                    # Check if object is within specified categories
                    if not data["obj_right"]:
                        return False
                    if not data["obj_right"] in categories and not 'all' in categories:
                        return False

                    # Check for empty or too large masks
                    aff_right = cv2.imread(os.path.join(folder, "aff_right.png"))
                    is_valid, number_of_white = check_threshold(aff_right, limit)
                    return is_valid

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--dataset', 
                        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dataset/EPIC_DATA')), 'dataset_bimanual'))
    parser.add_argument('--copy', default=None)
    parser.add_argument('--categories', default='all')
    parser.add_argument('--limit', default=20000)
    parser.add_argument('--verb-class-file', 
                        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dataset/EPIC_DATA')), 'EPIC_100_verb_classes.csv'))

    args = parser.parse_args()
    vals = vars(args)
    verb_classes = extract_verb_class_dict(vals['verb_class_file'])
    folder = os.path.join(vals['dataset'], vals['name'])
    sequences = os.listdir(folder)
    print("Starting filtering process...")
    for sequence in sequences:
        sequence_path = os.path.join(folder, sequence)
        if is_valid(sequence_path, vals['limit'], vals['categories'], verb_classes) and vals['copy']:
            out_folder = os.path.join(vals['copy'], vals['name'])
            out = os.path.join(out_folder, sequence)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            shutil.copytree(sequence_path, out)
        elif not is_valid(sequence_path, vals['limit'], vals['categories'], verb_classes) and not vals['copy']:
            print(f"Deleting: {sequence_path}")
            shutil.rmtree(sequence_path)

                    



