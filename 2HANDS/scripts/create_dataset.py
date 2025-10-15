import csv
import cv2
import h5py
import json
import numpy as np
import os

from argparse import ArgumentParser
from utils.compress_masks_to_json import convert_masks_to_json

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
    if not "annotation.json" in files:
        return False
    if not "inpainted_frame.png" in files:
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
            print('found invalid verb_class: ', verb_class)
            return False
        
        # Check bimanual case
        if taxonomy[0] == 0:

            # Check for completeness
            if not "aff_left.png" in files or not "aff_right.png" in files or not "obj_left.png" in files or not "obj_right.png" in files:
                return False

            # Check if object is within the specified categories
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
                    if not data["obj_left"] in categories and not 'all' in categories:
                        return False
                    
                    # Check for empty or too large masks
                    aff_left = cv2.imread(os.path.join(folder, "aff_left.png"))
                    is_valid, number_of_white = check_threshold(aff_left, limit)
                    return is_valid
                
                # Right Hand
                else:

                    # Check if object is within specified categories
                    if not data["obj_right"] in categories and not 'all' in categories:
                        return False

                    # Check for empty or too large masks
                    aff_right = cv2.imread(os.path.join(folder, "aff_right.png"))
                    is_valid, number_of_white = check_threshold(aff_right, limit)
                    return is_valid

def check_threshold(img, limit):

    # RGB channels
    if len(img.shape) > 2:
        number_of_white = np.sum(img)/765

    # Grayscale
    else:
        number_of_white = np.sum(img)

    return number_of_white < limit and number_of_white > 20, number_of_white

def build_dataset(dir, out, name, limit, categories, verb_class_file):
    aff_left = []
    aff_right = []
    inpainted = []
    obj_mask_left = []
    obj_mask_right = []
    obj_id_left = []
    obj_id_right = []
    noun = []
    verb = []
    narration = []
    taxonomy = []
    hand_vector = []
    bbox_left = []
    bbox_right = []
    valid_counter = 0
    invalid_counter = 0
    if not os.path.exists(os.path.join(out, 'h5')):
        os.makedirs(os.path.join(out, 'h5'))
    if not os.path.exists(os.path.join(out, 'jsons')):
        os.makedirs(os.path.join(out, 'jsons'))
    folders = os.listdir(dir)
    verb_classes = extract_verb_class_dict(verb_class_file)
    for folder in folders:
        folder_path = os.path.join(dir, folder)
        if is_valid(folder_path, limit, categories, verb_classes) or True:
            valid_counter += 1
            files = os.listdir(folder_path)
            f = open(os.path.join(folder_path, "annotation.json"))
            annotation = json.load(f)
            taxonomy.append(annotation["taxonomy"])
            narration.append(annotation["narration"])
            noun.append(annotation["noun"])
            verb.append(annotation["verb"])
            if annotation["vector"] != None:
                hand_vector.append(annotation["vector"])
            else:
                hand_vector.append(np.zeros((2, 2)))
            if annotation["obj_left"] != None:
                obj_id_left.append(annotation["obj_left"])
            else:
                obj_id_left.append("")
            if annotation["obj_right"] != None:
                obj_id_right.append(annotation["obj_right"])
            else:
                obj_id_right.append("")
            if "aff_right.png" in files:
                aff_mask = cv2.imread(os.path.join(folder_path, "aff_right.png"), cv2.IMREAD_GRAYSCALE)
                obj_mask = cv2.imread(os.path.join(folder_path, "obj_right.png"), cv2.IMREAD_GRAYSCALE)
                min_x_right = min(np.where(obj_mask != 0)[0])
                max_x_right = max(np.where(obj_mask != 0)[0])
                min_y_right = min(np.where(obj_mask != 0)[1])
                max_y_right = max(np.where(obj_mask != 0)[1])
                bbox = [[min_x_right, min_y_right], [max_x_right, max_y_right]]
                bbox_right.append(bbox)
                obj_mask_right.append(obj_mask)
                aff_right.append(aff_mask)
                if not "aff_left.png" in files:
                    aff_left.append(np.zeros(aff_mask.shape))
                    bbox_left.append(np.zeros((2,2)))
                    obj_mask_left.append(np.zeros(obj_mask.shape))
            if "aff_left.png" in files:
                aff_mask = cv2.imread(os.path.join(folder_path, "aff_left.png"), cv2.IMREAD_GRAYSCALE)
                obj_mask = cv2.imread(os.path.join(folder_path, "obj_left.png"), cv2.IMREAD_GRAYSCALE)
                min_x_left = min(np.where(obj_mask != 0)[0])
                max_x_left = max(np.where(obj_mask != 0)[0])
                min_y_left = min(np.where(obj_mask != 0)[1])
                max_y_left = max(np.where(obj_mask != 0)[1])
                bbox = [[min_x_left, min_y_left], [max_x_left, max_y_left]]
                bbox_left.append(bbox)
                obj_mask_left.append(obj_mask)
                aff_left.append(aff_mask)
                if not "aff_right.png" in files:
                    aff_right.append(np.zeros(aff_mask.shape))
                    bbox_right.append(np.zeros((2,2)))
                    obj_mask_right.append(np.zeros(obj_mask.shape))
            inpainted_img = cv2.imread(os.path.join(folder_path, "inpainted_frame.png"))
            inpainted.append(inpainted_img)
        else:
            invalid_counter += 1

    res_dict = {
        'inpainted' : np.array(inpainted, dtype=np.uint8),
        'obj_id_left' : obj_id_left,
        'obj_id_right' : obj_id_right,
        'noun' : noun,
        'verb' : verb,
        'narration' : narration,
        'taxonomy' : np.array(taxonomy, dtype=np.uint8)
    }


    import pdb; pdb.set_trace()
    convert_masks_to_json(np.array(aff_left, dtype=np.uint8), np.array(aff_right, dtype=np.uint8), np.array(obj_mask_left, dtype=np.uint8), np.array(obj_mask_right, dtype=np.uint8), os.path.join(out, 'jsons', name + ".json"))
    hdf5 = h5py.File(os.path.join(out, 'h5', name + ".h5"), 'w')
    grp = hdf5.create_group('data')
    for curr_key in res_dict.keys():
        dset = grp.create_dataset(curr_key, data=res_dict[curr_key])
    hdf5.close()
    print("Valid Frames Total: ", valid_counter)
    print("Invalid Frames Total: ", invalid_counter)
    print(str(round(valid_counter/(invalid_counter + valid_counter) * 100, 2)) + "% were valid frames")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--limit', default=30000, type=float)
    parser.add_argument('--categories', default=[], nargs='+')
    parser.add_argument('--verb-class-file', 
                        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/EPIC_DATA')), 'EPIC_100_verb_classes.csv'))

    args = parser.parse_args()
    vals = vars(args)

    if vals['dir'] != None and vals['out'] != None and vals['name'] != None and vals['categories'] != None:
        dir = vals['dir']
        out = vals['out']
        name = vals['name']
        limit = vals['limit']
        categories = vals['categories']
        verb_class_file = vals['verb_class_file']
        build_dataset(dir, out, name, limit, categories, verb_class_file)

