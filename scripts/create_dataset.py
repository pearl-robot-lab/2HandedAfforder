import cv2
import h5py
import json
import numpy as np
import os

from argparse import ArgumentParser

def is_valid(folder, limit_low, limit_up):
    files = os.listdir(folder)
    if not "annotation.json" in files:
        print("no annotation for frame: ", folder)
        return False
    if not "inpainted_frame.png" in files:
        print("no inpainted image for frame: ", folder)
        return False
    with open(os.path.join(folder, "annotation.json")) as json_file:
        data = json.load(json_file)
        taxonomy = data["taxonomy"]
        if data["noun"] == None or data["verb"] == None or data["narration"] == None:
            print("annotation was not complete for frame: ", folder)
            return False
        if taxonomy[0] == 0:
            if not "aff_left.png" in files or not "aff_right.png" in files or not "obj_left.png" in files or not "obj_right.png" in files:
                print("bimanual taxonomy but only the masks for one hand are provided for frame: ", folder)
                return False
            aff_left = cv2.imread(os.path.join(folder, "aff_left.png"))
            aff_right = cv2.imread(os.path.join(folder, "aff_right.png"))
            obj_left = cv2.imread(os.path.join(folder, "obj_left.png"))
            obj_right = cv2.imread(os.path.join(folder, "obj_right.png"))
            is_valid_left, ratio_left = check_threshold(aff_left, obj_left, limit_low, limit_up)
            is_valid_right, ratio_right = check_threshold(aff_right, obj_right, limit_low, limit_up)
            if is_valid_left and is_valid_right :
                print("valid frame: ", folder)
                return True
            else:
                #print("skipping the following frame because of too many/few white pixels: ", folder)
                #print("ratio left: ", ratio_left)
                #print("ratio right: ", ratio_right)
                print("invalid frame due to masks being empty: ", folder)
                return False
        else:
            if not (("aff_left.png" in files and "obj_left.png" in files) or("aff_right.png" in files and "obj_right.png" in files)):
                print("no hand masks provided for frame: ", folder)
                return False
            else:
                if "aff_left.png" in files and "obj_left.png" in files:
                    aff_left = cv2.imread(os.path.join(folder, "aff_left.png"))
                    obj_left = cv2.imread(os.path.join(folder, "obj_left.png"))
                    is_valid, ratio = check_threshold(aff_left, obj_left, limit_low, limit_up)
                    if is_valid:
                        print("valid frame: ", folder)
                        return True
                    else:
                        print("left mask is empty for frame: ", folder)
                        return False
                else:
                    aff_right = cv2.imread(os.path.join(folder, "aff_right.png"))
                    obj_right = cv2.imread(os.path.join(folder, "obj_right.png"))
                    is_valid, ratio = check_threshold(aff_right, obj_right, limit_low, limit_up)
                    if is_valid:
                        print("valid frame: ", folder)
                        return True
                    else:
                        print("right mask is empty for frame: ", folder)
                        return False

def check_threshold(img, ref_img, limit_low, limit_up):
    if len(img.shape) > 1:
        number_of_white_img = len(np.where(img != 0)[0])
    else:
        number_of_white_img = len(np.where(img != 0))
    if len(ref_img.shape) > 1:
        number_of_white_ref = len(np.where(ref_img != 0)[0])
    else:
        number_of_white_ref = len(np.where(ref_img != 0))
    if number_of_white_ref == 0:
        return False, -1
    ratio = number_of_white_img / number_of_white_ref
    if ratio < limit_low or ratio > limit_up:
        return False, ratio
    else:
        return True, ratio

def build_dataset(dir, out, name, limit_low, limit_up):
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
    if not os.path.exists(out):
        os.makedirs(out)
    folders = os.listdir(dir)
    for folder in folders:
        folder_path = os.path.join(dir, folder)
        if is_valid(folder_path, limit_low, limit_up):
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
            #print("CHECK FRAME: ", folder_path)
            #obj_id_left.append(annotation["obj_left"])
            #obj_id_right.append(annotation["obj_right"])
            if "aff_right.png" in files:
                aff_mask = cv2.imread(os.path.join(folder_path, "aff_right.png"))
                obj_mask = cv2.imread(os.path.join(folder_path, "obj_right.png"))
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
                aff_mask = cv2.imread(os.path.join(folder_path, "aff_left.png"))
                obj_mask = cv2.imread(os.path.join(folder_path, "obj_left.png"))
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
        'aff_left' : np.array(aff_left, dtype=np.uint8),
        'aff_right' : np.array(aff_right, dtype=np.uint8),
        'obj_mask_left' : np.array(obj_mask_left, dtype=np.uint8),
        'obj_mask_right' : np.array(obj_mask_right, dtype=np.uint8),
        'obj_id_left' : obj_id_left,
        'obj_id_right' : obj_id_right,
        'noun' : noun,
        'verb' : verb,
        'narration' : narration,
        'taxonomy' : np.array(taxonomy, dtype=np.uint8),
        'hand_vector' : np.array(hand_vector, dtype=np.int8),
        'bounding_box_left' : np.array(bbox_left, dtype=np.uint8),
        'bounding_box_right' : np.array(bbox_right, dtype=np.uint8)
    }

    hdf5 = h5py.File(os.path.join(out, name + ".h5"), 'w')
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
    parser.add_argument('--limit-low', default=1/9999999, type=float)
    parser.add_argument('--limit-up', default=1.0, type=float)

    args = parser.parse_args()
    vals = vars(args)

    if vals['dir'] != None and vals['out'] != None and vals['name'] != None:
        dir = vals['dir']
        out = vals['out']
        name = vals['name']
        limit_low = vals['limit_low']
        limit_up = vals['limit_up']
        build_dataset(dir, out, name, limit_low, limit_up)

