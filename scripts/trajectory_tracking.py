import cv2
import json
import numpy as np
import os
import shutil

from argparse import ArgumentParser
from PIL import Image

def extract_ref_point(img):
    # Get all the indices of points in the mask that are white
    white_indices = np.where(img != 0)
    # print("White indices during extraction: ", white_indices)

    # Get average point of indices
    # print(len(white_indices[0]))
    av_point_x = int(sum(white_indices[0])/len(white_indices[0]))
    av_point_y = int(sum(white_indices[1])/len(white_indices[0]))
    av_point = np.array([av_point_x, av_point_y])

    return av_point

def get_vector(m1, m2):
    av_point_m1 = extract_ref_point(m1)
    av_point_m2 = extract_ref_point(m2)
    #print("AV POINT M1: ", av_point_m1 )
    #print("AV POINT M2: ", av_point_m2 )
    return av_point_m2 - av_point_m1, av_point_m1

def get_vector_diff(v1, v2):
    if not np.any(v1) or not np.any(v2):
        return 0
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def is_symmetric(folder_left, folder_right, dataset_directory, threshold=0.95):
    files_left = sorted(os.listdir(folder_left))
    files_right = sorted(os.listdir(folder_right))
    symmetric_counter = 0
    len_counter = 0
    last_mask_left = None
    last_mask_right = None
    is_first_mask = True
    points_left = []
    points_right = []
    datas = []
    json_paths = []
    for file in files_left:
        if not file in files_right:
            continue
        print(file)
        mask_left_path = os.path.join(folder_left, file)
        mask_right_path = os.path.join(folder_right, file)
        mask_left = cv2.imread(mask_left_path)
        mask_right = cv2.imread(mask_right_path)
        # print("Mask Left White Indices Before: ", np.where(mask_left != 0))
        if len(np.where(mask_left != 0)[0]) == 0 or len(np.where(mask_right != 0)[0]) == 0:
            continue
        if is_first_mask:
            last_mask_left = mask_left
            last_mask_right = mask_right
            is_first_mask = False
            continue
        vector_left, av_point_left = get_vector(last_mask_left, mask_left)
        vector_right, av_point_right = get_vector(last_mask_right, mask_right)
        dst = os.path.join(dataset_directory, file.split(".")[0], "annotation.json")
        if os.path.exists(dst):
            json_paths.append(dst)
            f = open(dst, 'r')
            data = json.load(f)
            f.close()
            data["vector"] = [vector_left.tolist(), vector_right.tolist()]
            datas.append(data)

        

        points_left.append(av_point_left)
        points_right.append(av_point_right)
        len_vec_left = np.linalg.norm(vector_left)
        len_vec_right = np.linalg.norm(vector_right)
        len_counter += (len_vec_left + len_vec_right)/2
        last_mask_left = mask_left
        last_mask_right = mask_right
        if get_vector_diff(vector_left, vector_right) >= threshold:
            symmetric_counter += (len_vec_left + len_vec_right)/2
    #visualize_trajectory(points_left, points_right, last_mask_left.shape, last_mask_right.shape, folder_left)
    if symmetric_counter >= int(len_counter * 0.75):
        for i, json_path in enumerate(json_paths):
            f = open(json_path, "w")
            json.dump(datas[i], f)
            f.close()
        return True
    else:
        for i, json_path in enumerate(json_paths):
            datas[i]["taxonomy"] = [0, 0, 1]
            f = open(json_path, "w")
            json.dump(datas[i], f)
            f.close()
            

def assign_bimanual_taxonomy(folder_left, folder_right, ref_folder, dataset_directory, threshold=0.95):
    folders_left = os.listdir(folder_left)
    folders_right = os.listdir(folder_right)
    refs = os.listdir(ref_folder)
    for ref in refs:
        file_path = os.path.join(ref_folder, ref)
        ref_name = ref.split('.')[0]
        if not (ref_name in folders_left and ref_name in folders_right):
            print("Unimanual sequences: ", ref_name)
            #if not os.path.exists(out):
            #    os.makedirs(out)
            #new_file = os.path.join(out, ref)
            #shutil.copy(file_path, new_file)
            continue
        new_f_left = os.path.join(folder_left, ref_name)
        new_f_right = os.path.join(folder_right, ref_name)
        # the file is in json format. Open the file as dictionary
        with open(file_path, 'r') as f:
            data = json.load(f)
            if data["taxonomy"][1] == 1:
                if not is_symmetric(new_f_left, new_f_right, dataset_directory, threshold):
                    #data["taxonomy"] = [0, 0, 1]
                    print("Change symmetric to asymmetric for sequence: ", ref_name)
                else:
                    print("Unmodified symmetric sequence: ", ref_name)
                #if not os.path.exists(out):
                #    os.makedirs(out)
                #new_file = os.path.join(out, ref)
                #w = open(new_file, "x")
                #json.dump(data, w)
                #w.close()
        #with open(file_path, 'w') as f:
        #    json.dump(data, f)

def visualize_trajectory(points_left, points_right, shp_left, shp_right, name):
    img = np.zeros(shp_left)
    color_palette = [(i * 12, 0, 255) for i in range(20)]
    last_point_left = points_left[0]
    last_point_right = points_right[0]
    for i, point in enumerate(points_left[1:]):
        img = cv2.line(img, (last_point_left[1], last_point_left[0]), (point[1], point[0]), color=color_palette[i]) 
        img = cv2.line(img, (last_point_right[1], last_point_right[0]), (points_right[i][1], points_right[i][0]), color=color_palette[i]) 
        last_point_left = point
        last_point_right = points_right[i]
    if not os.path.exists("trajectories"):
        os.makedirs("trajectories")
    cv2.imwrite(os.path.join("trajectories", name.split('\\')[-1] + ".jpg"), img)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder_left', default=None)
    parser.add_argument('--folder_right', default=None)
    parser.add_argument('--json_directory', default=None)
    parser.add_argument('--dataset_directory', default=None)
    args = parser.parse_args()
    vals = vars(args)

    if vals["folder_left"] != None and vals["folder_right"] != None and vals["json_directory"] != None and vals["dataset_directory"] != None:
        folder_left = vals["folder_left"]
        folder_right = vals["folder_right"]
        json_directory = vals["json_directory"]
        dataset_directory = vals["dataset_directory"]
        assign_bimanual_taxonomy(folder_left, folder_right, json_directory, dataset_directory)


    
