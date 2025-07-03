import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob
import json
import csv
import pandas as pd
import os
from tqdm import tqdm
import json
import collections, functools, operator
from argparse import ArgumentParser
from PIL import Image
from scipy.stats import norm
from numpy import asarray

def rename_file(filename):
    parts = filename.split("_")
    
    # Search for the split string with at least three "0"
    for i, part in enumerate(parts):
        if part.count("0") >= 3:

            # Remove three "0" from the string
            new_part = part.replace("0", "", 3)

            # Merge parts back into filename
            new_filename = new_part
            return new_filename


def extract_bimanual_information(image_path, image_name, entities, output_directory):
    is_bimanual = False
    is_left = False
    is_right = False
    is_symmetric = False
    left_obj = None
    right_obj = None
    contact_obj_left = None
    contact_obj_right = None
    for entity in entities:
        if entity['name'] == 'left hand' and 'in_contact_object' in entity.keys():
            if entity['in_contact_object'] != 'inconclusive' and entity['in_contact_object'] != 'hand-not-in-contact' and entity['in_contact_object'] != 'none-of-the-above':
                is_left = True
                contact_obj_left = entity['in_contact_object']
        if entity['name'] == 'right hand' and 'in_contact_object' in entity.keys():
            if entity['in_contact_object'] != 'inconclusive' and entity['in_contact_object'] != 'hand-not-in-contact' and entity['in_contact_object'] != 'none-of-the-above':
                is_right = True
                contact_obj_right = entity['in_contact_object']
        if is_left and is_right:
            is_bimanual = True
            if contact_obj_right == contact_obj_left:
                is_symmetric = True
    
    if not is_left and not is_right:
        return

    for entity in entities:
        if entity['id'] == contact_obj_left:
            left_obj = entity['name']
        if entity['id'] == contact_obj_right:
            right_obj = entity['name']
    
    json_path = os.path.join(output_directory, image_path.split('/')[0])
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    filename = image_name.split('.')[0]
    filename = rename_file(filename)
    json_filename = os.path.join(json_path, filename + '.json')
    taxonomy = None
    if is_symmetric:
        taxonomy = [0, 1, 0]
    elif not is_symmetric and is_bimanual:
        taxonomy = [0, 0, 1]
    else:
        taxonomy = [1, 0, 0]
    annotations = {
        "taxonomy": taxonomy,
        "obj_left": left_obj,
        "obj_right": right_obj,
        "narration": None,
        "noun": None,
        "verb": None,
        "vector": None
    }
    with open(json_filename, 'w') as file:
        json.dump(annotations, file)
        file.close()          

def folder_of_jsons_to_annotation(json_files_path, output_directory):
        
    for json_file in tqdm(sorted(glob.glob(os.path.join(json_files_path ,'*.json')))):
        json_to_annotation(json_file, output_directory)

def json_to_annotation(filename, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    #print(f'unique object count of {filename.split("/")[-1]} is {max_count}')
    
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data["video_annotations"], key=lambda k: k['image']['image_path'])
    # Iterating through the json list
    for datapoint in data:
        image_name = datapoint["image"]["name"]
        image_path = datapoint["image"]["image_path"]
        masks_info = datapoint["annotations"]
        extract_bimanual_information(image_path, image_name, masks_info, output_directory) #this is for unique id for each object throughout the video

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json_directory', default=None)
    parser.add_argument('--out', default=None)

    args = parser.parse_args()
    vals = vars(args)

    if vals["json_directory"] != None and vals["out"] != None:
        json_directory = vals["json_directory"]
        out = vals["out"]
        folder_of_jsons_to_annotation(json_directory, out)
