import csv
import os
import glob
import json
import os
from tqdm import tqdm
import json
from argparse import ArgumentParser

def add_narrations(json_folders_path, narration_file):
    
    json_folders = os.listdir(json_folders_path)
    with open(narration_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row['video_id']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']
            narration = row['narration']
            verb = row['verb']
            noun = row['noun']
            for json_folder in json_folders:
                if video_id != json_folder:
                    continue
                json_files_path = os.path.join(json_folders_path, json_folder)
                json_files = os.listdir(json_files_path)
                for json_file in json_files:
                    json_file_path = os.path.join(json_files_path, json_file)
                    if int(json_file.split('.')[0]) in range(int(start_frame), int(stop_frame) + 1):
                        f = open(json_file_path, "r")
                        data = json.load(f)
                        data["narration"] = narration
                        data["verb"] = verb
                        data["noun"] = noun
                        f.close()
                        f = open(json_file_path, "w")
                        json.dump(data, f)
                        f.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json_directory', default=None)
    parser.add_argument('--narration_file', default=None)

    args = parser.parse_args()
    vals = vars(args)

    if vals["json_directory"] != None and vals["narration_file"] != None:
        json_directory = vals["json_directory"]
        narration_file = vals["narration_file"]
        add_narrations(json_directory, narration_file)
