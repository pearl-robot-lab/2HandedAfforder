import os 
import json
from argparse import ArgumentParser

def analyze_dataset(dirs):
    folders = os.listdir(dirs)
    labels = {}
    for folder in folders:
        annotation_dir = os.path.join(dirs, folder, "annotation.json")
        f = open(annotation_dir)
        annotation = json.load(f)
        if annotation["taxonomy"][0] == 1:
            if annotation["obj_left"] != None:
                if annotation["obj_left"] in labels.keys():
                    labels[annotation["obj_left"]] = labels[annotation["obj_left"]] + 1
                else:
                    labels[annotation["obj_left"]] = 1
            elif annotation["obj_right"] != None:
                if annotation["obj_right"] in labels.keys():
                    labels[annotation["obj_right"]] = labels[annotation["obj_right"]] + 1
                else:
                    labels[annotation["obj_right"]] = 1
    print(labels)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dirs", default=None)
    args = parser.parse_args()
    vals = vars(args)
    if vals["dirs"] != None:
        dirs = vals["dirs"]
        analyze_dataset(dirs)
    else:
        print("wrong argument")
