import cv2
import numpy as np
import os

from argparse import ArgumentParser
from PIL import Image

def perform_color_correction(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[...,::-1]
    Image.fromarray(img).save(img_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dir', default=None)
    args = parser.parse_args()
    vals = vars(args)

    if vals['dir'] != None:
        video = vals['dir']
        sequences = os.listdir(video)
        for sequence in sequences:
            sequence_path = os.path.join(video, sequence)
            img_path = os.path.join(sequence_path, 'inpainted_frame.png')
            perform_color_correction(img_path)