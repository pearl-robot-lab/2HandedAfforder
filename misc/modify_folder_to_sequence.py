import os
import shutil

from argparse import ArgumentParser

def modify_folder_to_sequence(dir, ref_folder, out):
    if not os.path.exists(out):
        os.makedirs(out)
    ref_files = os.listdir(ref_folder)
    files = os.listdir(dir)
    for ref_file in ref_files:
        ref_file_number = int(ref_file.split('.')[0])
        for i in range(ref_file_number-9, ref_file_number+12):
            file = str(i).zfill(7) + ".png"
            if file in files:
                old_path = os.path.join(dir, file)
                new_folder = os.path.join(out, ref_file.split('.')[0])
                new_path = os.path.join(new_folder, file)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                shutil.copy(old_path, new_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", default=None, type=str, help="Directory containing the images")
    parser.add_argument("--ref-folder", default=None, type=str, help="Reference folder containing sequence numbers")
    parser.add_argument("--out", default=None, type=str, help="Output folder")
    args = parser.parse_args()
    vals = vars(args)
    if vals["dir"] != None and vals["ref_folder"] != None and vals["out"] != None:
        modify_folder_to_sequence(args.dir, args.ref_folder, args.out)
