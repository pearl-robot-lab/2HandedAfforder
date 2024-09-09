import shutil
import os
import numpy as np
from argparse import ArgumentParser

def list_files(file_number, n, seq):
	files = []
	for i in range(n):
		new_file_number = int(file_number + i * (seq / n) - (seq / n))
		print(new_file_number)
		new_file = str(new_file_number).zfill(7) + ".png"
		#print(new_file)
		files.append(new_file)
	return files

def restructure_files(f_in, out,  n=4, seq=200):
	if not os.path.exists(out):
		os.makedirs(out)
	files = os.listdir(f_in)
	files.sort()
	counter = 0
	first_file_number = int(files[0].split('.')[0])
	for file in files:
		file_number = int(file.split('.')[0])
		file_list = list_files(file_number, n, seq)
		if (file_number - first_file_number) > seq:
			first_file_number = file_number
			counter = 0 
		if counter == int(seq/n):
			continue
		else:
			for img in file_list:
				if not os.path.exists(os.path.join(out, str(first_file_number) + "_"  + str(counter), "masks")):
					os.makedirs(os.path.join(out, str(first_file_number) + "_" + str(counter), "masks"))
				shutil.copy(os.path.join(f_in,  img), os.path.join(out, str(first_file_number) +  "_" + str(counter), "masks",  img))
		counter  += 1

def copy_frames(f_in, out):
	frames = os.listdir(f_in)
	mask_folders = os.listdir(out)
	for folder in mask_folders:
		if os.path.exists(os.path.join(out, folder, "mask")):
			print("renaming folder: ", folder)
			os.rename(os.path.join(out, folder, "mask"), os.path.join(out, folder, "masks"))
		files = os.listdir(os.path.join(out, folder, "masks"))
		out_folder = os.path.join(out, folder, "raw")
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)
		for file in files:
			img_file = file.split(".")[0] + ".jpg"
			shutil.copy(os.path.join(f_in, img_file), os.path.join(out_folder, img_file)) 

def copy_mask_with_reference_folder(f_in, ref, out, n=4, seq=200):
	if not os.path.exists(out):
		os.makedirs(out)
	files = os.listdir(f_in)
	target_files = os.listdir(ref)
	target_files = [str(int(file.split('_')[-1].split('.')[0])).zfill(7) + ".png" for file in target_files]
	files.sort()
	target_files.sort()
	counter = 0
	first_file_number = int(files[0].split('.')[0])
	largest_file = int(files[-1].split('.')[0])
	print(largest_file)
	for file in target_files[1:len(target_files)-1]:
		counter = 0
		for i in range(20):
			file_number = int(file.split('.')[0]) + i
			file_list = list_files(file_number, n, seq)
			print(file_number + seq / n)
			if (file_number + 2 * seq / n) > largest_file:
				return
			if (file_number - first_file_number) > seq:
				first_file_number = file_number
				counter = 0
			if counter == int(seq/n):
				continue
			else:
				valid = True
				for f in file_list:
					if not os.path.exists(os.path.join(f_in, f)):
						valid = False
				if valid:		
					for img in file_list:
						if not os.path.exists(os.path.join(out, str(first_file_number) + "_"  + str(counter), "masks")):
							os.makedirs(os.path.join(out, str(first_file_number) + "_" + str(counter), "masks"))
						shutil.copy(os.path.join(f_in,  img), os.path.join(out, str(first_file_number) +  "_" + str(counter), "masks",  img))
			counter  += 1

#f_in = "EPIC_DATA/xmem_masks/P05_01/hand/both"
#out = "../inpainting/agent_inpainting/data_gen/P05_01"
#restructure_files(f_in, out)

parser = ArgumentParser()
parser.add_argument('--f_in')
parser.add_argument('--out')
parser.add_argument('--ismask')
parser.add_argument('--ref')
#f_in = "EPIC_DATA/frames/P05/P05_01"
#out = "../inpainting/agent_inpainting/data_gen/P05_01"

args = parser.parse_args()
args = vars(args)
if args['ismask'] == 'True' and args['ref'] == None:
	restructure_files(args['f_in'], args['out'])
elif args['ismask'] == 'True' and args['ref'] != None:
	copy_mask_with_reference_folder(args['f_in'],  args['ref'], args['out'])		
elif args['ismask'] == 'False':
	copy_frames(args['f_in'], args['out'])
