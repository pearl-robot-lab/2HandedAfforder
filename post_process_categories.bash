#!/bin/sh

cd scripts
python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$1/hand/left --ref-folder ../EPIC_DATA/bimanual_annotations_json/$1 --out ../EPIC_DATA/sequences/$1/left
python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$1/hand/right --ref-folder ../EPIC_DATA/bimanual_annotations_json/$1 --out ../EPIC_DATA/sequences/$1/right
#python helper.py --crop_masks_dir ../EPIC_DATA/xmem_masks/$1/object/left --crop_masks_out ../EPIC_DATA/cropped_masks_unmodified/$1/object/left
#python helper.py --crop_masks_dir ../EPIC_DATA/xmem_masks/$1/object/right --crop_masks_out ../EPIC_DATA/cropped_masks_unmodified/$1/object/right
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$1/object/left --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified/$1/object/left
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$1/object/right --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified/$1/object/right

# MI-GAN
#python setup_dataset.py --video-id $1 --out ../EPIC_DATA/dataset_migan/$1 --obj-name padded_masks_unmodified --aff-name padded_masks --inp-name inpainted_hands_migan

# Agent-Inpainting
python setup_dataset.py --video-id $1 --out ../EPIC_DATA/dataset/$1 --obj-name padded_masks_unmodified --aff-name padded_masks

python trajectory_tracking.py --folder_left ../EPIC_DATA/sequences/$1/left --folder_right ../EPIC_DATA/sequences/$1/right --json_directory ../EPIC_DATA/bimanual_annotations_json/$1 --dataset_directory ../EPIC_DATA/dataset/$1
rm -r ../EPIC_DATA/sequences/$1
rm -r ../EPIC_DATA/cropped_masks_unmodified/$1

#python create_dataset.py --dir ../EPIC_DATA/dataset/$1 --out ../EPIC_DATA/hdf5/ --name $1

# MI-GAN
#python create_dataset_categories.py --dir ../EPIC_DATA/dataset_migan/$1 --out ../EPIC_DATA/hdf5_bowl_migan_categories --name $1 --categories bowl spoon milk milk_carton cereal_box knife plate

# Agent-Inpainting
#python create_dataset_categories.py --dir ../EPIC_DATA/dataset/$1 --out ../EPIC_DATA/hdf5_test --name $1 --categories bowl spoon milk milk_carton cereal_box knife plate
python create_dataset_categories.py --dir ../EPIC_DATA/dataset/$1 --out ../EPIC_DATA/hdf5_test --name $1 --categories all
#python create_dataset_categories.py --dir ../EPIC_DATA/dataset/$1 --out ../EPIC_DATA/hdf5_bowl_test --name $1 --categories bowl

