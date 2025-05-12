#!/bin/sh

# Download and extract the EPIC-KITCHENS video if it does not already exist (For different videos there exist different directories online)
if [ ! -d ../dataset/EPIC_DATA/frames/$1/$2 ]; then
cd ../dataset/EPIC_DATA/frames
mkdir $1
cd $1
mkdir $2
cd ..
wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$1/$2.tar
tar -xf $2.tar -C $1/$2
rm $2.tar
cd ../../../affordance_extraction
fi

# XMem Mask Propagation
cd scripts
python affordance_extraction_preparation.py --rename_files_video ../../dataset/EPIC_DATA/frames/$1/$2 && \
cd ../XMem_Batch && \
python demo.py --images ../../dataset/EPIC_DATA/frames/$1/$2 && \
rm -r workspace/$2

# Preprocess Masks for Inpainting
cd ../scripts
echo 'Starting preprocessing'
python affordance_extraction_preparation.py --dilate_and_recolor ../../dataset/EPIC_DATA/xmem_masks/$2 10 && \
python affordance_extraction_preparation.py --preprocess_for_agent_inpainting ../XMem_Batch/segmentations/$2/hand/both ../../dataset/EPIC_DATA/frames/$1/$2 2 && \

# Agent Inpainting
cd ../agent_inpainting && \
python demo_script_batch.py ../XMem_Batch/segmentations/$2/hand/both ../../dataset/EPIC_DATA/hand_inpainting/$2 && \

# Delete Temporary Folder
rm -r ../XMem_Batch/segmentations/$2 && \
cd ../scripts && \

# Preprocess Data for Mask Completion
python affordance_extraction_preparation.py --preprocess_for_mask_completion ../../dataset/EPIC_DATA/frames/$1/$2/ ../../dataset/EPIC_DATA/hand_inpainting/$2 ../../dataset/EPIC_DATA/seq_for_mask_completion/$2 && \
cd ../sam2 && \

# Mask Completion
echo "Starting Mask Completion" && \
python mask_completion.py --videos ../../dataset/EPIC_DATA/seq_for_mask_completion/$2 --masks-left ../../dataset/EPIC_DATA/xmem_masks/$2/object/left --masks-right ../../dataset/EPIC_DATA/xmem_masks/$2/object/right --out ../../dataset/EPIC_DATA/completed_object_masks/$2 && \
cd ../scripts && \

# Affordance Extraction
echo "Post Processing Mask Differences" && \
python affordance_extraction_preparation.py --extract_affordances ../../dataset/EPIC_DATA/completed_object_masks/$2 ../../dataset/EPIC_DATA/xmem_masks/$2/hand ../../dataset/EPIC_DATA/affordance_masks/$2 && \
python affordance_extraction_preparation.py --process_affordances ../../dataset/EPIC_DATA/affordance_masks/$2 10 && \

# Preprocess and Reconfigure Data for Dataset
echo "Preprocess and reconfigure data for dataset"
#python affordance_extraction_preparation.py --modify_folder_to_sequence ../../dataset/EPIC_DATA/xmem_masks/$2/hand ../../dataset/EPIC_DATA/bimanual_annotations_json/$2 ../../dataset/EPIC_DATA/sequences_bimanual/$2
python affordance_extraction_preparation.py --apply_padding ../../dataset/EPIC_DATA/xmem_masks/$2/object && \

# Setup Dataset
echo "Setup Dataset"
cd data_setup
python setup_dataset.py --video-id $2 --out ../../../dataset/EPIC_DATA/$3/$2 --obj-name xmem_masks --aff-name affordance_masks && \
python filter_dataset.py $2 --dataset ../../../dataset/EPIC_DATA/$3 && \
python clean_up_data.py ../../../dataset/EPIC_DATA/$3/$2 && \
python update_taxonomy.py ../../../dataset/EPIC_DATA/$3/$2 && \
cd ..

# Augment Data
echo "Augment Data"
cd data_augmentation
python horizontal_flip.py ../../../dataset/EPIC_DATA/$3/$2 && \
python process_cropped_sequences.py ../../../dataset/EPIC_DATA/$3/$2 && \
python process_cropped_sequences.py ../../../dataset/EPIC_DATA/$3/flipped_$2 && \
python perform_color_correction.py ../../../dataset/EPIC_DATA/$3/$2 && \
python perform_color_correction.py ../../../dataset/EPIC_DATA/$3/flipped_$2 && \
python apply_jitter.py ../../../dataset/EPIC_DATA/$3/$2 && \
python apply_jitter.py ../../../dataset/EPIC_DATA/$3/flipped_$2 && \
cd ..

# Create Dataset
echo "Create Dataset"
python create_dataset.py --dir ../../dataset/EPIC_DATA/$3/$2 --out ../../dataset/EPIC_DATA/hdf5_sets/$3 --name $2 --categories all && \
python create_dataset.py --dir ../../dataset/EPIC_DATA/$3/flipped_$2 --out ../../dataset/EPIC_DATA/hdf5_sets/$3 --name flipped_$2 --categories all && \

# Remove Unnecessary Folders
rm -r ../../dataset/EPIC_DATA/sequences_bimanual/$2
rm -r ../../dataset/EPIC_DATA/padded_masks_unmodified_bimanual/$2
rm -r ../../dataset/EPIC_DATA/mask_completion/$2
rm -r ../../dataset/EPIC_DATA/xmem_masks_batched/$2
rm -r ../../dataset/EPIC_DATA/seq_for_mask_completion/$2
rm -r ../../dataset/EPIC_DATA/padded_frames/$2
#rm -r ../../dataset/EPIC_DATA/$3/$2
#rm -r ../../dataset/EPIC_DATA/$3/flipped_$2

# Archive Data
# mkdir ../../dataset/Archived_Set/affordance_masks
# mkdir ../../dataset/Archived_Set/frames
# mkdir ../../dataset/Archived_Set/hand_inpainting
# mkdir ../../dataset/Archived_Set/affordance_masks
# mkdir ../../dataset/Archived_Set/xmem_masks
# mkdir ../../dataset/Archived_Set/frames/$1/
# mkdir -p ../../dataset/Archived_Set/$3
# echo 'Zipping Affordance Masks'
# tar -czvf ../../dataset/Archived_Set/affordance_masks/$2.tar.gz ../../dataset/EPIC_DATA/affordance_masks/$2 --remove-files
# echo 'Zipping Hand Inpainting Masks'
# tar -czvf ../../dataset/Archived_Set/hand_inpainting/$2.tar.gz ../../dataset/EPIC_DATA/hand_inpainting/$2 --remove-files
# echo 'Zipping Mask Differences'
# tar -czvf ../../dataset/Archived_Set/affordance_masks/$2.tar.gz ../../dataset/EPIC_DATA/affordance_masks/$2 --remove-files
# echo 'Zipping Padded Masks'
# tar -czvf ../../dataset/Archived_Set/updated_padded_masks/$2.tar.gz ../../dataset/EPIC_DATA/updated_padded_masks/$2 --remove-files
# echo 'Zipping Xmem Masks'
# tar -czvf ../../dataset/Archived_Set/xmem_masks/$2.tar.gz ../../dataset/EPIC_DATA/xmem_masks/$2 --remove-files
# echo 'Zipping Frames'
# tar -czvf ../../dataset/Archived_Set/frames/$1/$2.tar.gz ../../dataset/EPIC_DATA/frames/$1/$2 --remove-files
# echo 'Zipping Dataset'
# tar -czvf ../../dataset/Archived_Set/$3/$2.tar.gz ../../dataset/EPIC_DATA/$3/$2 --remove-files
# tar -czvf ../../dataset/Archived_Set/$3/flipped_$2.tar.gz ../../dataset/EPIC_DATA/$3/flipped_$2 --remove-files














