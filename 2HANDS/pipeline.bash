#!/bin/sh

# Get script directory and set default data directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../dataset/EPIC_DATA}"
# Convert to absolute path
DATA_DIR="$(cd "${DATA_DIR}" 2>/dev/null && pwd || echo "${DATA_DIR}")"

# Download and extract the EPIC-KITCHENS video if it does not already exist (For different videos there exist different directories online)
if [ ! -d ${DATA_DIR}/frames/$1/$2 ]; then
cd ${DATA_DIR}/frames
mkdir $1
cd $1
mkdir $2
cd ..
wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$1/$2.tar
tar -xf $2.tar -C $1/$2
rm $2.tar
cd ${SCRIPT_DIR}
fi

# XMem Mask Propagation
cd scripts
python affordance_extraction_preparation.py --rename_files_video ${DATA_DIR}/frames/$1/$2 && \
cd ../modules/XMem_Batch && \
python demo.py --images ${DATA_DIR}/frames/$1/$2 && \
rm -r workspace/$2

# Preprocess Masks for Inpainting
cd ../../scripts
echo 'Starting preprocessing'
python affordance_extraction_preparation.py --dilate_and_recolor ${DATA_DIR}/xmem_masks/$2 10 && \
python affordance_extraction_preparation.py --preprocess_for_agent_inpainting ../modules/XMem_Batch/segmentations/$2/hand/both ${DATA_DIR}/frames/$1/$2 2 && \

# Agent Inpainting
cd ../modules/agent_inpainting && \
python demo_script_batch.py ../XMem_Batch/segmentations/$2/hand/both ${DATA_DIR}/hand_inpainting/$2 && \

# Delete Temporary Folder
rm -r ../XMem_Batch/segmentations/$2 && \
cd ../../scripts && \

# Preprocess Data for Mask Completion
python affordance_extraction_preparation.py --preprocess_for_mask_completion ${DATA_DIR}/frames/$1/$2/ ${DATA_DIR}/hand_inpainting/$2 ${DATA_DIR}/seq_for_mask_completion/$2 && \
cd ../modules/sam2 && \

# Mask Completion
echo "Starting Mask Completion" && \
python mask_completion.py --videos ${DATA_DIR}/seq_for_mask_completion/$2 --masks-left ${DATA_DIR}/xmem_masks/$2/object/left --masks-right ${DATA_DIR}/xmem_masks/$2/object/right --out ${DATA_DIR}/completed_object_masks/$2 && \
cd ../../scripts && \

# Affordance Extraction
echo "Post Processing Mask Differences" && \
python affordance_extraction_preparation.py --extract_affordances ${DATA_DIR}/completed_object_masks/$2 ${DATA_DIR}/xmem_masks/$2/hand ${DATA_DIR}/affordance_masks/$2 && \
python affordance_extraction_preparation.py --process_affordances ${DATA_DIR}/affordance_masks/$2 10 && \

# Preprocess and Reconfigure Data for Dataset
echo "Preprocess and reconfigure data for dataset"
#python affordance_extraction_preparation.py --modify_folder_to_sequence ${DATA_DIR}/xmem_masks/$2/hand ${DATA_DIR}/bimanual_annotations_json/$2 ${DATA_DIR}/sequences_bimanual/$2
python affordance_extraction_preparation.py --apply_padding ${DATA_DIR}/xmem_masks/$2/object && \

# Setup Dataset
echo "Setup Dataset"
cd data_setup
python setup_dataset.py --video-id $2 --out ${DATA_DIR}/$3/$2 --obj-name xmem_masks --aff-name affordance_masks && \
python filter_dataset.py $2 --dataset ${DATA_DIR}/$3 && \
python clean_up_data.py ${DATA_DIR}/$3/$2 && \
python update_taxonomy.py ${DATA_DIR}/$3/$2 && \
cd ..

# Augment Data
echo "Augment Data"
cd data_augmentation
python horizontal_flip.py ${DATA_DIR}/$3/$2 && \
python process_cropped_sequences.py ${DATA_DIR}/$3/$2 && \
python process_cropped_sequences.py ${DATA_DIR}/$3/flipped_$2 && \
python perform_color_correction.py ${DATA_DIR}/$3/$2 && \
python perform_color_correction.py ${DATA_DIR}/$3/flipped_$2 && \
python apply_jitter.py ${DATA_DIR}/$3/$2 && \
python apply_jitter.py ${DATA_DIR}/$3/flipped_$2 && \
cd ..

# Create Dataset
echo "Create Dataset"
python create_dataset.py --dir ${DATA_DIR}/$3/$2 --out ${DATA_DIR}/hdf5_sets/$3 --name $2 --categories all && \
python create_dataset.py --dir ${DATA_DIR}/$3/flipped_$2 --out ${DATA_DIR}/hdf5_sets/$3 --name flipped_$2 --categories all && \

# Remove Unnecessary Folders
rm -r ${DATA_DIR}/sequences_bimanual/$2
rm -r ${DATA_DIR}/padded_masks_unmodified_bimanual/$2
rm -r ${DATA_DIR}/mask_completion/$2
rm -r ${DATA_DIR}/xmem_masks_batched/$2
rm -r ${DATA_DIR}/seq_for_mask_completion/$2
rm -r ${DATA_DIR}/padded_frames/$2
#rm -r ${DATA_DIR}/$3/$2
#rm -r ${DATA_DIR}/$3/flipped_$2

# Archive Data
# ARCHIVE_DIR="${SCRIPT_DIR}/../dataset/Archived_Set"
# mkdir ${ARCHIVE_DIR}/affordance_masks
# mkdir ${ARCHIVE_DIR}/frames
# mkdir ${ARCHIVE_DIR}/hand_inpainting
# mkdir ${ARCHIVE_DIR}/affordance_masks
# mkdir ${ARCHIVE_DIR}/xmem_masks
# mkdir ${ARCHIVE_DIR}/frames/$1/
# mkdir -p ${ARCHIVE_DIR}/$3
# echo 'Zipping Affordance Masks'
# tar -czvf ${ARCHIVE_DIR}/affordance_masks/$2.tar.gz ${DATA_DIR}/affordance_masks/$2 --remove-files
# echo 'Zipping Hand Inpainting Masks'
# tar -czvf ${ARCHIVE_DIR}/hand_inpainting/$2.tar.gz ${DATA_DIR}/hand_inpainting/$2 --remove-files
# echo 'Zipping Mask Differences'
# tar -czvf ${ARCHIVE_DIR}/affordance_masks/$2.tar.gz ${DATA_DIR}/affordance_masks/$2 --remove-files
# echo 'Zipping Padded Masks'
# tar -czvf ${ARCHIVE_DIR}/updated_padded_masks/$2.tar.gz ${DATA_DIR}/updated_padded_masks/$2 --remove-files
# echo 'Zipping Xmem Masks'
# tar -czvf ${ARCHIVE_DIR}/xmem_masks/$2.tar.gz ${DATA_DIR}/xmem_masks/$2 --remove-files
# echo 'Zipping Frames'
# tar -czvf ${ARCHIVE_DIR}/frames/$1/$2.tar.gz ${DATA_DIR}/frames/$1/$2 --remove-files
# echo 'Zipping Dataset'
# tar -czvf ${ARCHIVE_DIR}/$3/$2.tar.gz ${DATA_DIR}/$3/$2 --remove-files
# tar -czvf ${ARCHIVE_DIR}/$3/flipped_$2.tar.gz ${DATA_DIR}/$3/flipped_$2 --remove-files














