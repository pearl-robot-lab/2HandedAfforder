#!/bin/sh

# Download and extract the EPIC-KITCHENS video if it does not already exist
if [ ! -d EPIC_DATA/frames/$1/$2 ]; then
cd EPIC_DATA/frames
mkdir $1
cd $1
mkdir $2
cd ..
wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$1/$2.tar
tar -xf $2.tar -C $1/$2
rm $2.tar
cd ../..
fi

# XMem mask propagation
cd scripts
python helper.py --rename_files_video $2
cd ../XMem_Batch
python demo.py --images ../EPIC_DATA/frames/$1/$2
rm -r workspace/$2

# Preprocess masks for inpainting
cd ../scripts
python helper.py --dilate_masks_dir ../EPIC_DATA/xmem_masks/$2/hand/both
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/both
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/left
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/right
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/object/left
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/object/right
python helper.py --fill_zeros_files ../XMem_Batch/segmentations/$2/hand/both
python helper.py --restructure_folder_for_agent_inpaint_folder ../XMem_Batch/segmentations/$2/hand/both --restructure_folder_for_agent_inpaint_out ../EPIC_DATA/xmem_masks_restructured/$2
python helper.py --add_raw_to_dir_dir ../EPIC_DATA/xmem_masks_restructured/$2 --add_raw_to_dir_raw ../EPIC_DATA/frames/$1/$2
python helper.py --recolor_whole_folder_structure ../EPIC_DATA/xmem_masks_restructured/$2
python helper.py --apply_padding_dir ../EPIC_DATA/xmem_masks_restructured/$2 --apply_padding_out ../EPIC_DATA/xmem_masks_padded/$2
python helper.py --batchify_agent_inp_dirs ../EPIC_DATA/xmem_masks_padded/$2 --batchify_agent_inp_out ../EPIC_DATA/xmem_masks_batched/$2 --batchify_agent_inp_batchsize 2

# Agent inpainting
cd ../agent_inpainting
python demo_script_batch.py ../EPIC_DATA/xmem_masks_batched/$2 ../EPIC_DATA/hand_inpainting/$2
rm -r ../XMem_Batch/segmentations/$2
rm -r ../EPIC_DATA/xmem_masks_restructured/$2
rm -r ../EPIC_DATA/xmem_masks_padded/$2
rm -r ../EPIC_DATA/xmem_masks_batched/$2
cd ../scripts

# Preprocess masks for object completion
python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/left --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/left
python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/right --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/right
# python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/both --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/both

# Object completion
# cd ../MI-GAN 
# python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/xmem_masks/$2/object/left --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/left --output-dir ../EPIC_DATA/mask_completion/$2/left --device cuda --invert-mask --img-extension .png
# python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/xmem_masks/$2/object/right --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/right --output-dir ../EPIC_DATA/mask_completion/$2/right --device cuda --invert-mask --img-extension .png
# python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/frames/$1/$2 --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/both --output-dir ../EPIC_DATA/inpainted_hands_migan/$2 --device cuda --invert-mask

# Process results
cd ../scripts
python helper.py --regenerate_mask_src ../EPIC_DATA/mask_completion/$2/left 
python helper.py --regenerate_mask_src ../EPIC_DATA/mask_completion/$2/right
python helper.py --calculate_mask_difference_pre ../EPIC_DATA/xmem_masks/$2/object/left/ --calculate_mask_difference_post ../EPIC_DATA/mask_completion/$2/left --calculate_mask_difference_out ../EPIC_DATA/mask_differences/$2/left
python helper.py --calculate_mask_difference_pre ../EPIC_DATA/xmem_masks/$2/object/right/ --calculate_mask_difference_post ../EPIC_DATA/mask_completion/$2/right --calculate_mask_difference_out ../EPIC_DATA/mask_differences/$2/right
python helper.py --delete_empty_masks ../EPIC_DATA/mask_differences/$2/left
python helper.py --delete_empty_masks ../EPIC_DATA/mask_differences/$2/right
python helper.py --erode_masks_dir ../EPIC_DATA/mask_differences/$2/left
python helper.py --erode_masks_dir ../EPIC_DATA/mask_differences/$2/right
python helper.py --dilate_masks_dir ../EPIC_DATA/mask_differences/$2/left
python helper.py --dilate_masks_dir ../EPIC_DATA/mask_differences/$2/right
python helper.py --recolor_masks_white ../EPIC_DATA/mask_differences/$2/left
python helper.py --recolor_masks_white ../EPIC_DATA/mask_differences/$2/right
# python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/left --apply_padding_single_folder_out ../EPIC_DATA/padded_masks/$2/left
# python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/right --apply_padding_single_folder_out ../EPIC_DATA/padded_masks/$2/right

# Preprocess data for dataset
# python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$2/hand/left --ref-folder ../EPIC_DATA/bimanual_annotations_json/$2 --out ../EPIC_DATA/sequences_bimanual/$2/left
# python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$2/hand/right --ref-folder ../EPIC_DATA/bimanual_annotations_json/$2 --out ../EPIC_DATA/sequences_bimanual/$2/right
# python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$2/object/left --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified_bimanual/$2/object/left
# python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$2/object/right --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified_bimanual/$2/object/right

# Create dataset
# python setup_dataset.py --video-id $2 --out ../EPIC_DATA/$3/$2 --obj-name padded_masks_unmodified_bimanual --aff-name padded_masks
# python filter_dataset.py $2 --dataset ../EPIC_DATA/$3
# python clean_up_data.py ../EPIC_DATA/$3/$2
# python update_taxonomy.py ../EPIC_DATA/$3/$2
# python horizontal_flip.py ../EPIC_DATA/$3/$2
# python process_cropped_sequences.py ../EPIC_DATA/$3/$2
# python process_cropped_sequences.py ../EPIC_DATA/$3/flipped_$2
# python perform_color_correction.py ../EPIC_DATA/$3/$2
# python perform_color_correction.py ../EPIC_DATA/$3/flipped_$2
# python apply_jitter.py ../EPIC_DATA/$3/$2
# python apply_jitter.py ../EPIC_DATA/$3/flipped_$2
# python create_dataset.py --dir ../EPIC_DATA/$3/$2 --out ../EPIC_DATA/hdf5_sets/$3 --name $2 --categories all
# python create_dataset.py --dir ../EPIC_DATA/$3/flipped_$2 --out ../EPIC_DATA/hdf5_sets/$3 --name flipped_$2 --categories all

python preprocess_for_mask_completion.py ../EPIC_DATA/frames/$1/$2/ ../EPIC_DATA/hand_inpainting/$2 ../EPIC_DATA/seq_for_mask_completion/$2
cd ../sam2
echo "Starting Mask Completion"
python mask_completion.py --videos ../EPIC_DATA/seq_for_mask_completion/$2 --masks-left ../EPIC_DATA/xmem_masks/$2/object/left --masks-right ../EPIC_DATA/xmem_masks/$2/object/right --out ../EPIC_DATA/updated_mask_completion/$2
cd ../scripts
echo "Calculating Mask Differences"
python helper.py --calculate_mask_difference_pre ../EPIC_DATA/xmem_masks/$2/object/left/ --calculate_mask_difference_post ../EPIC_DATA/updated_mask_completion/$2/left --calculate_mask_difference_out ../EPIC_DATA/updated_mask_differences/$2/left
python helper.py --calculate_mask_difference_pre ../EPIC_DATA/xmem_masks/$2/object/right/ --calculate_mask_difference_post ../EPIC_DATA/updated_mask_completion/$2/right --calculate_mask_difference_out ../EPIC_DATA/updated_mask_differences/$2/right
echo "Post Processing Mask Differences"
python determine_mask_overlap.py ../EPIC_DATA/updated_mask_differences/$2/left ../EPIC_DATA/xmem_masks/$2/hand/left
python determine_mask_overlap.py ../EPIC_DATA/updated_mask_differences/$2/right ../EPIC_DATA/xmem_masks/$2/hand/right
python helper.py --delete_empty_masks ../EPIC_DATA/updated_mask_differences/$2/left
python helper.py --delete_empty_masks ../EPIC_DATA/updated_mask_differences/$2/right
python helper.py --erode_masks_dir ../EPIC_DATA/updated_mask_differences/$2/left
python helper.py --erode_masks_dir ../EPIC_DATA/updated_mask_differences/$2/right
python helper.py --dilate_masks_dir ../EPIC_DATA/updated_mask_differences/$2/left
python helper.py --dilate_masks_dir ../EPIC_DATA/updated_mask_differences/$2/right
python helper.py --recolor_masks_white ../EPIC_DATA/updated_mask_differences/$2/left
python helper.py --recolor_masks_white ../EPIC_DATA/updated_mask_differences/$2/right
#python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/left --apply_padding_single_folder_out ../EPIC_DATA/updated_padded_masks/$2/left
#python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/right --apply_padding_single_folder_out ../EPIC_DATA/updated_padded_masks/$2/right
#python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/frames/$1/$2 --apply_padding_single_folder_out ../EPIC_DATA/padded_frames/$2
#echo "Visualizing Masks"
#python visualize_masks.py ../EPIC_DATA/hand_inpainting/$2 ../EPIC_DATA/updated_padded_masks/$2/left ../EPIC_DATA/updated_mask_differences/$2/left ../EPIC_DATA/padded_frames/$2 ../EPIC_DATA/visualizations/$2/left --iou_threshold 0.9
#python visualize_masks.py ../EPIC_DATA/hand_inpainting/$2 ../EPIC_DATA/updated_padded_masks/$2/right ../EPIC_DATA/updated_mask_differences/$2/right ../EPIC_DATA/padded_frames/$2 ../EPIC_DATA/visualizations/$2/right --iou_threshold 0.9
# Preprocess data for dataset
python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$2/hand/left --ref-folder ../EPIC_DATA/bimanual_annotations_json/$2 --out ../EPIC_DATA/sequences_bimanual/$2/left
python modify_folder_to_sequence.py --dir ../EPIC_DATA/xmem_masks/$2/hand/right --ref-folder ../EPIC_DATA/bimanual_annotations_json/$2 --out ../EPIC_DATA/sequences_bimanual/$2/right
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$2/object/left --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified_bimanual/$2/object/left
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/xmem_masks/$2/object/right --apply_padding_single_folder_out ../EPIC_DATA/padded_masks_unmodified_bimanual/$2/object/right

# Create dataset
python setup_dataset.py --video-id $2 --out ../EPIC_DATA/$3/$2 --obj-name padded_masks_unmodified_bimanual --aff-name updated_mask_differences
python filter_dataset.py $2 --dataset ../EPIC_DATA/$3
python clean_up_data.py ../EPIC_DATA/$3/$2
python update_taxonomy.py ../EPIC_DATA/$3/$2
python horizontal_flip.py ../EPIC_DATA/$3/$2
python process_cropped_sequences.py ../EPIC_DATA/$3/$2
python process_cropped_sequences.py ../EPIC_DATA/$3/flipped_$2
python perform_color_correction.py ../EPIC_DATA/$3/$2
python perform_color_correction.py ../EPIC_DATA/$3/flipped_$2
python apply_jitter.py ../EPIC_DATA/$3/$2
python apply_jitter.py ../EPIC_DATA/$3/flipped_$2
python create_dataset.py --dir ../EPIC_DATA/$3/$2 --out ../EPIC_DATA/hdf5_sets/$3 --name $2 --categories all
python create_dataset.py --dir ../EPIC_DATA/$3/flipped_$2 --out ../EPIC_DATA/hdf5_sets/$3 --name flipped_$2 --categories all

# Remove unnecessary folders
cd ../EPIC_DATA
# "$3/$2" "$3/flipped_$2"
#directories=("sequences_bimanual/$2" "updated_mask_completion/$2" "seq_for_mask_completion/$2" "padded_frames/$2" "padded_masks_unmodified_bimanual/$2" "xmem_masks_batched/$2")

#for dir in "${directories[@]}"; do
#  # Check if the directory exists
#  if [ -d "$dir" ]; then
#  echo "Deleting directory: $dir"
#    rm -rf "$dir"  # Use rm -rf to remove the directory and its contents
#  else
#    echo "Directory $dir does not exist, skipping."
#  fi
#done

# Remove unnecessary folders
#rm -r ../EPIC_DATA/sequences_bimanual/$2
#rm -r ../EPIC_DATA/padded_masks_unmodified_bimanual/$2
#rm -r ../EPIC_DATA/mask_completion/$2
#rm -r ../EPIC_DATA/updated_mask_completion/$2
#rm -r ../EPIC_DATA/xmem_masks_batched/$2
#rm -r ../EPIC_DATA/seq_for_mask_completion/$2
#rm -r ../EPIC_DATA/padded_frames/$2
#rm -r ../EPIC_DATA/$3/$2

# Archive data
#mkdir ../Archived_Set/affordance_masks
#mkdir ../Archived_Set/frames
#mkdir ../Archived_Set/hand_inpainting
#mkdir ../Archived_Set/updated_mask_differences
#mkdir ../Archived_Set/xmem_masks
#mkdir ../Archived_Set/frames/$1/
#mkdir -p ../Archived_Set/$3
#echo 'Zipping Affordance Masks'
#tar -czvf ../Archived_Set/affordance_masks/$2.tar.gz ../EPIC_DATA/affordance_masks/$2 --remove-files
#echo 'Zipping Hand Inpainting Masks'
#tar -czvf ../Archived_Set/hand_inpainting/$2.tar.gz ../EPIC_DATA/hand_inpainting/$2 --remove-files
#echo 'Zipping Mask Differences'
#tar -czvf ../Archived_Set/updated_mask_differences/$2.tar.gz ../EPIC_DATA/updated_mask_differences/$2 --remove-files
#echo 'Zipping Padded Masks'
#tar -czvf ../Archived_Set/updated_padded_masks/$2.tar.gz ../EPIC_DATA/updated_padded_masks/$2 --remove-files
#echo 'Zipping Xmem Masks'
#tar -czvf ../Archived_Set/xmem_masks/$2.tar.gz ../EPIC_DATA/xmem_masks/$2 --remove-files
#echo 'Zipping Frames'
#tar -czvf ../Archived_Set/frames/$1/$2.tar.gz ../EPIC_DATA/frames/$1/$2 --remove-files
#echo 'Zipping Dataset'
#tar -czvf ../Archived_Set/$3/$2.tar.gz ../EPIC_DATA/$3/$2 --remove-files
#tar -czvf ../Archived_Set/$3/flipped_$2.tar.gz ../EPIC_DATA/$3/flipped_$2 --remove-files














