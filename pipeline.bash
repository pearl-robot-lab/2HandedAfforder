#!/bin/sh

# Download and extract the EPIC-KITCHENS video

if [ ! -d EPIC_DATA/frames/$1/$2 ]; then
cd EPIC_DATA/frames
mkdir $1
cd $1
mkdir $2
cd ..
wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$1/$2.tar
tar -xf $2.tar -C $1/$2
rm $2.tar

# Propagate mask segmentation of in-contact frames
# cd scripts # Comment out if new video
cd ../..
fi
cd scripts
python helper.py --rename_files_video $2
cd ../XMem_Batch
python demo.py --images ../EPIC_DATA/frames/$1/$2
rm -r workspace/$2
# Preprocess masks
cd ../scripts
#cd scripts # Comment out if new video
python helper.py --dilate_masks_dir ../EPIC_DATA/xmem_masks/$2/hand/both
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/both
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/left
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/hand/right
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/object/left
python helper.py --recolor_masks_white ../EPIC_DATA/xmem_masks/$2/object/right
cd ..

# Perform hand inpainting
# iopaint run --image EPIC_DATA/frames/$1/$2 --mask EPIC_DATA/xmem_masks/$2/hand/both --output EPIC_DATA/hand_inpainting/$2 --model migan
cd scripts
python helper.py --fill_zeros_files ../XMem_Batch/segmentations/$2/hand/both
python helper.py --restructure_folder_for_agent_inpaint_folder ../XMem_Batch/segmentations/$2/hand/both --restructure_folder_for_agent_inpaint_out ../EPIC_DATA/xmem_masks_restructured/$2
python helper.py --add_raw_to_dir_dir ../EPIC_DATA/xmem_masks_restructured/$2 --add_raw_to_dir_raw ../EPIC_DATA/frames/$1/$2
python helper.py --recolor_whole_folder_structure ../EPIC_DATA/xmem_masks_restructured/$2
python helper.py --apply_padding_dir ../EPIC_DATA/xmem_masks_restructured/$2 --apply_padding_out ../EPIC_DATA/xmem_masks_padded/$2
python helper.py --batchify_agent_inp_dirs ../EPIC_DATA/xmem_masks_padded/$2 --batchify_agent_inp_out ../EPIC_DATA/xmem_masks_batched/$2 --batchify_agent_inp_batchsize 2
cd ../agent_inpainting
python demo_script_batch.py ../EPIC_DATA/xmem_masks_batched/$2 ../EPIC_DATA/hand_inpainting/$2
rm -r ../XMem_Batch/segmentations/$2
rm -r ../EPIC_DATA/xmem_masks_restructured/$2
rm -r ../EPIC_DATA/xmem_masks_batched/$2
cd ../scripts
# Perform object completion
#iopaint run --image EPIC_DATA/xmem_masks/$2/object/left --mask EPIC_DATA/xmem_masks/$2/hand/left --output EPIC_DATA/mask_completion/$2/left --model migan
#iopaint run --image EPIC_DATA/xmem_masks/$2/object/right --mask EPIC_DATA/xmem_masks/$2/hand/right --output EPIC_DATA/mask_completion/$2/right --model migan
python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/left --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/left
python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/right --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/right
python helper.py --batchify_migan_dir ../EPIC_DATA/xmem_masks/$2/hand/both --batchify_migan_batchsize 16 --batchify_migan_out ../EPIC_DATA/xmem_masks_batched/$2/hand/both
cd ../MI-GAN 
python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/xmem_masks/$2/object/left --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/left --output-dir ../EPIC_DATA/mask_completion/$2/left --device cuda --invert-mask --img-extension .png
python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/xmem_masks/$2/object/right --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/right --output-dir ../EPIC_DATA/mask_completion/$2/right --device cuda --invert-mask --img-extension .png
python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir ../EPIC_DATA/frames/$1/$2 --masks-dir ../EPIC_DATA/xmem_masks_batched/$2/hand/both --output-dir ../EPIC_DATA/inpainted_hands_migan/$2 --device cuda --invert-mask
rm -r ../EPIC_DATA/xmem_masks_batched/$2

# Perform post processing
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
#python helper.py --crop_masks_dir ../EPIC_DATA/mask_differences/$2/left --crop_masks_out ../EPIC_DATA/cropped_masks/$2/left
#python helper.py --crop_masks_dir ../EPIC_DATA/mask_differences/$2/right --crop_masks_out ../EPIC_DATA/cropped_masks/$2/right
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/left --apply_padding_single_folder_out ../EPIC_DATA/padded_masks/$2/left
python helper.py --apply_padding_single_folder_dir ../EPIC_DATA/mask_differences/$2/right --apply_padding_single_folder_out ../EPIC_DATA/padded_masks/$2/right
# Draw affordance dot
#python helper.py --draw_affordances_img ../EPIC_DATA/hand_inpainting/$2 --draw_affordances_mask ../EPIC_DATA/cropped_masks/$2/left --draw_affordances_dst ../EPIC_DATA/affordance_dots/$2/left
#python helper.py --draw_affordances_img ../EPIC_DATA/hand_inpainting/$2 --draw_affordances_mask ../EPIC_DATA/cropped_masks/$2/right --draw_affordances_dst ../EPIC_DATA/affordance_dots/$2/right
#python helper.py --merge_affordances_images ../EPIC_DATA/hand_inpainting/$2 --merge_affordances_folder_left ../EPIC_DATA/cropped_masks/$2/left --merge_affordances_folder_right ../EPIC_DATA/cropped_masks/$2/right --merge_affordances_out ../EPIC_DATA/affordance_dots/$2/both

# Draw affordance mask
#python helper.py --draw_mask_affordance_images ../EPIC_DATA/hand_inpainting/$2 --draw_mask_affordance_masks_left ../EPIC_DATA/cropped_masks/$2/left --draw_mask_affordance_masks_right ../EPIC_DATA/cropped_masks/$2/right --draw_mask_affordance_out ../EPIC_DATA/affordance_masks/$2/both
python helper.py --draw_mask_affordance_images ../EPIC_DATA/hand_inpainting/$2 --draw_mask_affordance_masks_left ../EPIC_DATA/padded_masks/$2/left --draw_mask_affordance_masks_right ../EPIC_DATA/padded_masks/$2/right --draw_mask_affordance_out ../EPIC_DATA/affordance_masks/$2/both




