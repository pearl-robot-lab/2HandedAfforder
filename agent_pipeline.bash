#!/bin/sh

cd scripts
python helper.py --fill_zeros_files ../XMem_Batch/segmentations/$2/hand/both
python helper.py --restructure_folder_for_agent_inpaint_folder ../XMem_Batch/segmentations/$2/hand/both --restructure_folder_for_agent_inpaint_out ../EPIC_DATA/xmem_masks_restructured/$2
python helper.py --add_raw_to_dir_dir ../EPIC_DATA/xmem_masks_restructured/$2 --add_raw_to_dir_raw ../EPIC_DATA/frames/$1/$2
python helper.py --recolor_whole_folder_structure ../EPIC_DATA/xmem_masks_restructured/$2

