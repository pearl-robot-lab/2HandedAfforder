#!/bin/sh

cd EPIC_DATA
rm -r affordance_masks/$1
rm -r cropped_masks/$1
rm -r hand_inpainting/$1
#rm -r frames/$1/$2
rm -r mask_completion/$1
rm -r mask_differences/$1
rm -r padded_masks/$1
rm -r xmem_masks/$1
rm -r xmem_masks_padded/$1
rm -r xmem_masks_restructured/$1
rm -r xmem_masks_batched/$1
rm -r ../XMem_Batch/segmentations/$1
rm -r ../XMem_Batch/workspace/$1
