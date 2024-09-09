import os
import shutil
from argparse import ArgumentParser
    
def setup_dataset_folder_structure(video_id, base_path, out, bim_name, aff_name, inp_name, obj_name):
    bimanual_annotations_path = os.path.join(base_path, bim_name, video_id)
    aff_mask_left_path = os.path.join(base_path, aff_name, video_id, "left")
    aff_mask_right_path = os.path.join(base_path, aff_name, video_id, "right")
    hand_inpainted_path = os.path.join(base_path, inp_name, video_id)
    obj_mask_left_path = os.path.join(base_path, obj_name, video_id, "object", "left")
    obj_mask_right_path = os.path.join(base_path, obj_name, video_id, "object", "right")

    if not os.path.exists(out):
        os.makedirs(out)
    
    aff_mask_left = os.listdir(aff_mask_left_path)
    aff_mask_right = os.listdir(aff_mask_right_path)
    bimanual_annotations = os.listdir(bimanual_annotations_path)

    for file in aff_mask_left:
        file_name = file.split('.')[0]

        corresponding_annotation = None
        for annotation in bimanual_annotations:
            annotation_frame = int(annotation.split('.')[0])
            if int(file_name) in range(annotation_frame - 10, annotation_frame + 10):
                corresponding_annotation = annotation
                break
        if corresponding_annotation == None:
            continue

        if not os.path.exists(os.path.join(out, file_name)):
            os.makedirs(os.path.join(out, file_name))
        if not (file in os.listdir(obj_mask_left_path) and file in os.listdir(hand_inpainted_path)):
            continue
        shutil.copy(os.path.join(bimanual_annotations_path, corresponding_annotation), os.path.join(out, file_name, "annotation.json"))
        shutil.copy(os.path.join(aff_mask_left_path, file), os.path.join(out, file_name, "aff_left.png"))
        shutil.copy(os.path.join(obj_mask_left_path, file), os.path.join(out, file_name, "obj_left.png"))
        if file in aff_mask_right:
            shutil.copy(os.path.join(aff_mask_right_path, file), os.path.join(out, file_name, "aff_right.png"))
            shutil.copy(os.path.join(obj_mask_right_path, file), os.path.join(out, file_name, "obj_right.png"))
        shutil.copy(os.path.join(hand_inpainted_path, file_name + ".png"), os.path.join(out, file_name, "inpainted_frame.png"))
    
    for file in aff_mask_right:
        if not (file in os.listdir(obj_mask_right_path) and file in os.listdir(hand_inpainted_path)):
            continue
        if not file in aff_mask_left:
            file_name = file.split('.')[0]
            corresponding_annotation = None
            for annotation in bimanual_annotations:
                annotation_frame = int(annotation.split('.')[0])
                if int(file_name) in range(annotation_frame - 10, annotation_frame + 10):
                    corresponding_annotation = annotation
                    break
            if corresponding_annotation == None:
                continue

            if not os.path.exists(os.path.join(out, file_name)):
                os.makedirs(os.path.join(out, file_name))

            shutil.copy(os.path.join(bimanual_annotations_path, corresponding_annotation), os.path.join(out, file_name, "annotation.json"))
            shutil.copy(os.path.join(aff_mask_right_path, file), os.path.join(out, file_name, "aff_right.png"))
            shutil.copy(os.path.join(obj_mask_right_path, file), os.path.join(out, file_name, "obj_right.png"))
            shutil.copy(os.path.join(hand_inpainted_path, file_name + ".png"), os.path.join(out, file_name, "inpainted_frame.png")) 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-id', default=None)  
    parser.add_argument('--out', default=None)
    parser.add_argument('--basepath', default="../EPIC_DATA")
    parser.add_argument('--bim-name', default="bimanual_annotations_json")
    parser.add_argument('--aff-name', default="cropped_masks")
    parser.add_argument('--inp-name', default="hand_inpainting")
    parser.add_argument('--obj-name', default="xmem_masks")

    args = parser.parse_args()
    vals = vars(args)

    if vals["video_id"] != None and vals["out"] != None:
        video_id = vals["video_id"]
        out = vals["out"]
        base_path = vals["basepath"]
        bim_name = vals["bim_name"]
        aff_name = vals["aff_name"]
        inp_name = vals["inp_name"]
        obj_name = vals["obj_name"]
        setup_dataset_folder_structure(video_id, base_path, out, bim_name, aff_name, inp_name, obj_name)
    else:
        print("Necessary arguments missing")
