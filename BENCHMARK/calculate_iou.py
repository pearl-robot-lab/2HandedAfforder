import os
import numpy as np
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import directed_hausdorff
import random

def calculate_hausdorff(mask1, mask2):
    shp = mask1.shape
    mask1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(mask2) == 0:
        print("I was here")
        return np.sqrt(shp[0]**2 + shp[1] ** 2), np.sqrt(shp[0]**2 + shp[1] ** 2)
    if len(mask1) == 0:
        return 0, 0
    mask1 = np.vstack(mask1[0]).squeeze()
    mask2 = np.vstack(mask2[0]).squeeze()
    if len(mask2.shape) == 1:
        mask2 = np.array([mask2])
    if len(mask1.shape) == 1:
        mask1 = np.array([mask1])
    return directed_hausdorff(mask2, mask1)[0], max(directed_hausdorff(mask1, mask2)[0], directed_hausdorff(mask2, mask1)[0])

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Args:mask1
    - mask1: First binary mask (numpy array).
    - mask2: Second binary mask (numpy array).
    Returns:
    - IoU value (float).
    """
    if np.array_equal(mask1, np.zeros((0, 0))) or np.array_equal(mask2, np.zeros((0, 0))):
        return

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0.0
    return iou

def create_overlay(image_path, aff_left_mask, aff_right_mask, caption, alpha=0.5, beta=0.5, gamma=0):
    """
    Create an overlay of the inpainting image with aff_left and aff_right masks.
    Uses cv2.addWeighted to combine the inpainting and mask images.
    
    Args:
    - image_path: Path to the inpainting image (png or jpeg).
    - aff_left_mask: The aff_left binary mask (numpy array).
    - aff_right_mask: The aff_right binary mask (numpy array).
    - caption: Caption text to be added to the image.
    - alpha: Weight of the inpainting image.
    - beta: Weight of the combined masks (aff_left and aff_right).
    - gamma: Scalar added to each sum.
    
    Returns:
    - Overlayed image in PIL format with caption.
    """
    # Load the inpainting image using OpenCV
    inpainting = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Ensure masks are binary (0 or 255)
    aff_left_mask = np.uint8(aff_left_mask * 255)  # Convert to 0-255 range
    aff_right_mask = np.uint8(aff_right_mask * 255)  # Convert to 0-255 range

    mask_size = (inpainting.shape[1], inpainting.shape[0])  # (width, height)
    aff_left_mask = cv2.resize(aff_left_mask, mask_size, interpolation=cv2.INTER_NEAREST)
    aff_right_mask = cv2.resize(aff_right_mask, mask_size, interpolation=cv2.INTER_NEAREST)

    # Convert masks to 3-channel (RGB) images to overlay them on inpainting
    aff_left_mask_rgb = cv2.cvtColor(aff_left_mask, cv2.COLOR_GRAY2BGR)
    aff_right_mask_rgb = cv2.cvtColor(aff_right_mask, cv2.COLOR_GRAY2BGR)

    # Color the masks (Red for aff_left, Blue for aff_right)
    aff_left_mask_rgb[:, :, :2] = 0  # Zero out Green and Blue channels for Red color
    aff_right_mask_rgb[:, :, 0] = 0  # Zero out Red and Green channels for Blue color
    aff_right_mask_rgb[:, :, 2] = 0

    # Combine the inpainting with the colored masks using cv2.addWeighted
    combined = cv2.addWeighted(inpainting, alpha, aff_left_mask_rgb, beta, gamma)
    combined = cv2.addWeighted(combined, 1, aff_right_mask_rgb, beta, gamma)

    # Convert the result back to PIL format
    overlay_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    if caption != None:
        # Add the caption to the overlay
        draw = ImageDraw.Draw(overlay_pil)
        font = ImageFont.load_default()  # You can specify a custom font here
        text_width, text_height = draw.textsize(caption, font=font)
        text_position = (overlay_pil.width - text_width - 10, 10)  # Position the text at the top-right corner
        draw.text(text_position, caption, fill="white", font=font)

    return overlay_pil

def calculate_iocm(benchmark_mask, comparison_mask):
    """
    Calculate Intersection over Comparison Mask (IoCM) between two binary masks.
    
    Args:
    - benchmark_mask: The benchmark mask (numpy array).
    - comparison_mask: The comparison mask (numpy array).
    
    Returns:
    - IoCM value (float).
    """
    if np.array_equal(comparison_mask, np.zeros((0, 0))) or np.array_equal(benchmark_mask, np.zeros((0, 0))):
        return None  # If comparison mask is empty, return None
    
    intersection = np.logical_and(benchmark_mask, comparison_mask).sum()
    comparison_area = comparison_mask.sum()
    
    iocm = intersection / comparison_area if comparison_area != 0 else 0.0
    return iocm


def visualize_examples(benchmark_folder, comparison_folder, visualize_dir, caption="Aff-Ex", visualize=True, n_examples=20, files_folder=None, only='', calc_map=False, is_cropped=False, take_intersection=False):
    """
    Visualize a random selection of examples by overlaying aff_left, aff_right on inpainting.png
    and concatenating images. Also calculate IoU and IoCM for each example.
    """
    all_subfolders = os.listdir(benchmark_folder)
    if only == 'ego':
        all_subfolders = [s for s in all_subfolders if not s.startswith('P')]
    if only == 'epic':
        all_subfolders = [s for s in all_subfolders if s.startswith('P')]
    if calc_map:
        threshold_folders = os.listdir(comparison_folder)
    else:
        threshold_folders = ['.']

    random.shuffle(all_subfolders)

    th_iocms = []
    th_ious = []
    th_hds = []
    th_dir_hds = []
    if not is_cropped:
        orig_shape = (855, 855)
    #import pdb; pdb.set_trace()
    for threshold_folder in threshold_folders:
        thresh_dir = os.path.join(comparison_folder, threshold_folder)
        total_iou = 0
        total_iocm = 0
        total_hd = 0
        total_directed_hd = 0
        count = 0
        zero_count = 0
        for subfolder in all_subfolders:
            benchmark_subfolder = os.path.join(benchmark_folder, subfolder)
            comparison_subfolder = os.path.join(thresh_dir, subfolder)
            if files_folder:
                files = os.listdir(files_folder)
                vids = []
                frames = []
                for file in files:
                    parts = file.split('_')
                    if len(parts) > 2:
                        vids.append(parts[0] + '_' + parts[1])
                        frames.append(parts[2].split('.')[0])
                    else:
                        vids.append(parts[0])
                        frames.append(parts[1].split('.')[0])
            #import pdb; pdb.set_trace()
            if os.path.isdir(benchmark_subfolder) and os.path.isdir(comparison_subfolder):
                for leaf_subfolder in os.listdir(benchmark_subfolder):
                    benchmark_leaf_subfolder = os.path.join(benchmark_subfolder, leaf_subfolder)
                    comparison_leaf_subfolder = os.path.join(comparison_subfolder, leaf_subfolder)
                    if os.path.isdir(benchmark_leaf_subfolder) and os.path.isdir(comparison_leaf_subfolder):
                        if files_folder:
                            if leaf_subfolder in frames:
                                if not vids[frames.index(leaf_subfolder)] == subfolder:
                                    continue
                            else:
                                continue
                        # Load the inpainting image for both benchmark and comparison  
                        benchmark_inpainting_path = os.path.join(benchmark_leaf_subfolder, 'inpainting.png')
                        comparison_inpainting_path = os.path.join(benchmark_leaf_subfolder, 'inpainting.png')
                        #print(cv2.imread(benchmark_inpainting_path).shape)
                        #print(cv2.imread(comparison_inpainting_path).shape)
                        if is_cropped:
                            orig_shape = cv2.imread(benchmark_inpainting_path).shape[:2]
                            orig_shape = (orig_shape[1], orig_shape[0])
                        #print("Orig Shape: ", orig_shape)

                        if os.path.exists(benchmark_inpainting_path) and os.path.exists(comparison_inpainting_path):
                            # Read aff_left and aff_right masks for both benchmark and comparison
                            aff_left_benchmark_path = os.path.join(benchmark_leaf_subfolder, 'aff_left.png')
                            aff_right_benchmark_path = os.path.join(benchmark_leaf_subfolder, 'aff_right.png')
                            aff_left_comparison_path = os.path.join(comparison_leaf_subfolder, 'aff_left.png')
                            aff_right_comparison_path = os.path.join(comparison_leaf_subfolder, 'aff_right.png')

                            # Load masks
                            benchmark_aff_left = np.zeros((0, 0))
                            benchmark_aff_right = np.zeros((0, 0))
                            comparison_aff_left = np.zeros((0, 0))
                            comparison_aff_right = np.zeros((0, 0))

                            if os.path.exists(aff_left_benchmark_path):
                                benchmark_aff_left = cv2.imread(aff_left_benchmark_path, cv2.IMREAD_GRAYSCALE)
                                benchmark_aff_left = benchmark_aff_left > 0  # Binary mask

                            if os.path.exists(aff_right_benchmark_path):
                                benchmark_aff_right = cv2.imread(aff_right_benchmark_path, cv2.IMREAD_GRAYSCALE)
                                benchmark_aff_right = benchmark_aff_right > 0  # Binary mask

                            if os.path.exists(aff_left_comparison_path):
                                comparison_aff_left = cv2.imread(aff_left_comparison_path, cv2.IMREAD_GRAYSCALE)
                                comparison_aff_left = cv2.resize(comparison_aff_left, orig_shape)

                                if take_intersection:
                                    obj_left = cv2.imread(os.path.join(benchmark_leaf_subfolder, 'obj_left.png'), cv2.IMREAD_GRAYSCALE)
                                    if obj_left is None:
                                        continue
                                    else:
                                        if obj_left.shape != comparison_aff_left.shape:
                                            continue
                                        comparison_aff_left = cv2.bitwise_and(comparison_aff_left, obj_left)
                                
                                comparison_aff_left = comparison_aff_left > 0  # Binary mask

                            if os.path.exists(aff_right_comparison_path):
                                comparison_aff_right = cv2.imread(aff_right_comparison_path, cv2.IMREAD_GRAYSCALE)
                                comparison_aff_right = cv2.resize(comparison_aff_right, orig_shape)

                                if take_intersection:
                                    obj_right = cv2.imread(os.path.join(benchmark_leaf_subfolder, 'obj_right.png'), cv2.IMREAD_GRAYSCALE)
                                    if obj_right is None:
                                        continue
                                    else:
                                        if obj_right.shape != comparison_aff_right.shape:
                                            continue
                                        comparison_aff_right = cv2.bitwise_and(comparison_aff_right, obj_right)
                                comparison_aff_right = comparison_aff_right > 0  # Binary mask

                            # Union of aff_left and aff_right for benchmark
                            if not np.array_equal(benchmark_aff_left, np.zeros((0, 0))) and not np.array_equal(benchmark_aff_right, np.zeros((0, 0))):
                                benchmark_union = np.logical_or(benchmark_aff_left, benchmark_aff_right)
                            elif not np.array_equal(benchmark_aff_left, np.zeros((0, 0))):
                                benchmark_union = benchmark_aff_left
                                benchmark_aff_right = np.zeros_like(benchmark_aff_left)
                            else:
                                benchmark_union = benchmark_aff_right
                                benchmark_aff_left = np.zeros_like(benchmark_aff_right)

                            # Union of aff_left and aff_right for comparison
                            if not np.array_equal(comparison_aff_left, np.zeros((0, 0))) and not np.array_equal(comparison_aff_right, np.zeros((0, 0))):
                                comparison_union = np.logical_or(comparison_aff_left, comparison_aff_right)
                            elif not np.array_equal(comparison_aff_left, np.zeros((0, 0))):
                                comparison_union = comparison_aff_left
                                comparison_aff_right = np.zeros_like(comparison_aff_left)
                            else:
                                comparison_union = comparison_aff_right
                                comparison_aff_left = np.zeros_like(comparison_aff_right)

                            # Calculate IoU
                            iou = calculate_iou(benchmark_union, comparison_union)
                            iocm = calculate_iocm(benchmark_union, comparison_union)

                            if iou is not None and iocm is not None:
                                directed_hd, hd = calculate_hausdorff(benchmark_union, comparison_union)
                                # Print IoU and IoCM for each subfolder and leaf subfolder
                                print(f"IoU for {subfolder}/{leaf_subfolder}: {iou:.4f}")
                                print(f"IoCM for {subfolder}/{leaf_subfolder}: {iocm:.4f}")
                                total_iou += iou
                                total_iocm += iocm
                                print(f"HD: {hd} and Directed HD: {directed_hd}")
                                total_hd += hd
                                total_directed_hd += directed_hd

                                # If visualize flag is set, create the visualizations
                                if visualize:
                                    if caption != '.':
                                        # Create overlay for benchmark image
                                        benchmark_caption = f"{subfolder}/{leaf_subfolder}"
                                        benchmark_img = create_overlay(benchmark_inpainting_path, benchmark_aff_left, benchmark_aff_right, benchmark_caption)

                                        # Create overlay for comparison image
                                        comparison_caption = caption
                                        comparison_img = create_overlay(comparison_inpainting_path, comparison_aff_left, comparison_aff_right, comparison_caption)

                                        # Concatenate images horizontally
                                        concatenated_img = Image.new('RGB', (benchmark_img.width + comparison_img.width, max(benchmark_img.height, comparison_img.height)))
                                        concatenated_img.paste(benchmark_img, (0, 0))
                                        concatenated_img.paste(comparison_img, (benchmark_img.width, 0))

                                        # Add IoU and IoCM captions to the concatenated image
                                        draw = ImageDraw.Draw(concatenated_img)
                                        font = ImageFont.load_default()
                                        draw.text((10, 10), f"IoU: {iou:.4f}", fill="white", font=font)

                                        # Save the concatenated image
                                        concatenated_output_path = os.path.join(visualize_dir, f"{subfolder}_{leaf_subfolder}_concatenated.png")
                                        concatenated_img.save(concatenated_output_path)
                                    elif iou > 0.2 or files_folder:
                                        if benchmark_subfolder.split('/')[-1] == '126fe8f1-226a-4161-ad6e-84f9e5baeb3a' and benchmark_leaf_subfolder.split('/')[-1] == '00000459':
                                            comparison_aff_right = np.zeros_like(comparison_aff_left)
                                        comparison_img = create_overlay(comparison_inpainting_path, comparison_aff_left, comparison_aff_right, None)
                                        output_path = os.path.join(visualize_dir, f"{subfolder}_{leaf_subfolder}.png")
                                        comparison_img.save(output_path)
                                
                                if iou == 0 and iocm == 0:
                                    zero_count += 1
                                count += 1
                                if count >= n_examples:
                                    return
                    else:
                        print(f"Invalid Leaf Directory {comparison_leaf_subfolder}")
            else:
                print(f"Invalid Directory {comparison_subfolder}")
        th_iocms.append(total_iocm/count)
        th_ious.append(total_iou/count)
        th_hds.append(total_hd/count)
        th_dir_hds.append(total_directed_hd/count)
    max_idx = np.argmax(np.array(th_iocms))
    best_iocm = th_iocms[max_idx]
    best_version = threshold_folders[max_idx]
    best_iou = th_ious[max_idx]
    best_hd = th_hds[max_idx]
    best_dir_hd = th_dir_hds[max_idx]
    if not calc_map:
        print(f"Total Failed Predictions: {zero_count}")
        print(f"Total Averaged IoU: {best_iou}")
        print(f"Total Averaged IoCM: {best_iocm}")
        print(f"Total Averaged Hausdorff Distance: {best_hd}")
        print(f"Total Averaged Directed Hausdorff Distance: {best_dir_hd}")
    else:
        print(f"mean average precision: {sum(th_iocms)/len(th_iocms)}")
        print(f"Best performing threshold was {best_version}")
        print(f"IoU: {best_iou}")
        print(f"Precision: {best_iocm}")
        print(f"Hausdorff-Distance: {best_hd}")
        print(f"Directed Hausdorff-Distance: {best_dir_hd}")
        
    


def main():
    parser = argparse.ArgumentParser(description="Calculate IoU between corresponding leaf subfolders in benchmark and comparison folders.")
    parser.add_argument('--benchmark_folder', type=str, default='data/cropped', help="Benchmark folder containing subfolders and leaf subfolders with 'aff_left.png' and 'aff_right.png'.")
    parser.add_argument('--comparison_folder', type=str, help="Comparison folder containing subfolders and leaf subfolders with 'aff_left.png' and/or 'aff_right.png'.")
    parser.add_argument('--files_folder', type=str, default=None)
    parser.add_argument('--visualize', action='store_true', help="If set, visualize 20 random examples.")
    parser.add_argument('--visualize-dir', type=str, default='./visualizations_new', help="Directory to save visualized images.")
    parser.add_argument('--caption', type=str, default="Aff-Ex")
    parser.add_argument('--num-examples', type=int, default=20)
    parser.add_argument('--only', default=None)
    parser.add_argument('--map', default=None, action='store_true')
    parser.add_argument('--cropped', default=None, action='store_true')
    parser.add_argument('--intersection', default=None, action='store_true')

    args = parser.parse_args()

    visualize = args.visualize  # Set the flag based on user input

    # Visualize if the flag is set
    if visualize:
        os.makedirs(args.visualize_dir, exist_ok=True)
        visualize_examples(args.benchmark_folder, args.comparison_folder, args.visualize_dir, caption=args.caption, n_examples=args.num_examples, files_folder=args.files_folder)
    else:
        # Print the IoU values
        visualize_examples(args.benchmark_folder, args.comparison_folder, args.visualize_dir, visualize=visualize, n_examples=float('inf'), only=args.only, calc_map=args.map, is_cropped=args.cropped, take_intersection=args.intersection)

if __name__ == '__main__':
    main()
