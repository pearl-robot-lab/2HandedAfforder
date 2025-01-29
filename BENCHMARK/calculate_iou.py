import os
import numpy as np
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
import random

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Args:
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
    aff_left_mask_rgb[:, :, 1:] = 0  # Zero out Green and Blue channels for Red color
    aff_right_mask_rgb[:, :, :2] = 0  # Zero out Red and Green channels for Blue color

    # Combine the inpainting with the colored masks using cv2.addWeighted
    combined = cv2.addWeighted(inpainting, alpha, aff_left_mask_rgb, beta, gamma)
    combined = cv2.addWeighted(combined, 1, aff_right_mask_rgb, beta, gamma)

    # Convert the result back to PIL format
    overlay_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

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


def visualize_examples(benchmark_folder, comparison_folder, visualize_dir, caption="Aff-Ex", visualize=True, n_examples=20):
    """
    Visualize a random selection of examples by overlaying aff_left, aff_right on inpainting.png
    and concatenating images. Also calculate IoU and IoCM for each example.
    """
    all_subfolders = os.listdir(benchmark_folder)
    random.shuffle(all_subfolders)

    total_iou = 0
    total_iocm = 0
    count = 0
    orig_shape = (855, 855)

    for subfolder in all_subfolders:
        benchmark_subfolder = os.path.join(benchmark_folder, subfolder)
        comparison_subfolder = os.path.join(comparison_folder, subfolder)

        if os.path.isdir(benchmark_subfolder) and os.path.isdir(comparison_subfolder):
            for leaf_subfolder in os.listdir(benchmark_subfolder):
                benchmark_leaf_subfolder = os.path.join(benchmark_subfolder, leaf_subfolder)
                comparison_leaf_subfolder = os.path.join(comparison_subfolder, leaf_subfolder)

                if os.path.isdir(benchmark_leaf_subfolder) and os.path.isdir(comparison_leaf_subfolder):
                    # Load the inpainting image for both benchmark and comparison
                    benchmark_inpainting_path = os.path.join(benchmark_leaf_subfolder, 'inpainting.png')
                    comparison_inpainting_path = os.path.join(benchmark_leaf_subfolder, 'inpainting.png')
                    print(cv2.imread(benchmark_inpainting_path).shape)
                    print(cv2.imread(comparison_inpainting_path).shape)
                    orig_shape = cv2.imread(benchmark_inpainting_path).shape[:2]
                    orig_shape = (orig_shape[1], orig_shape[0])
                    print("Orig Shape: ", orig_shape)

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
                            
                            comparison_aff_left = comparison_aff_left > 0  # Binary mask

                        if os.path.exists(aff_right_comparison_path):
                            comparison_aff_right = cv2.imread(aff_right_comparison_path, cv2.IMREAD_GRAYSCALE)
                            comparison_aff_right = cv2.resize(comparison_aff_right, orig_shape)
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
                            # Print IoU and IoCM for each subfolder and leaf subfolder
                            print(f"IoU for {subfolder}/{leaf_subfolder}: {iou:.4f}")
                            print(f"IoCM for {subfolder}/{leaf_subfolder}: {iocm:.4f}")
                            total_iou += iou
                            total_iocm += iocm

                            # If visualize flag is set, create the visualizations
                            if visualize:
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
                            
                            #if iou != 0 and iocm != 0:
                            #    count += 1
                            count += 1
                            if count >= n_examples:
                                return
    print(f"Total Averaged IoU: {total_iou/count}")
    print(f"Total Averaged IoCM: {total_iocm/count}")


def main():
    parser = argparse.ArgumentParser(description="Calculate IoU between corresponding leaf subfolders in benchmark and comparison folders.")
    parser.add_argument('--benchmark_folder', type=str, default='data/cropped', help="Benchmark folder containing subfolders and leaf subfolders with 'aff_left.png' and 'aff_right.png'.")
    parser.add_argument('--comparison_folder', type=str, help="Comparison folder containing subfolders and leaf subfolders with 'aff_left.png' and/or 'aff_right.png'.")
    parser.add_argument('--visualize', action='store_true', help="If set, visualize 20 random examples.")
    parser.add_argument('--visualize-dir', type=str, default='./visualizations', help="Directory to save visualized images.")
    parser.add_argument('--caption', type=str, default="Aff-Ex")
    parser.add_argument('--num-examples', type=int, default=20)

    args = parser.parse_args()

    visualize = args.visualize  # Set the flag based on user input

    # Visualize if the flag is set
    if visualize:
        os.makedirs(args.visualize_dir, exist_ok=True)
        visualize_examples(args.benchmark_folder, args.comparison_folder, args.visualize_dir, caption=args.caption, n_examples=args.num_examples)
    else:
        # Print the IoU values
        visualize_examples(args.benchmark_folder, args.comparison_folder, args.visualize_dir, visualize=visualize, n_examples=float('inf'))

if __name__ == '__main__':
    main()
