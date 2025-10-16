import os
import shutil
import argparse

def remove_leading_zeros(folder_name):
    """Remove leading zeros from a folder name for comparison."""
    return folder_name.lstrip('0')

def copy_matching_files(src_dir1, src_dir2, output_dir):
    # List all subfolders in both directories
    subfolders_dir1 = os.listdir(src_dir1)
    subfolders_dir2 = os.listdir(src_dir2)

    # Remove leading zeros from subfolder names for comparison
    subfolders_dir1_no_zeros = {remove_leading_zeros(folder) for folder in subfolders_dir1}
    subfolders_dir2_no_zeros = {remove_leading_zeros(folder) for folder in subfolders_dir2}

    # Find common subfolders after removing leading zeros
    matching_subfolders = subfolders_dir1_no_zeros.intersection(subfolders_dir2_no_zeros)

    # Iterate through matching subfolders
    for subfolder in matching_subfolders:
        # Get the original subfolder name from the first and second directory
        original_subfolder_dir1 = next(folder for folder in subfolders_dir1 if remove_leading_zeros(folder) == subfolder)
        original_subfolder_dir2 = next(folder for folder in subfolders_dir2 if remove_leading_zeros(folder) == subfolder)

        # Create the output subfolder path using the original name (with leading zeros)
        output_subfolder = os.path.join(output_dir, original_subfolder_dir1)
        os.makedirs(output_subfolder, exist_ok=True)

        # Define paths for the files to copy
        aff_left_file = os.path.join(src_dir1, original_subfolder_dir1, 'aff_left.png')
        aff_right_file = os.path.join(src_dir1, original_subfolder_dir1, 'aff_right.png')
        inpainted_frame_file = os.path.join(src_dir2, original_subfolder_dir2, 'inpainted_frame.png')
        annotations_file = os.path.join(src_dir2, original_subfolder_dir2, 'annotation.json')

        # Check if aff_left.png or aff_right.png exist and copy them
        if os.path.exists(aff_left_file):
            shutil.copy(aff_left_file, output_subfolder)
        if os.path.exists(aff_right_file):
            shutil.copy(aff_right_file, output_subfolder)

        # Check if the other files exist and copy them
        if os.path.exists(inpainted_frame_file):
            shutil.copy(inpainted_frame_file, output_subfolder)
        else:
            print(f"Warning: {inpainted_frame_file} does not exist")

        if os.path.exists(annotations_file):
            shutil.copy(annotations_file, output_subfolder)
        else:
            print(f"Warning: {annotations_file} does not exist")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Copy matching subfolder files after removing leading zeros for comparison.")
    parser.add_argument('folder1', type=str, help="Path to the first folder")
    parser.add_argument('folder2', type=str, help="Path to the second folder")
    parser.add_argument('output_dir', type=str, help="Path to the output directory")

    args = parser.parse_args()

    # Run the function to copy matching files
    copy_matching_files(args.folder1, args.folder2, args.output_dir)

if __name__ == "__main__":
    main()
