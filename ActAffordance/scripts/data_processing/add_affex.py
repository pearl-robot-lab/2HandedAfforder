import os
import shutil
import argparse

def process_files(benchmark_folder, mask_differences, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check if 'mask_differences' contains 'left' and 'right' subfolders (Case 1)
    left_folder = os.path.join(mask_differences, 'left')
    right_folder = os.path.join(mask_differences, 'right')

    # If the left and right subfolders exist, it follows Case 1
    if os.path.isdir(left_folder) and os.path.isdir(right_folder):
        print("Processing in 'left' and 'right' folder structure (Case 1)")
        
        # Iterate over the files in the 'left' folder
        for img_file in os.listdir(left_folder):
            if img_file.endswith('.png'):
                base_name = img_file.split('.')[0]  # Extract the base name (no 7-digit check)
                target_subfolder = os.path.join(benchmark_folder, base_name)
                
                # Check if matching folder exists in the benchmark folder
                if os.path.isdir(target_subfolder):
                    # Create the corresponding subfolder in the output directory if not already exists
                    output_subfolder = os.path.join(output_folder, base_name)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Copy and rename the file from 'left' to 'aff_left.png'
                    source_file = os.path.join(left_folder, img_file)
                    output_file = os.path.join(output_subfolder, 'aff_left.png')

                    # Copy only if file doesn't exist
                    if not os.path.exists(output_file):
                        shutil.copy(source_file, output_file)
                        print(f"Copied {source_file} to {output_file}")
                    else:
                        print(f"File {output_file} already exists. Skipping copy.")

        # Iterate over the files in the 'right' folder
        for img_file in os.listdir(right_folder):
            if img_file.endswith('.png'):
                base_name = img_file.split('.')[0]  # Extract the base name (no 7-digit check)
                target_subfolder = os.path.join(benchmark_folder, base_name)
                
                # Check if matching folder exists in the benchmark folder
                if os.path.isdir(target_subfolder):
                    # Create the corresponding subfolder in the output directory if not already exists
                    output_subfolder = os.path.join(output_folder, base_name)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Copy and rename the file from 'right' to 'aff_right.png'
                    source_file = os.path.join(right_folder, img_file)
                    output_file = os.path.join(output_subfolder, 'aff_right.png')

                    # Copy only if file doesn't exist
                    if not os.path.exists(output_file):
                        shutil.copy(source_file, output_file)
                        print(f"Copied {source_file} to {output_file}")
                    else:
                        print(f"File {output_file} already exists. Skipping copy.")
    
    # Case 2: If there are no 'left' and 'right' folders, assume structure mirrors benchmark_folder
    else:
        print("Processing in direct folder structure (Case 2)")
        
        # Iterate over the subfolders in the benchmark folder
        for subfolder_name in os.listdir(benchmark_folder):
            benchmark_subfolder = os.path.join(benchmark_folder, subfolder_name)
            
            # Check if the subfolder exists in the benchmark folder
            if os.path.isdir(benchmark_subfolder):
                # Check if there's a matching subfolder in mask_differences
                mask_subfolder = os.path.join(mask_differences, subfolder_name)
                
                if os.path.isdir(mask_subfolder):
                    # Create the corresponding subfolder in the output directory if not already exists
                    output_subfolder = os.path.join(output_folder, subfolder_name)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Check for aff_left.png and aff_right.png in the mask_differences subfolder
                    for file_name in os.listdir(mask_subfolder):
                        if file_name == 'aff_left.png' or file_name == 'aff_right.png':
                            source_file = os.path.join(mask_subfolder, file_name)
                            target_file = os.path.join(output_subfolder, file_name)

                            # Copy only if file doesn't exist
                            if not os.path.exists(target_file):
                                shutil.copy(source_file, target_file)
                                print(f"Copied {source_file} to {target_file}")
                            else:
                                print(f"File {target_file} already exists. Skipping copy.")
                else:
                    print(f"No matching subfolder found in mask_differences for {subfolder_name}")

def main():
    parser = argparse.ArgumentParser(description="Process image files from benchmark and mask_differences folders.")
    parser.add_argument('benchmark_folder', type=str, help="Benchmark folder containing numbered subfolders.")
    parser.add_argument('mask_differences', type=str, help="Folder containing matching subfolders with 'aff_left.png' and 'aff_right.png' or 'left'/'right' subfolders.")
    parser.add_argument('output_folder', type=str, help="Output folder to store the processed files.")
    
    args = parser.parse_args()
    
    process_files(args.benchmark_folder, args.mask_differences, args.output_folder)

if __name__ == '__main__':
    main()
