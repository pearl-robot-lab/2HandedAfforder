import os
import h5py

def rename_h5_files_with_indices(folder_path):
    # Initialize the cumulative length tracker
    cumulative_length = 0
    
    # List all h5 files in the directory and sort them for sequential processing
    h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])

    for filename in h5_files:
        file_path = os.path.join(folder_path, filename)

        # Open the h5 file and calculate the length of 'obj_mask_left' under the 'data' group
        with h5py.File(file_path, 'r') as h5_file:
            length = len(h5_file['data']['inpainted'])
        json_path = os.path.join(folder_path, '..', 'jsons', os.path.splitext(os.path.basename(filename))[0] + '.json')
        # Define the new filename with cumulative index range
        new_filename = f"{cumulative_length}-{cumulative_length + length - 1}_{filename}"
        new_file_path = os.path.join(folder_path, new_filename)
        new_json_path = os.path.join(folder_path, '..', 'jsons', os.path.splitext(os.path.basename(new_filename))[0] + '.json')
        
        # Rename the file
        os.rename(file_path, new_file_path)
        os.rename(json_path, new_json_path)
        
        # Update cumulative length for the next file
        cumulative_length += length
        
        print(f"Renamed {filename} to {new_filename}")

    print("\nAll files renamed successfully with cumulative index ranges.")

import os

def rename_files_in_folder_back(folder_path):
    """
    Renames all files in the specified folder by removing all characters 
    up to and including the first occurrence of '_'.

    Args:
        folder_path (str): Path to the folder containing files to rename.
    """
    try:
        # Verify the folder exists
        if not os.path.isdir(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return

        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            # Build full path
            full_path = os.path.join(folder_path, filename)

            # Ensure it's a file (not a folder)
            if os.path.isfile(full_path):
                # Find the first occurrence of '_' and rename accordingly
                if '_' in filename and filename[0] != 'f' and filename[0] != 'P':
                    new_name = filename.split('_', 1)[-1]  # Keep everything after the first '_'
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(full_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                else:
                    print(f"No '_' found in: {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        # Skip directories
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue
        
        # Find the first occurrence of 'P'
        new_name_index = filename.find('P')
        
        # Only proceed if 'P' is found in the filename
        if new_name_index != -1:
            # Create the new filename by slicing from 'P' onward
            new_filename = filename[new_name_index:]
            
            # Rename the file
            os.rename(
                os.path.join(folder_path, filename),
                os.path.join(folder_path, new_filename)
            )
            print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage
folder_path = '../EPIC_DATA/hdf5_sets/datasets/dset'

# Example usage:
rename_h5_files_with_indices('../EPIC_DATA/hdf5_sets/datasets/dset_updated/jsons')
