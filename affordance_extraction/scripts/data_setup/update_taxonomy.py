import os
import json

from argparse import ArgumentParser

def update_taxonomy(folder_path):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Check if annotations.json exists in the current directory
        if 'annotation.json' in files:
            # Define file paths for obj and aff images
            obj_right_exists = 'obj_right.png' in files
            obj_left_exists = 'obj_left.png' in files
            aff_right_exists = 'aff_right.png' in files
            aff_left_exists = 'aff_left.png' in files

            # Load annotations.json
            with open(os.path.join(root, 'annotation.json'), 'r') as json_file:
                data = json.load(json_file)

            # Modify the taxonomy field if it exists
            if 'taxonomy' in data:
                taxonomy = data['taxonomy']
                if len(taxonomy) == 3:
                    # Initialize the new 4-dimensional taxonomy array with an extra 0 at the start
                    new_taxonomy = [0] + taxonomy

                    # Only adjust the new 0th index if the old taxonomy[0] is 1
                    if taxonomy[0] == 1:
                        # Determine the condition for the new 0th index
                        if not obj_right_exists and not aff_right_exists:
                            new_taxonomy[0] = 1  # No right files exist
                            new_taxonomy[1] = 0
                        elif not obj_left_exists and not aff_left_exists:
                            new_taxonomy[1] = 1  # No left files exist
                            new_taxonomy[0] = 0

                    # Update the taxonomy field in the data
                    data['taxonomy'] = new_taxonomy

                    # Write the updated data back to annotations.json
                    with open(os.path.join(root, 'annotation.json'), 'w') as json_file:
                        json.dump(data, json_file, indent=4)
                    print(f"Changed Taxonomy of: {os.path.join(root, 'annotation.json')}")

# Provide the path to your main folder containing subfolders
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', default=None)
    args = parser.parse_args()
    vals = vars(args)
    if vals['dataset']:
        update_taxonomy(vals['dataset'])
