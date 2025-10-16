# ActAffordance Directory Structure

This folder has been reorganized for better navigation and maintainability.

## Directory Overview

```
ActAffordance/
├── scripts/              # Python scripts organized by purpose
│   ├── data_processing/  # Data preparation and configuration scripts
│   ├── evaluation/       # Evaluation and metrics calculation scripts
│   └── utils/            # Utility scripts for image processing
├── notebooks/            # Jupyter notebooks for experiments
├── annotations/          # Annotation data and related files
├── data/                 # Main data directory
│   ├── cropped/         # Cropped image data
│   └── masks/           # Mask data
└── data_zipped/         # Compressed archives of data
    ├── cropped/
    └── masks/
```

## Scripts

### Data Processing (`scripts/data_processing/`)
- **add_affex.py** - Process and copy affordance mask files
- **configure_bench.py** - Configure benchmark datasets
- **prepare_folders.py** - Copy and organize files between folders
- **preprocess_video.py** - Extract frames from videos and create annotations

### Evaluation (`scripts/evaluation/`)
- **calculate_iou.py** - Calculate IoU and IoCM metrics between masks
- **show_lab_results.py** - Visualize and overlay masks on images

### Utilities (`scripts/utils/`)
- **create_benchmark_imgs.py** - Create benchmark images (currently empty)
- **gaussian.py** - Apply Gaussian blur and thresholding to images
- **restore_image_padding.py** - Restore padded images to original size

## Notebooks

- **extract_masks.ipynb** - Extract masks from TORAS annotations
- **prep_dataset.ipynb** - Prepare dataset with original and inpainted frames
- **prep_dataset_2.ipynb** - Create benchmark frame sets
- **prep_dataset_3.ipynb** - Shuffle and reorganize benchmark frames

## Important Notes

### Path Updates

Scripts have been updated to work from their new locations:

- **calculate_iou.py** now uses `../../data/cropped` as the default benchmark folder path
- When running scripts from their subdirectories, paths are relative to the script location
- To run scripts from the repository root, use: `python scripts/evaluation/calculate_iou.py [args]`

### Annotation Files

The `annotations/` folder contains:
- TORAS annotation JSON files
- COCO annotation files  
- Dated annotation folders with their own extract_masks notebooks

## Usage Examples

```bash
# Calculate IoU (from ActAffordance root)
python scripts/evaluation/calculate_iou.py --comparison_folder path/to/comparison

# Process video
python scripts/data_processing/preprocess_video.py video.mp4 "task description" output/

# Apply Gaussian filtering
python scripts/utils/gaussian.py --input_dir path/to/images --threshold 0.5
```

