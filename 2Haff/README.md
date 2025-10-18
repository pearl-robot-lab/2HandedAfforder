# 2Haff - Bimanual Affordance Prediction Model

This directory contains the 2HandedAfforder model implementation for predicting task-specific affordance regions for bimanual manipulation.

## Model & Dataset Resources

### Pre-trained Model
The pre-trained 2HandedAfforder model is available on HuggingFace:
- **Model**: [sjauhri/2HAff](https://huggingface.co/sjauhri/2HAff)

### Dataset
The 2HANDS dataset is available on HuggingFace:
- **Dataset**: [sjauhri/2HANDS](https://huggingface.co/datasets/sjauhri/2HANDS)

## Environment Setup

```bash
conda create -n 2haff python=3.8
conda activate 2haff
pip install -r requirements.txt
pip install scikit-image wandb h5py
```

**Note:** `datasets` is already included in `requirements.txt`.

## Inference

### Basic Inference

To run inference on a benchmark directory:

```bash
python inference.py \
  --version sjauhri/2HAff \
  --benchmark-dir /path/to/benchmark \
  --vis_save_path ./vis_output \
  --precision bf16
```

**Parameters:**
- `--version`: HuggingFace model identifier (default: `sjauhri/2HAff`)
- `--benchmark-dir`: Directory containing benchmark examples with subdirectories that include:
  - `inpainting.png`: Input image
  - `annotation.json`: Task narration
- `--vis_save_path`: Output directory for predicted affordance masks
- `--precision`: Inference precision (`fp32`, `bf16`, or `fp16`)
- `--load_in_4bit`: Use 4-bit quantization (optional)
- `--load_in_8bit`: Use 8-bit quantization (optional)

### Example Usage

The model takes an image and a task description as input and outputs:
- Left hand affordance mask (`aff_left.png`)
- Right hand affordance mask (`aff_right.png`)
- Taxonomy classification (left, right, or both hands)

The prompt format is:
```
"Where would you interact with the object to perform action [task_description]"
```

### Robot Demo

For continuous inference (e.g., for robot applications):

```bash
python robot_demo.py \
  --version aff_weights \
  --zed2_img_path robot_demo/in \
  --vis_save_path ./robot_demo/out \
  --force_both
```

This will continuously monitor the input directory for new images and prompts, and generate affordance predictions.

## Training

To train the model on the 2HANDS dataset from HuggingFace:

```bash
python train_ds.py \
  --dataset_dir sjauhri/2HANDS \
  --version MBZUAI/LLaVA-Llama-3-8B-v1.1 \
  --epochs 10 \
  --batch_size 2
```

The training script automatically detects and loads the dataset from HuggingFace when you provide a dataset identifier (e.g., `sjauhri/2HANDS`).