# 2HANDS Setup

## Setup Files

1. Clone the repository:
   ```bash
   git clone https://github.com/MarvinHei/2HandedAfforder_DataGen.git
   ```

2. Create required directories:
   ```bash
   mkdir -p dataset/EPIC_DATA/frames dataset/EPIC_DATA/segmentations
   ```

3. Download EPIC VISOR annotations:
   - Download from: https://data.bris.ac.uk/data/dataset/2v6cgv1x04ol22qp9rm9x2j6a7
   - Place the zip file in `VISOR-VIS_Mod/`
   - Unzip and rename the folder to `epick_visor`

4. Download EPIC-KITCHENS CSV files and place them in `dataset/EPIC_DATA/`:
   - `EPIC_100_noun_classes.csv`
   - `EPIC_100_train.csv`
   - `EPIC_100_validation.csv`
   - `EPIC_100_verb_classes.csv`

## Setup Environment

```bash
conda create -n 2handedafforder python=3.8
conda activate 2handedafforder
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
cd modules/agent_inpainting
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e .
pip install hydra-core==1.2.0 iopath
```

## Setup Data

```bash
cd modules/VISOR-VIS_Mod
python demo.py
python extract_bimanual_information.py \
  --json_directory /path/to/VISOR-VIS_Mod/epick_visor/GroundTruth-SparseAnnotations/annotations/train \
  --out /path/to/dataset/EPIC_DATA/bimanual_annotations
cd ../../scripts/utils
python add_narrations.py \
  --json_directory ../../../dataset/EPIC_DATA/bimanual_annotations_json/ \
  --narration_file ../../../dataset/EPIC_DATA/EPIC_100_train.csv
```

**Note:** Replace `/path/to/` with your actual absolute paths.

## Run Pipeline

Process EPIC-KITCHENS videos through the data generation pipeline:

```bash
bash pipeline.bash PXX PXX_XX <set>
```

**Parameters:**
- `PXX`: Participant ID (e.g., P01, P02)
- `PXX_XX`: Video ID (e.g., P01_01)
- `<set>`: Dataset split (train/validation)

