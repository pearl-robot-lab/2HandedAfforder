# Data Generation for 2HandedAfforder
This repository extract the ground truth affordance annotation for the bimanual affordance task from the EPIC-KITCHENS dataset

## Step 1: Installation
* `git clone https://github.com/MarvinHei/2HandedAfforder_DataGen.git`
* `conda create -n "2handedafforder" python=3.8`
* `conda activate 2handedafforder`
* `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
* Others: `pip install -r requirements.txt`

## Step 2: Generate initial sparse masks
Download the VISOR-VIS annotations from https://data.bris.ac.uk/data/dataset/2v6cgv1x04ol22qp9rm9x2j6a7 and place it in `VISOR-VIS_Mod/epick_visor`.
Within `VISOR-VIS_Mod` run
```
python demo.py
```
to generate the sparse masks. These will get stored in `../EPIC_DATA/segmentations`

## Step 3: Run the pipeline
Within the main folder run
```
chmod +x pipeline.bash
./pipeline.bash PXX PXX_XX
```
Exchange XX for the video file of EPIC-KITCHENS you want to process. Each individual step will get stored within the `EPIC_DATA` folder
