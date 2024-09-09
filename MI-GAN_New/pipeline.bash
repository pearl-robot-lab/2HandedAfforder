#!/bin/sh
cd ../scripts
python helper.py --batchify_migan_dir $2 --batchify_migan_batchsize 16 --batchify_migan_out ../MI-GAN/batched
cd ../MI-GAN
python -m scripts.demo --model-name migan-512 --model-path ./models/migan_512_places2.pt --images-dir $1 --masks-dir batched --output-dir $3 --device cuda --invert-mask
rm -r batched
