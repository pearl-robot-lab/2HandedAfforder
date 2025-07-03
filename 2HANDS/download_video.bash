#!/bin/sh
if [ ! -d EPIC_DATA/frames/$1/$2 ]; then
    cd EPIC_DATA/frames
    mkdir $1
    cd $1
    mkdir $2
    cd ..
    if [ $3 = "100" ]; then
        wget https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/$1/rgb_frames/$2.tar
    else
        wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/$3/$1/$2.tar
        #wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_14.tar
        #wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/object_detection_images/$3/$1/$2.tar
        
    fi
    tar -xf $2.tar -C $1/$2
    rm $2.tar
fi
