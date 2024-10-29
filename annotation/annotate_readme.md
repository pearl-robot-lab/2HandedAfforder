## Setup coco-annotator
- Clone repo and cd into it https://github.com/jsbroks/coco-annotator
- Install docker-compose :https://docs.docker.com/compose/install/standalone/. Note: Use sudo chmod +x /usr/local/bin/docker-compose for permissions

## Run:
- Run coco-annotator docker
https://github.com/jsbroks/coco-annotator/wiki/Getting-Started#downloading
- Go to http://localhost:5000/
- Check usage: https://github.com/jsbroks/coco-annotator/wiki/Usage
- Copy over the images to the dataset folder in coco-annotator repo path
- See image annotator guide: https://github.com/jsbroks/coco-annotator/wiki/Image-Annotator
- Check keyboard shortcuts in the settings

## Export dataset
- Use the export or export COCO option
- Convert to masks:
    - pip install pycocotools
    - example script: coco_to_mask.py
    - https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset

## Exit
`docker-compose down`