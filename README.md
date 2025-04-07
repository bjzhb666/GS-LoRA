# GS-LoRA

This repo is used to conducted the code about baselines (retrain,ewc,mas,l2) for single step forgetting and continual forgetting.

This repo can alse run continual forgetting for GS-LoRA.

## Environment
Please follow the instructions in Deformable DETR
https://github.com/fundamentalvision/Deformable-DETR

## Data
We use the COCO dataset for training and validation.
Please create a folder named data and put the COCO dataset in it. The folder structure should look like this:
```
data
|-- coco
|   |-- annotations
|   |-- train2017
|   |-- val2017
```