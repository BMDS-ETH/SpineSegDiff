#!/bin/bash

python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./config/train/nnUNet/input_SPIDERT1wT2w.yaml" \
    --trainer_class_name nnUNetTrainer_250epochs\
    --config "2d" \
    --fold 4\
