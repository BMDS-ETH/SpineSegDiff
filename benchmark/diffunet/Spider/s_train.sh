#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 600 -c 4 -s True -l "./results/semantic_seg_t2_fold0" -f 0
