#!/bin/bash
#SBATCH -n 1
#SBATCH --time=1:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/both_semantic_seg_patientwise" -s True -l "./results/patientwise_semantic_seg_both_fold0" -c 4 -e 1 -f 0