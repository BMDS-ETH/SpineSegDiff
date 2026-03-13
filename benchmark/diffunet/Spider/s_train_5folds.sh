#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1

# Submit job for fold 1
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH -o fold1_train_output.txt
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 700 -c 4 -s True -l "./results/semantic_seg_t2_fold1" -f 1
EOF

# Submit job for fold 2
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH -o fold2_train_output.txt
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 700 -c 4 -s True -l "./results/semantic_seg_t2_fold2" -f 2
EOF

# Submit job for fold 3
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH -o fold3_train_output.txt
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 700 -c 4 -s True -l "./results/semantic_seg_t2_fold3" -f 3
EOF

# Submit job for fold 4
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH -o fold4_train_output.txt
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 700 -c 4 -s True -l "./results/semantic_seg_t2_fold4" -f 4
EOF

# Submit job for fold 5
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH -o fold5_train_output.txt
export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python train.py -d "./datasets/Dataset018_SPIDER" -b 4 -v 700 -c 4 -s True -l "./results/semantic_seg_t2_fold5" -f 5
EOF
