#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1

sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1

export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/Dataset018_SPIDER" -s True -l "./results/semantic_seg_t1_fold0" -c 4 -e 5 -f 1
EOF


# Submit job for fold 1
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1

export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/Dataset018_SPIDER" -s True -l "./results/semantic_seg_t1_fold1" -c 4 -e 5 -f 1
EOF

# Submit job for fold 2
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1

export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/Dataset018_SPIDER" -s True -l "./results/semantic_seg_t1_fold2" -c 4 -e 5 -f 2
EOF

# Submit job for fold 3
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1

export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/Dataset018_SPIDER" -s True -l "./results/semantic_seg_t1_fold3" -c 4 -e 5 -f 3
EOF

# Submit job for fold 4
sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:1

export PYTHONPATH="/cluster/project/bmds_lab/Diff-UNet"
python test.py -d  "./datasets/Dataset018_SPIDER" -s True -l "./results/semantic_seg_t1_fold4" -c 4 -e 5 -f 4
EOF

