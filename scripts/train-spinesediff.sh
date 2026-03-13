#!/bin/bash
F=0
source ./venv-spinesegdiff/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
echo "Training fold ${F}"
contrast="T1wT2w"
python src/train.py -d "./data/SPIDER_${contrast}" -b 4 -v 250 -c 4 -s true -l "./results/SpineSegDiff/${contrast}/fold_${F}" -t 1000 -f "${F}"