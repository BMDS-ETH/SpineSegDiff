#!/bin/bash
#source ./venv-spinesegdiff/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
contrast="T1wT2w"
DATASET="SPIDER_${contrast}"

TS=10
WEIGTHS_PATH="./models/LumbarSpineSegDiff/${DATASET}/fold-${F}"
RESULTS_PATH="./results/LumbarSpineSegDiff/S15-Ts${TS}/${DATASET}/fold-${F}"
python src/test.py -d "./data/${DATASET}" -c 4  -f "${F}" -e 15 -ts ${TS} -w ${WEIGTHS_PATH} -sp "$RESULTS_PATH/outputs"

