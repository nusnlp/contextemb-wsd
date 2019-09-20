#!/bin/bash
DATA_DIR=${HOME}/demo/data
WSD_TESTDIR=${DATA_DIR}/WSD_Unified_Evaluation_Datasets
TESTSET=$1
WSD_TESTPF=${WSD_TESTDIR}/${TESTSET}/${TESTSET}
MODEL_DIR=$2
OUTPUT_DIR=${MODEL_DIR%_model}_${TESTSET}_output

python test.py ${WSD_TESTPF}.data.xml ${MODEL_DIR} ${OUTPUT_DIR}
python backoff_mfs.py ${OUTPUT_DIR}/result.result ${TESTSET}.mfs.txt > ${OUTPUT_DIR}/pp.result
java -cp ${WSD_TESTDIR}:. Scorer ${WSD_TESTPF}.gold.key.txt ${OUTPUT_DIR}/pp.result
