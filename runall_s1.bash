#!/bin/bash
WSD_DATA_DIR=${HOME}/demo/data/WSD_Unified_Evaluation_Datasets
WSD_TRAIN_DIR=${HOME}/demo/data/WSD_Training_Corpora/SemCor
MODELS_DIR=models/AW

mkdir -p ${MODELS_DIR}
for SD in 111 222 123; do
# 1. Simple (1sent+1sur)
python train.py --bert-model bert-base-cased --dropout 0 --optimizer bert-adam --layer 11 \
    --num-context 1 --seed ${SD} \
    --devset-path ${WSD_DATA_DIR}/semeval2007/semeval2007.data.xml \
    --devkey-path ${WSD_DATA_DIR}/semeval2007/semeval2007.gold.key.txt \
    ${WSD_TRAIN_DIR}/semcor.data.xml ${WSD_TRAIN_DIR}/semcor.gold.key.txt \
    ${MODELS_DIR}/semcor_enbert_s1_sd${SD}_model

for TESTSET in semeval2007 senseval2 senseval3 semeval2013 semeval2015; do
    bash test_postproc.sh ${TESTSET} ${MODELS_DIR}/semcor_enbert_s1_sd${SD}_model
done
# 2. LW (1sent+1sur)
python train.py --bert-model bert-base-cased --dropout 0 --optimizer bert-adam \
    --num-context 1 --seed ${SD} \
    --devset-path ${WSD_DATA_DIR}/semeval2007/semeval2007.data.xml \
    --devkey-path ${WSD_DATA_DIR}/semeval2007/semeval2007.gold.key.txt \
    ${WSD_TRAIN_DIR}/semcor.data.xml ${WSD_TRAIN_DIR}/semcor.gold.key.txt \
    ${MODELS_DIR}/semcor_lwa_enbert_s1_sd${SD}_model

for TESTSET in semeval2007 senseval2 senseval3 semeval2013 semeval2015; do
    bash test_postproc.sh ${TESTSET} ${MODELS_DIR}/semcor_lwa_enbert_s1_sd${SD}_model
done
# 3. GLU (1sent+1sur)
python train.py --bert-model bert-base-cased --dropout 0.1 --optimizer bert-adam --layer 11 \
    --num-context 1 --use-glu --residual-glu --seed ${SD} \
    --devset-path ${WSD_DATA_DIR}/semeval2007/semeval2007.data.xml \
    --devkey-path ${WSD_DATA_DIR}/semeval2007/semeval2007.gold.key.txt \
    ${WSD_TRAIN_DIR}/semcor.data.xml ${WSD_TRAIN_DIR}/semcor.gold.key.txt \
    ${MODELS_DIR}/semcor_glur_do0.1_enbert_s1_sd${SD}_model

for TESTSET in semeval2007 senseval2 senseval3 semeval2013 semeval2015; do
    bash test_postproc.sh ${TESTSET} ${MODELS_DIR}/semcor_glur_do0.1_enbert_s1_sd${SD}_model
done
# 4. GLU+LW (1sent+1sur)
python train.py --bert-model bert-base-cased --dropout 0.1 --optimizer bert-adam \
    --num-context 1 --use-glu --residual-glu --seed ${SD} \
    --devset-path ${WSD_DATA_DIR}/semeval2007/semeval2007.data.xml \
    --devkey-path ${WSD_DATA_DIR}/semeval2007/semeval2007.gold.key.txt \
    ${WSD_TRAIN_DIR}/semcor.data.xml ${WSD_TRAIN_DIR}/semcor.gold.key.txt \
    ${MODELS_DIR}/semcor_lwaglur_do0.1_enbert_s1_sd${SD}_model

for TESTSET in semeval2007 senseval2 senseval3 semeval2013 semeval2015; do
    bash test_postproc.sh ${TESTSET} ${MODELS_DIR}/semcor_lwaglur_do0.1_enbert_s1_sd${SD}_model
done

done
