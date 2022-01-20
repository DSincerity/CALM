#!/bin/bash

NGPU=$1
DATASET=$2
MODEL_NAME=$3

cmd="python train.py  \
    task=nlp/text_classification \
    dataset=nlp/text_classification/$DATASET \
    training=glue \
    trainer=glue \
    trainer.gpus=$NGPU "

KO_MODEL_PREFIX="ko"
if [[ $MODEL_NAME == *"$KO_MODEL_PREFIX"* ]]; then
    echo "Finetuning for Korean Model(${MODEL_NAME})"
    cmd+="backbone=nlp/${MODEL_NAME}"
else
    echo "Finetuning for Enlgish Model(${MODEL_NAME})"
    cmd+="backbone.pretrained_model_name_or_path=$MODEL_NAME"
fi

echo $cmd
eval $cmd

# make directory if not exist
mkdir -p ../model_binary
mkdir -p ../model_binary/google

# copy model binaryfile to the directory we want
cp outputs/checkpoints/test.ckpt ../model_binary/$MODEL_NAME-$DATASET.ckpt
# remove outputs directory
rm -rf outputs
