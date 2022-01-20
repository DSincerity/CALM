MODEL_NAME=$1

datasets="rte mnli qnli"

for data in $datasets
do
    python train.py --backbone_model_name $MODEL_NAME --dataset $data --sts_only
    rm -rf model_binary
done

