MODEL_NAME=$1
REPEATS=$2

datasets="rte mnli qnli"

for data in $datasets
do
    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type validation --input_format original
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd

    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type validation --input_format reverse
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd

    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type validation --input_format signal
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd

    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type test --input_format original
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd

    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type test --input_format reverse
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd

    cmd="python inference.py --sts_only --backbone_model_name $MODEL_NAME --data_type test --input_format signal
    --save_dir ../result/$REPEATS --dataset $data"
    echo $cmd
    eval $cmd
done


