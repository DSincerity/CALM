MODEL_NAME=$1
REPEATS=$2
LANG=$3

if [ $LANG == "en" ]
then
    datasets="rte mrpc mnli qnli qqp"
elif [ $LANG == "kr" ]
then
    datasets="kornli klue_sts klue_nli"
else
    datasets=""
fi

for data in $datasets
do
    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation --input_format reverse
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation --input_format signal
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type test --input_format original
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type test --input_format reverse
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type test --input_format signal
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd
done
