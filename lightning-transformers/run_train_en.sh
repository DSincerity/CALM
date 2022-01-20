NGPU=$1
MODEL_NAME=$2

datasets="rte mrpc mnli qnli qqp"

for data in $datasets
do
    cmd="bash train.sh $NGPU $data $MODEL_NAME
    "
    echo $cmd
    eval $cmd
done
