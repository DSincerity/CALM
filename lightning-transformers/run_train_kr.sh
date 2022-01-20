NGPU=$1
MODEL_NAME=$2

datasets="klue_sts klue_nli kornli"

for data in $datasets
do
    cmd="bash train.sh $NGPU $data $MODEL_NAME
    "
    echo $cmd
    eval $cmd
done
