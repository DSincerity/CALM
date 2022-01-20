# CALM
Consistency Analysis on Language Model


### set up
```
pip install -r requirements.txt
```

### download task datasets
- If you want to download tasks we applied in this project, execute a code below.
```
python datasets/download.py
```

### Training model using lightning-transformer
- train shell script
```bash
cd lightning_transformers
bash train.sh [n_gpu] [task_name] [model_type]
```
- Tasks available: rte, mnli, qnli, qqp, mrpc, kornli, klue_nli, klue_sts
- english models available:
    - bert-base-cased, bert-large-cased
    - roberta-base, roberta-large
    - google/electra-small-discriminator, google/electra-large-discriminator
    - albert-base-v2, albert-large-v2
    - gpt2, gpt2-large
- korean models available
    - monologg/koelectra-base-v2-discriminator
    - kobert
    - kobart
    - kogpt2




### Adding new tasks
1) Implement dataset processor
- add data processor file in lightening_transformers/task/nlp
- refer mnli/mrpc/qnli/qqp/rte in the abovementioned folder

2) Add config
- add config in conf/dataset/nlp/text_classification/
- refer mnli.yaml file in the abovementioned folder

