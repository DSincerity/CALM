### Training Multi-task Models
- Model shares encoder but have different classifier layers
- Training config is in *config.yaml* file

#### 1. Training
- Training paraphrase-only model
```bash
bash train_para.sh roberta-base
```

- Training all-multitask modell
```bash
bash train_all.sh roberta-base
```


#### 2. Inference
- Inference paraphrase-only model
```bash
bash inference_para.sh roberta-base
```

- Inference all-multitask modell
```bash
bash inference_all.sh roberta-base 02
```