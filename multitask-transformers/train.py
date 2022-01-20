# -*- coding: utf-8 -*-
import transformers
import os
import torch
import argparse
import yaml
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, EarlyStoppingCallback
from model import MultitaskModel
from data import MultitaskClassificationDataMudule
from utils import MultitaskTrainer, NLPDataCollator, DataLoaderWithTaskname


metric = load_metric("accuracy")


def save_state_dict(model, save_path: str, save_prefix: str):
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, save_prefix + '.ckpt')
    model = model.cpu()
    torch.save(model.state_dict(), filename)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, 'config.yaml'), 'r') as readFile:
        config_file = yaml.load(readFile, Loader=yaml.SafeLoader)
    cfg = config_file.get('cfg')

    n_class_dict = {
        "mnli": 3,
        "rte": 2,
        "qnli": 2
    }

    if args.sts_only:
        multitask_model = MultitaskModel.create(
            model_name=args.backbone_model_name,
            model_type_dict={
                f"{args.dataset}": transformers.AutoModelForSequenceClassification,
                "qqp": transformers.AutoModelForSequenceClassification,
                "mrpc": transformers.AutoModelForSequenceClassification,
            },
            model_config_dict={
                f"{args.dataset}": transformers.AutoConfig.from_pretrained(args.backbone_model_name,
                                                                           num_labels=n_class_dict[args.dataset]),
                "qqp": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
                "mrpc": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
            },
        )
    else:  # Multitask for all models
        multitask_model = MultitaskModel.create(
            model_name=args.backbone_model_name,
            model_type_dict={
                "mnli": transformers.AutoModelForSequenceClassification,
                "qnli": transformers.AutoModelForSequenceClassification,
                "rte": transformers.AutoModelForSequenceClassification,
                "qqp": transformers.AutoModelForSequenceClassification,
                "mrpc": transformers.AutoModelForSequenceClassification,
            },
            model_config_dict={
                "mnli": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=3),
                "qnli": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
                "rte": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
                "qqp": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
                "mrpc": transformers.AutoConfig.from_pretrained(args.backbone_model_name, num_labels=2),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_name)

    if tokenizer.pad_token is None:  # for gpt2
        tokenizer.pad_token = tokenizer.eos_token

    data_module = MultitaskClassificationDataMudule(
        args.dataset,
        tokenizer,
        cfg.get('max_length'),
        sts_only=args.sts_only,
        input_format='original'
    )
    feature_dict = data_module()

    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in feature_dict.items()
    }

    eval_dataset = {
        task_name: dataset["validation"]
        for task_name, dataset in feature_dict.items()
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=os.path.join(dir_path, cfg.get('output_dir')),
            overwrite_output_dir=True,
            learning_rate=float(cfg.get('learning_rate')),
            do_train=True,
            num_train_epochs=cfg.get('epochs'),
            per_device_train_batch_size=cfg.get('batch_size'),
            per_device_eval_batch_size=cfg.get('batch_size'),
            metric_for_best_model='accuracy',
            load_best_model_at_end=True,
            greater_is_better=True,
            evaluation_strategy=transformers.IntervalStrategy('epoch')
        ),
        data_collator=NLPDataCollator.collate_batch,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get('patience'))]
    )

    trainer.train()

    trained_model = trainer.model

    model_name = args.backbone_model_name.split('/')[-1]
    if args.sts_only:
        save_prefix = f"multitask-{model_name}-{args.dataset}-para100"
    else:
        save_prefix = f"multitask-{model_name}"

    save_state_dict(trained_model, os.path.join(dir_path, '../model_binary'), save_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='rte')
    parser.add_argument('--backbone_model_name', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--sts_only', action='store_true')

    args = parser.parse_args()
    main(args)