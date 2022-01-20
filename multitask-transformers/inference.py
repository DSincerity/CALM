# -*- coding: utf-8 -*-

import torch
import transformers
import os
import yaml
import json
import argparse
from transformers import AutoTokenizer
from typing import List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import MultitaskModel
from data import MultitaskClassificationDataMudule


class MultitaskInferencer:

    def __init__(
            self,
            model,
            batch_size: int = 64
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.model.to(self.device)

    def __call__(self, task_name, dataset) -> List:
        outp = []
        for start in tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[start:start+self.batch_size]

            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            logits = self.model(**inputs)['logits']
            preds = logits.argmax(dim=-1)
            preds = preds.detach().cpu().tolist()
            outp += preds
        return outp


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
        input_format=args.input_format,
        is_train=False
    )

    feature_dict = data_module()

    # load model
    model_name = args.backbone_model_name.split('/')[-1]

    if args.sts_only:
        file_name = os.path.join(args.model_dir,
                                 f'multitask-{model_name}-{args.dataset}-para100.ckpt')
    else:
        file_name = os.path.join(args.model_dir,
                                 f'multitask-{model_name}.ckpt')
    if torch.cuda.is_available():
        savefile = torch.load(os.path.join(dir_path, file_name))
    else:
        savefile = torch.load(os.path.join(dir_path, file_name), map_location=torch.device('cpu'))
    multitask_model.load_state_dict(savefile)
    multitask_model.eval()

    inferencer = MultitaskInferencer(multitask_model.taskmodels_dict[args.dataset])
    pred_dataset = feature_dict[args.dataset][args.data_type]
    predictions = inferencer(task_name=args.dataset, dataset=pred_dataset)

    perf_dict = {}
    if args.data_type == 'validation':
        acc = accuracy_score(y_true=pred_dataset['label'], y_pred=predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=pred_dataset['label'],
            y_pred=predictions,
            average='weighted'
        )
        perf_dict['accuracy'] = acc
        perf_dict['precision'] = precision
        perf_dict['recall'] = recall
        perf_dict['f1'] = f1
        print(f"{args.backbone_model_name}|{args.dataset}| Accuracy: {acc}")

    outputs = {
        "idx": [i for i in range(len(predictions))],
        "preds": predictions
    }

    if perf_dict:
        outputs.update(perf_dict)

    if args.sts_only:
        save_path = os.path.join(dir_path, args.save_dir, f"multitask-{model_name}-para100")
    else:
        save_path = os.path.join(dir_path, args.save_dir, f"multitask-{model_name}")

    os.makedirs(save_path, exist_ok=True)
    file_name = f"{args.dataset}-{args.data_type}-{args.input_format}.json"
    with open(os.path.join(save_path, file_name), 'w') as saveFile:
        json.dump(outputs, saveFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model_name', type=str, default='google/electra-large-discriminator',
                        help='type or pre-trained models')
    parser.add_argument('--dataset', type=str, default='rte',
                        help='name of the dataset')
    parser.add_argument('--sts_only', action='store_true')

    parser.add_argument('--data_type', type=str, default='validation', choices=['test', 'validation'],
                        help='type of data for inference')
    parser.add_argument('--input_format', type=str, default='original', choices=['original', 'reverse', 'signal'],
                        help='type of input format')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of batch for inference')
    parser.add_argument('--save_dir', type=str, default='../result/',
                        help='directory to save results')
    parser.add_argument('--model_dir', type=str, default='../model_binary/',
                        help='directory path where binary file is saved')

    args = parser.parse_args()

    main(args)
