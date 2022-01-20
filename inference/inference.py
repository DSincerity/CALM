# -*- coding: utf-8 -*-
import torch
import os
import sys
sys.path.append('../lightning-transformers')
import argparse
import json
import re
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5TokenizerFast, T5ForConditionalGeneration
from typing import List, Text
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from models.kobart.tokenization_kobart import KoBartTokenizerFast as tok_kobart
from models.kobert.tokenization_kobert import KoBertTokenizer as tok_kobert
from models.kogpt2.tokenization_kogpt2 import KoGPT2TokenizerFast as tok_kogpt2

AUTOMODEL_LIST = ['bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large',
                  'albert-base-v2', 'albert-large-v2', 'google/electra-small-discriminator',
                  'google/electra-large-discriminator', 'facebook/bart-base', 'monologg/koelectra-base-v2-discriminator',
                  'gpt2', 'gpt2-large', 'monologg/kobert', 'hyunwoongko/kobart', 'skt/kogpt2-base-v2']

CUSTOM_TOKENIZER = ['kobert', 'kobart', 'kogpt2']

T5MODEL_LIST = ['t5-base', 't5-large', 't5-3B', 't5-11B']

GLUE_TASK_LIST = ['rte', 'mrpc', 'mnli', 'qnli', 'qqp']
KOR_TASK_LIST = ['kornli', 'klue_nli', 'klue_sts']


class AutoModelInferencer:

    def __init__(
            self,
            model,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.model.to(self.device)

    def __call__(self, data_loader) -> List:
        outp = []
        for batch in tqdm(data_loader):
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


class T5ModelInferencer:

    def __init__(
            self,
            model,
            tokenizer
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer

    def __call__(self, data_loader) -> List:
        outp = []
        for batch in tqdm(data_loader):
            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            generated = self.model.generate(**inputs, max_length=16)
            for g_ in generated:
                o_ = self.tokenizer.decode(g_)
                outp.append(self.prepro_generated_sent(o_))
        return outp

    @staticmethod
    def prepro_generated_sent(sent: Text) -> Text:
        PREPRO_PATTERN = re.compile('<[/a-zA-Z]+>')
        return PREPRO_PATTERN.sub(repl='', string=sent).strip()


def prepare_model(args):
    n_class_dict = {
        "mnli": 3,
        "rte": 2,
        "qqp": 2,
        "qnli": 2,
        "mrpc": 2,
        "kornli": 3,
        "klue_sts": 2,
        "klue_nli": 3
    }

    print(f'model type: {args.model_type}')
    if args.model_type in AUTOMODEL_LIST:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_type,
                                                                   num_labels=n_class_dict.get(args.dataset))

        nomalized_model_type = args.model_type.split('/')[-1].split('-')[0]
        assert nomalized_model_type is not None, 'model_type error'
        if nomalized_model_type in CUSTOM_TOKENIZER:
            tok = eval(f'tok_{nomalized_model_type}')
            tokenizer = tok.from_pretrained(args.model_type)
            print(f'> load a custom tokenizer : {nomalized_model_type}')
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_type)

        if tokenizer.pad_token is None or model.config.pad_token_id is None:
            print('> A pad token is set to a eod_token')
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
    elif args.model_type in T5MODEL_LIST:
        model = T5ForConditionalGeneration.from_pretrained(args.model_type)
        tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    else:
        raise NotImplementedError
    return model, tokenizer


def prepare_data(args, tokenizer):
    if args.dataset in GLUE_TASK_LIST:
        if args.dataset == 'rte':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import RTEAutoInferenceDataset, RTEAutoInferenceReverseDataset, \
                    RTEAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return RTEAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return RTEAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return RTEAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            elif args.model_type in T5MODEL_LIST:
                from dataset import RTET5InferenceDataset, RTET5InferenceReverseDataset, RTET5InferenceSignalDataset
                if args.input_format == 'original':
                    return RTET5InferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return RTET5InferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return RTET5InferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError

        elif args.dataset == 'mrpc':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import MRPCAutoInferenceDataset, MRPCAutoInferenceReverseDataset, \
                    MRPCAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return MRPCAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return MRPCAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return MRPCAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            elif args.model_type in T5MODEL_LIST:
                from dataset import MRPCT5InferenceDataset, MRPCT5InferenceReverseDataset, \
                    MRPCT5InferenceSignalDataset
                if args.input_format == 'original':
                    return MRPCT5InferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return MRPCT5InferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return MRPCT5InferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError

        elif args.dataset == 'mnli':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import MNLIAutoInferenceDataset, MNLIAutoInferenceReverseDataset, \
                    MNLIAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return MNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return MNLIAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return MNLIAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            elif args.model_type in T5MODEL_LIST:
                from dataset import MNLIT5InferenceDataset, MNLIT5InferenceReverseDataset, \
                    MNLIT5InferenceSignalDataset
                if args.input_format == 'original':
                    return MNLIT5InferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return MNLIT5InferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return MNLIT5InferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError

        elif args.dataset == 'qnli':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import QNLIAutoInferenceDataset, QNLIAutoInferenceReverseDataset, \
                    QNLIAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return QNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return QNLIAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return QNLIAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            elif args.model_type in T5MODEL_LIST:
                from dataset import QNLIT5InferenceDataset, QNLIT5InferenceReverseDataset, \
                    QNLIT5InferenceSignalDataset
                if args.input_format == 'original':
                    return QNLIT5InferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return QNLIT5InferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return QNLIT5InferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError

        elif args.dataset == 'qqp':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import QQPAutoInferenceDataset, QQPAutoInferenceReverseDataset, \
                    QQPAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return QQPAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return QQPAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return QQPAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            elif args.model_type in T5MODEL_LIST:
                from dataset import QQPT5InferenceDataset, QQPT5InferenceReverseDataset, \
                    QQPT5InferenceSignalDataset
                if args.input_format == 'original':
                    return QQPT5InferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return QQPT5InferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return QQPT5InferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    elif args.dataset in KOR_TASK_LIST:
        if args.dataset == 'kornli':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import KorNLIAutoInferenceDataset, KorNLIAutoInferenceReverseDataset, \
                    KorNLIAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return KorNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return KorNLIAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return KorNLIAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError
        elif args.dataset == 'klue_nli':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import KlueNLIAutoInferenceDataset, KlueNLIAutoInferenceReverseDataset, \
                    KlueNLIAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return KlueNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return KlueNLIAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return KlueNLIAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError
        elif args.dataset == 'klue_sts':
            if args.model_type in AUTOMODEL_LIST:
                from dataset import KlueSTSAutoInferenceDataset, KlueSTSAutoInferenceReverseDataset, \
                    KlueSTSAutoInferenceSignalDataset
                if args.input_format == 'original':
                    return KlueSTSAutoInferenceDataset(tokenizer, data_type=args.data_type)
                elif args.input_format == 'reverse':
                    return KlueSTSAutoInferenceReverseDataset(tokenizer, data_type=args.data_type)
                else:
                    return KlueSTSAutoInferenceSignalDataset(tokenizer, data_type=args.data_type)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def load_model_from_statedict(model, args):
    """
    This function is implemented to match the keys in state dict
    """
    nomalized_model_type = args.model_type.split('/')[-1].split('-')[0]
    if torch.cuda.is_available():
        print('> load model path : ', os.path.join(args.dir_path, args.model_dir, f'{nomalized_model_type}-{args.dataset}.ckpt'))
        savefile = torch.load(os.path.join(args.dir_path, args.model_dir, f'{nomalized_model_type}-{args.dataset}.ckpt'))
    else:
        savefile = torch.load(os.path.join(args.dir_path, args.model_dir, f'{nomalized_model_type}-{args.dataset}.ckpt'),
                              map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in savefile['state_dict'].items():
        if nomalized_model_type in ['kobart', 'bart']:
            if 'classification' in key:
                new_state_dict[key.replace('model.', '')] = value # match keys
                continue
            new_state_dict[key.replace('model.model', 'model')] = value # match keys
        else:
            new_state_dict[key.replace('model.', '')] = value # match keys
    model.load_state_dict(new_state_dict)
    return model


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    args.dir_path = dir_path

    print("Loading models...")
    if args.dataset in GLUE_TASK_LIST + KOR_TASK_LIST:
        model, tokenizer = prepare_model(args)
    else:
        raise NotImplementedError

    print("Loading data...")
    dataset = prepare_data(args, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    if args.model_type in AUTOMODEL_LIST:
        model = load_model_from_statedict(model, args)
        inferencer = AutoModelInferencer(model)
    elif args.model_type in T5MODEL_LIST:
        inferencer = T5ModelInferencer(model, tokenizer)
    else:
        raise NotImplementedError
    model.eval()

    predictions = inferencer(data_loader)
    perf_dict = {}
    if args.data_type == 'validation':
        acc = accuracy_score(y_true=dataset.label, y_pred=predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=dataset.label,
            y_pred=predictions,
            average='weighted'
        )
        perf_dict['accuracy'] = acc
        perf_dict['precision'] = precision
        perf_dict['recall'] = recall
        perf_dict['f1'] = f1
        print(f"{args.model_type}|{args.dataset}| Accuracy: {acc}")

    outputs = {
        "idx": [i for i in range(len(predictions))],
        # "inputs": [s for s in dataset.input_sents],
        "preds": predictions
    }

    if perf_dict:
        outputs.update(perf_dict)

    save_path = os.path.join(dir_path, args.save_dir, args.model_type.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{args.dataset}-{args.data_type}-{args.input_format}.json"
    with open(os.path.join(save_path, file_name), 'w') as saveFile:
        json.dump(outputs, saveFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='google/electra-small-discriminator',
                        help='type or pre-trained models')
    parser.add_argument('--dataset', type=str, default='mnli',
                        help='finetuning task name')
    parser.add_argument('--data_type', type=str, default='test', choices=['test', 'validation'],
                        help='type of data for inference')
    parser.add_argument('--input_format', type=str, default='original', choices=['original', 'reverse', 'signal'],
                        help='type of input format')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='size of batch for inference')
    parser.add_argument('--save_dir', type=str, default='../result/',
                        help='directory to save results')
    parser.add_argument('--model_dir', type=str, default='../model_binary/',
                        help='directory path where binary file is saved')

    args = parser.parse_args()

    main(args)
