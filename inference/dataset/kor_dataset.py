# -*- coding: utf-8 -*-


import torch
import datasets
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Text


def load_kor_dataset(dataset: Text):
    if dataset == 'klue_sts':
        train = load_dataset('klue', 'sts', split='train')
        validation = load_dataset('klue', 'sts', split='validation[:50%]')
        test = load_dataset('klue', 'sts', split='validation[50%:]')
    elif dataset == 'klue_nli':
        train = load_dataset('klue', 'nli', split='train')
        validation = load_dataset('klue', 'nli', split='validation[:50%]')
        test = load_dataset('klue', 'nli', split='validation[50%:]')
    elif dataset == 'kornli':
        train = load_dataset('kor_nli', 'snli', split='train[:-20000]')
        validation = load_dataset('kor_nli', 'snli', split='train[-20000:-10000]')
        test = load_dataset('kor_nli', 'snli', split='train[-10000:]')
    else:
        raise NotImplementedError

    outp = datasets.DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test
        }
    )
    return outp


class KorAuToInferenceDataset(Dataset):

    task_name = 'kor'
    padding = 'max_length'
    max_length = 128
    truncation = 'longest_first'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        assert data_type in ['test', 'validation']
        data_ = load_kor_dataset(self.task_name)
        data = data_[data_type]

        input_1, input_2 = self.process_input(data)

        if self.task_name == 'klue_sts':
            self.label = [l['binary-label'] for l in data['labels']]
        else:
            self.label = data['label']
        self.input_encodes = tokenizer(
            input_1,
            input_2,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation
        )
        self.input_sents = [f"{i1} {i2}" for i1, i2 in zip(input_1, input_2)]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        outputs = {key: torch.LongTensor(value[item]) for key, value in self.input_encodes.items()}
        outputs['labels'] = torch.LongTensor([self.label[item]])
        return outputs

    @staticmethod
    def process_input(data):
        raise NotImplementedError

