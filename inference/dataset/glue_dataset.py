# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Text


class GLUEAuToInferenceDataset(Dataset):

    task_name = 'glue'
    padding = 'max_length'
    max_length = 128
    truncation = 'longest_first'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        assert data_type in ['test', 'validation']
        data_ = load_dataset('glue', self.task_name)

        if self.task_name == 'mnli':
            if data_type == 'validation':
                # data = data_['validation_matched']
                data_1 = data_['validation_matched']
                data_2 = data_['validation_mismatched']
            else:
                # data = data_['test_matched']
                data_1 = data_['test_matched']
                data_2 = data_['test_mismatched']

            data = {}
            for key in data_1.features.keys():
                data[key] = data_1[key] + data_2[key]
        else:
            data = data_[data_type]

        input_1, input_2 = self.process_input(data)

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


class GLUET5InferenceDataset(Dataset):
    task_name = 'glue'
    padding = 'max_length'
    max_length = 128
    truncation = 'longest_first'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        assert data_type in ['test', 'validation']
        data_ = load_dataset('glue', self.task_name)

        if self.task_name == 'mnli':
            if data_type == 'validation':
                # data = data_['validation_matched']
                data_1 = data_['validation_matched']
                data_2 = data_['validation_mismatched']
            else:
                # data = data_['test_matched']
                data_1 = data_['test_matched']
                data_2 = data_['test_mismatched']

            data = {}
            for key in data_1.features.keys():
                data[key] = data_1[key] + data_2[key]
        else:
            data = data_[data_type]

        inputs = self.process_input(data)

        if data_type == 'validation':
            label_names = data_['train'].features['label'].names
            label_dict = {i: v for i, v in enumerate(label_names)}
            self.label = [label_dict[i] for i in data['label']]
        else:
            self.label = None

        self.input_encodes = tokenizer(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation
        )
        self.input_sents = [i1 for i1 in inputs]

    def __len__(self):
        return len(self.input_encodes['input_ids'])

    def __getitem__(self, item):
        outputs = {key: torch.LongTensor(value[item]) for key, value in self.input_encodes.items()}
        return outputs

    def process_input(self, data):
        raise NotImplementedError


class GLUEAuToParaInferenceDataset(Dataset):

    task_name = 'glue'
    padding = 'max_length'
    max_length = 128
    truncation = 'longest_first'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        assert data_type in ['test', 'validation']
        data_ = load_dataset('glue', self.task_name)

        if self.task_name == 'mnli':
            if data_type == 'validation':
                # data = data_['validation_matched']
                data_1 = data_['validation_matched']
                data_2 = data_['validation_mismatched']
            else:
                # data = data_['test_matched']
                data_1 = data_['test_matched']
                data_2 = data_['test_mismatched']

            data = {}
            for key in data_1.features.keys():
                data[key] = data_1[key] + data_2[key]
        else:
            data = data_[data_type]

        inputs = self.process_input(data)

        self.label = data['label']
        self.input_encodes = tokenizer(
            [self.task_name] * len(inputs),
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation
        )
        self.input_sents = [f"{self.task_name} {i}" for i in zip(inputs)]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        outputs = {key: torch.LongTensor(value[item]) for key, value in self.input_encodes.items()}
        outputs['labels'] = torch.LongTensor([self.label[item]])
        return outputs

    @staticmethod
    def process_input(data):
        raise NotImplementedError
