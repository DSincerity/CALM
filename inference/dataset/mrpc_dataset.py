# -*- coding: utf-8 -*-

from .glue_dataset import GLUEAuToInferenceDataset, GLUET5InferenceDataset


class MRPCAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'mrpc'

    @staticmethod
    def process_input(data):
        input1 = [f'Sentence1: {p}' for p in data['sentence1']]
        input2 = [f'Sentence2: {p}' for p in data['sentence2']]
        return input1, input2


class MRPCAutoInferenceReverseDataset(GLUEAuToInferenceDataset):
    task_name = 'mrpc'

    @staticmethod
    def process_input(data):
        input1 = [f'Sentence2: {p}' for p in data['sentence2']]
        input2 = [f'Sentence1: {p}' for p in data['sentence1']]
        return input1, input2


class MRPCAutoInferenceSignalDataset(GLUEAuToInferenceDataset):
    task_name = 'mrpc'

    @staticmethod
    def process_input(data):
        input1 = [f'[Sentence1] {p}' for p in data['sentence1']]
        input2 = [f'[Sentence2] {p}' for p in data['sentence2']]
        return input1, input2


class MRPCT5InferenceDataset(GLUET5InferenceDataset):
    task_name = 'mrpc'

    def process_input(self, data):
        inputs = [f"{self.task_name} sentence1: {h} sentence2: {p}" for h, p in zip(data['sentence1'], data['sentence2'])]
        return inputs


class MRPCT5InferenceReverseDataset(GLUET5InferenceDataset):
    task_name = 'mrpc'

    def process_input(self, data):
        inputs = [f"{self.task_name} sentence2: {p} sentence1: {h}" for h, p in zip(data['sentence1'], data['sentence2'])]
        return inputs


class MRPCT5InferenceSignalDataset(GLUET5InferenceDataset):
    task_name = 'mrpc'

    def process_input(self, data):
        inputs = [f"{self.task_name} [sentence1] {h} [sentence2] {p}" for h, p in
                  zip(data['sentence1'], data['sentence2'])]
        return inputs
