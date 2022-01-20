# -*- coding: utf-8 -*-

from .glue_dataset import GLUEAuToInferenceDataset, GLUET5InferenceDataset


class QQPAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'qqp'

    @staticmethod
    def process_input(data):
        input1 = [f'Question1: {p}' for p in data['question1']]
        input2 = [f'Question2: {p}' for p in data['question2']]
        return input1, input2


class QQPAutoInferenceReverseDataset(GLUEAuToInferenceDataset):
    task_name = 'qqp'

    @staticmethod
    def process_input(data):
        input1 = [f'Question2: {p}' for p in data['question2']]
        input2 = [f'Question1: {p}' for p in data['question1']]
        return input1, input2


class QQPAutoInferenceSignalDataset(GLUEAuToInferenceDataset):
    task_name = 'qqp'

    @staticmethod
    def process_input(data):
        input1 = [f'[Question1] {p}' for p in data['question1']]
        input2 = [f'[Question2] {p}' for p in data['question2']]
        return input1, input2


class QQPT5InferenceDataset(GLUET5InferenceDataset):
    task_name = 'qqp'

    def process_input(self, data):
        inputs = [f"{self.task_name} question1: {h} question2: {p}" for h, p in zip(data['question1'], data['question2'])]
        return inputs


class QQPT5InferenceReverseDataset(GLUET5InferenceDataset):
    task_name = 'qqp'

    def process_input(self, data):
        inputs = [f"{self.task_name} question2: {p} question1: {h}" for h, p in zip(data['question1'], data['question2'])]
        return inputs


class QQPT5InferenceSignalDataset(GLUET5InferenceDataset):
    task_name = 'qqp'

    def process_input(self, data):
        inputs = [f"{self.task_name} [question1] {h} [question2] {p}" for h, p in
                  zip(data['question1'], data['question2'])]
        return inputs
