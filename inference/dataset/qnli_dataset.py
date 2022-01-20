# -*- coding: utf-8 -*-

from .glue_dataset import GLUEAuToInferenceDataset, GLUET5InferenceDataset, GLUEAuToParaInferenceDataset


class QNLIAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'qnli'

    @staticmethod
    def process_input(data):
        input1 = [f'Question: {p}' for p in data['question']]
        input2 = [f'Sentence: {p}' for p in data['sentence']]
        return input1, input2


class QNLIAutoInferenceReverseDataset(GLUEAuToInferenceDataset):
    task_name = 'qnli'

    @staticmethod
    def process_input(data):
        input1 = [f'Sentence: {p}' for p in data['sentence']]
        input2 = [f'Question: {p}' for p in data['question']]
        return input1, input2


class QNLIAutoInferenceSignalDataset(GLUEAuToInferenceDataset):
    task_name = 'qnli'

    @staticmethod
    def process_input(data):
        input1 = [f'[Question] {p}' for p in data['question']]
        input2 = [f'[Sentence] {p}' for p in data['sentence']]
        return input1, input2


class QNLIT5InferenceDataset(GLUET5InferenceDataset):
    task_name = 'qnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} question: {h} sentence: {p}" for h, p in zip(data['question'], data['sentence'])]
        return inputs


class QNLIT5InferenceReverseDataset(GLUET5InferenceDataset):
    task_name = 'qnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} sentence: {p} question: {h}" for h, p in zip(data['question'], data['sentence'])]
        return inputs


class QNLIT5InferenceSignalDataset(GLUET5InferenceDataset):
    task_name = 'qnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} [question] {h} [sentence] {p}" for h, p in
                  zip(data['question'], data['sentence'])]
        return inputs

