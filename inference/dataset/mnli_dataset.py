# -*- coding: utf-8 -*-

from .glue_dataset import GLUEAuToInferenceDataset, GLUET5InferenceDataset


class MNLIAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'mnli'

    @staticmethod
    def process_input(data):
        input1 = [f'Hypothesis: {p}' for p in data['hypothesis']]
        input2 = [f'Premise: {p}' for p in data['premise']]
        return input1, input2


class MNLIAutoInferenceReverseDataset(GLUEAuToInferenceDataset):
    task_name = 'mnli'

    @staticmethod
    def process_input(data):
        input1 = [f'Premise: {p}' for p in data['premise']]
        input2 = [f'Hypothesis: {p}' for p in data['hypothesis']]
        return input1, input2


class MNLIAutoInferenceSignalDataset(GLUEAuToInferenceDataset):
    task_name = 'mnli'

    @staticmethod
    def process_input(data):
        input1 = [f'[Hypothesis] {p}' for p in data['hypothesis']]
        input2 = [f'[Premise] {p}' for p in data['premise']]
        return input1, input2
    
    
class MNLIT5InferenceDataset(GLUET5InferenceDataset):
    task_name = 'mnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} hypothesis: {h} premise: {p}" for h, p in zip(data['hypothesis'], data['premise'])]
        return inputs


class MNLIT5InferenceReverseDataset(GLUET5InferenceDataset):
    task_name = 'mnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} premise: {p} hypothesis: {h}" for h, p in zip(data['hypothesis'], data['premise'])]
        return inputs


class MNLIT5InferenceSignalDataset(GLUET5InferenceDataset):
    task_name = 'mnli'

    def process_input(self, data):
        inputs = [f"{self.task_name} [hypothesis] {h} [premise] {p}" for h, p in zip(data['hypothesis'], data['premise'])]
        return inputs
