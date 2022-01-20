# -*- coding: utf-8 -*-
from .kor_dataset import KorAuToInferenceDataset


class KlueNLIAutoInferenceDataset(KorAuToInferenceDataset):
    task_name = 'klue_nli'

    @staticmethod
    def process_input(data):
        input1 = [f'가정: {p}' for p in data['hypothesis']]
        input2 = [f'전제: {p}' for p in data['premise']]
        return input1, input2


class KlueNLIAutoInferenceReverseDataset(KorAuToInferenceDataset):
    task_name = 'klue_nli'

    @staticmethod
    def process_input(data):
        input1 = [f'전제: {p}' for p in data['premise']]
        input2 = [f'가정: {p}' for p in data['hypothesis']]
        return input1, input2


class KlueNLIAutoInferenceSignalDataset(KorAuToInferenceDataset):
    task_name = 'klue_nli'

    @staticmethod
    def process_input(data):
        input1 = [f'[가정] {p}' for p in data['hypothesis']]
        input2 = [f'[전제] {p}' for p in data['premise']]
        return input1, input2
