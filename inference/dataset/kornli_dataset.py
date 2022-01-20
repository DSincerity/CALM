# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from .kor_dataset import KorAuToInferenceDataset


class KorNLIAutoInferenceDataset(KorAuToInferenceDataset):
    task_name = 'kornli'

    @staticmethod
    def process_input(data):
        input1 = [f'가정: {p}' for p in data['hypothesis']]
        input2 = [f'전제: {p}' for p in data['premise']]
        return input1, input2


class KorNLIAutoInferenceReverseDataset(KorAuToInferenceDataset):
    task_name = 'kornli'

    @staticmethod
    def process_input(data):
        input1 = [f'전제: {p}' for p in data['premise']]
        input2 = [f'가정: {p}' for p in data['hypothesis']]
        return input1, input2


class KorNLIAutoInferenceSignalDataset(KorAuToInferenceDataset):
    task_name = 'kornli'

    @staticmethod
    def process_input(data):
        input1 = [f'[가정] {p}' for p in data['hypothesis']]
        input2 = [f'[전제] {p}' for p in data['premise']]
        return input1, input2
