# -*- coding: utf-8 -*-
from .kor_dataset import KorAuToInferenceDataset


class KlueSTSAutoInferenceDataset(KorAuToInferenceDataset):
    task_name = 'klue_sts'

    @staticmethod
    def process_input(data):
        input1 = [f'문장1: {p}' for p in data['sentence1']]
        input2 = [f'문장2: {p}' for p in data['sentence2']]
        return input1, input2


class KlueSTSAutoInferenceReverseDataset(KorAuToInferenceDataset):
    task_name = 'klue_sts'

    @staticmethod
    def process_input(data):
        input1 = [f'문장2: {p}' for p in data['sentence2']]
        input2 = [f'문장1: {p}' for p in data['sentence1']]
        return input1, input2


class KlueSTSAutoInferenceSignalDataset(KorAuToInferenceDataset):
    task_name = 'klue_sts'

    @staticmethod
    def process_input(data):
        input1 = [f'[문장1] {p}' for p in data['sentence1']]
        input2 = [f'[문장2] {p}' for p in data['sentence2']]
        return input1, input2
