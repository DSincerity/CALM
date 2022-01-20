# -*- coding: utf-8 -*-

import datasets


class MultitaskClassificationDataMudule:

    def __init__(
            self,
            dataset,
            tokenizer,
            max_length: int,
            sts_only: bool = True,
            input_format: str = 'original',
            is_train: bool = True
    ):
        assert input_format in ['original', 'reverse', 'signal']

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sts_only = sts_only
        self.input_format = input_format
        self.is_train = is_train
        self.dataset = dataset

    def __call__(self, *args, **kwargs):
        load_data_dict = {
            "mnli": self.load_mnli,
            "qnli": self.load_qnli,
            "rte": self.load_rte,
            "qqp": self.load_qqp,
            "mrpc": self.load_mrpc
        }

        load_convert_dict = {
            "mnli": self.convert_to_mnli_features,
            "qnli": self.convert_to_qnli_features,
            "rte": self.convert_to_rte_features,
            "qqp": self.convert_to_qqp_features,
            "mrpc": self.convert_to_mrpc_features
        }
        if self.is_train:

            if self.sts_only:
                dataset_dict = {
                    f"{self.dataset}": load_data_dict[self.dataset](self.is_train),
                    "qqp": self.load_qqp(self.is_train, 100),
                    "mrpc": self.load_mrpc(self.is_train, 100),
                }

                convert_func_dict = {
                    f"{self.dataset}": load_convert_dict[self.dataset],
                    "qqp": self.convert_to_qqp_features,
                    "mrpc": self.convert_to_mrpc_features,
                }

            else:
                dataset_dict = {
                    "mnli": self.load_mnli(self.is_train),
                    "qnli": self.load_qnli(self.is_train),
                    "rte": self.load_rte(self.is_train),
                    "qqp": self.load_qqp(self.is_train, 100),
                    "mrpc": self.load_mrpc(self.is_train, 100),
                }

                convert_func_dict = {
                    "mnli": self.convert_to_mnli_features,
                    "qnli": self.convert_to_qnli_features,
                    "rte": self.convert_to_rte_features,
                    "qqp": self.convert_to_qqp_features,
                    "mrpc": self.convert_to_mrpc_features,
                }
        else:
            dataset_dict = {
                f"{self.dataset}": load_data_dict[self.dataset](self.is_train),
            }

            convert_func_dict = {
                f"{self.dataset}": load_convert_dict[self.dataset],
            }

        features_dict = {}
        for task_name, dataset in dataset_dict.items():
            features_dict[task_name] = {}
            for phase, phase_dataset in dataset.items():
                features_dict[task_name][phase] = phase_dataset.map(
                    convert_func_dict[task_name],
                    batched=True,
                    load_from_cache_file=False,
                )

                try:
                    columns_dict = {key: ['input_ids', 'attention_mask', 'token_type_ids', 'labels'] for key in
                                    dataset_dict}

                    features_dict[task_name][phase].set_format(
                        type="torch",
                        columns=columns_dict[task_name],
                    )
                except ValueError:
                    columns_dict = {key: ['input_ids', 'attention_mask', 'labels'] for key in
                                    dataset_dict}

                    features_dict[task_name][phase].set_format(
                        type="torch",
                        columns=columns_dict[task_name],
                    )

        return features_dict

    def convert_to_rte_features(self, example_batch):
        keys = ['sentence1', 'sentence2']
        if self.input_format == 'original':
            inputs = [[f"{keys[0].capitalize()}: {s1}", f"{keys[1].capitalize()}: {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        elif self.input_format == 'reverse':
            inputs = [[f"{keys[1].capitalize()}: {s2}", f"{keys[0].capitalize()}: {s1}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        else:
            inputs = [[f"[{keys[0].capitalize()}] {s1}", f"[{keys[1].capitalize()}] {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_mnli_features(self, example_batch):
        keys = ['hypothesis', 'premise']
        if self.input_format == 'original':
            inputs = [[f"{keys[0].capitalize()}: {s1}", f"{keys[1].capitalize()}: {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        elif self.input_format == 'reverse':
            inputs = [[f"{keys[1].capitalize()}: {s2}", f"{keys[0].capitalize()}: {s1}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        else:
            inputs = [[f"[{keys[0].capitalize()}] {s1}", f"[{keys[1].capitalize()}] {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_qnli_features(self, example_batch):
        keys = ['question', 'sentence']
        if self.input_format == 'original':
            inputs = [[f"{keys[0].capitalize()}: {s1}", f"{keys[1].capitalize()}: {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        elif self.input_format == 'reverse':
            inputs = [[f"{keys[1].capitalize()}: {s2}", f"{keys[0].capitalize()}: {s1}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        else:
            inputs = [[f"[{keys[0].capitalize()}] {s1}", f"[{keys[1].capitalize()}] {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_qqp_features(self, example_batch):
        keys = ['question1', 'question2']
        if self.input_format == 'original':
            inputs = [[f"{keys[0].capitalize()}: {s1}", f"{keys[1].capitalize()}: {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        elif self.input_format == 'reverse':
            inputs = [[f"{keys[1].capitalize()}: {s2}", f"{keys[0].capitalize()}: {s1}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        else:
            inputs = [[f"[{keys[0].capitalize()}] {s1}", f"[{keys[1].capitalize()}] {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_mrpc_features(self, example_batch):
        keys = ['sentence1', 'sentence2']
        if self.input_format == 'original':
            inputs = [[f"{keys[0].capitalize()}: {s1}", f"{keys[1].capitalize()}: {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        elif self.input_format == 'reverse':
            inputs = [[f"{keys[1].capitalize()}: {s2}", f"{keys[0].capitalize()}: {s1}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]
        else:
            inputs = [[f"[{keys[0].capitalize()}] {s1}", f"[{keys[1].capitalize()}] {s2}"] for s1, s2 in
                      zip(example_batch[keys[0]], example_batch[keys[1]])]

        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    @staticmethod
    def load_mnli(is_train):
        def merge_dataset(match, mismatch):
            merged = {}
            for key, value in match.features.items():
                merged[key] = match[key] + mismatch[key]

            new_dataset = datasets.Dataset.from_dict(merged)
            for key, value in match.features.items():
                new_dataset.features[key] = value
            return new_dataset

        data = datasets.load_dataset('glue', 'mnli')
        validation = merge_dataset(data['validation_matched'], data['validation_mismatched'])
        test = merge_dataset(data['test_matched'], data['test_mismatched'])

        outp = datasets.DatasetDict(
            {
                "validation": validation,
                "test": test
            }
        )

        if is_train:
            train = data['train']
            outp['train'] = train

        return outp

    @staticmethod
    def load_rte(is_train):

        validation = datasets.load_dataset('glue', 'rte', split='validation')
        test = datasets.load_dataset('glue', 'rte', split='test')

        outp = datasets.DatasetDict(
            {
                "validation": validation,
                "test": test
            }
        )

        if is_train:
            train = datasets.load_dataset('glue', 'rte', split='train')
            outp['train'] = train

        return outp

    @staticmethod
    def load_qnli(is_train):

        validation = datasets.load_dataset('glue', 'qnli', split='validation')
        test = datasets.load_dataset('glue', 'qnli', split='test')

        outp = datasets.DatasetDict(
            {
                "validation": validation,
                "test": test
            }
        )

        if is_train:
            train = datasets.load_dataset('glue', 'qnli', split='train')
            outp['train'] = train
        return outp

    @staticmethod
    def load_qqp(is_train, ratio: float = 100):
        assert ratio <= 100
        assert ratio >= 0

        # validation = datasets.load_dataset('glue', 'qqp', split=f'validation[:{ratio}%]')
        # test = datasets.load_dataset('glue', 'qqp', split=f'test[:{ratio}%]')
        validation = datasets.load_dataset('glue', 'qqp', split=f'validation')
        test = datasets.load_dataset('glue', 'qqp', split=f'test')

        outp = datasets.DatasetDict(
            {
                "validation": validation,
                "test": test
            }
        )

        if is_train:
            train = datasets.load_dataset('glue', 'qqp', split=f'train[:{ratio}%]')
            outp['train'] = train

        return outp

    @staticmethod
    def load_mrpc(is_train, ratio: float = 100):
        assert ratio <= 100
        assert ratio >= 0

        # validation = datasets.load_dataset('glue', 'mrpc', split=f'validation[:{ratio}%]')
        # test = datasets.load_dataset('glue', 'mrpc', split=f'test[:{ratio}%]')
        validation = datasets.load_dataset('glue', 'mrpc', split=f'validation')
        test = datasets.load_dataset('glue', 'mrpc', split=f'test')

        outp = datasets.DatasetDict(
            {
                "validation": validation,
                "test": test
            }
        )

        if is_train:
            train = datasets.load_dataset('glue', 'mrpc', split=f'train[:{ratio}%]')
            outp['train'] = train
        return outp
