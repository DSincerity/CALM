# coding=utf-8
from __future__ import absolute_import, division, print_function

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """\
everyones corpus : spoken
"""
_HOMEPAGE = "NONE"
_LICENSE = "NONE"

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {'konli': "NONE"}


def load_klue_sts():
    train = datasets.load_dataset('kor_nli', 'snli', split='train[:-20000]')
    validation = datasets.load_dataset('kor_nli', 'snli', split='train[-20000:-10000]')
    test = datasets.load_dataset('kor_nli', 'snli', split='train[-10000:]')

    outp = datasets.DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test
        }
    )
    return outp


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class kornli(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="kornli",
                               version=VERSION,
                               description="kornli")
    ]

    DEFAULT_CONFIG_NAME = "kornli"

    def __init__(self, **config):
        self.data = load_klue_sts()
        super(kornli, self).__init__(**config)

    def _info(self):
        features = datasets.Features({
            "premise":
            datasets.Value("string"),
            "hypothesis":
            datasets.Value("string"),
            "label":
            datasets.Value("int32")
            # These are the features of your dataset like images, labels ...
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        training = self.data['train']
        validation = self.data['validation']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "dataset": training,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "dataset": validation,
                },
            ),
        ]

    def _generate_examples(self, split, dataset):
        import pandas as pd

        idx = [i for i in range(dataset.num_rows)]
        premise = [f'전제: {p}' for p in dataset['premise']]
        hypothesis = [f'가정: {p}' for p in dataset['hypothesis']]
        labels = dataset['label']

        data_dict = {'idx': idx, 'premise': premise, 'hypothesis': hypothesis, 'label': labels}
        data = pd.DataFrame.from_dict(data_dict)
        print(f"Data point example: {data.iloc[1]}")

        for docid_, datum in data.iterrows():
            yield docid_, {
                    "premise": datum["premise"],
                    "hypothesis": datum["hypothesis"],
                    "label": int(datum['label']),
                }


if __name__ == '__main__':
    pass