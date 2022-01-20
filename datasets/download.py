import os
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict


def save_jsonl(list: List[Dict], filepath: str):

    def _save(f, kwarg):
        assert kwarg, "No contents"

        f.write(json.dumps(kwarg, ensure_ascii=False))
        f.write("\n")

    with open(filepath, 'w') as file:
        for data in list:
            _save(file, data)

    print(f'> Done in saving to a file ({filepath})')


if __name__ == "__main__":

    # Load a dataset and print the first example in the training set
    datasets = [('glue', 'rte'), ('glue', 'mnli'), ('glue', 'mrpc'), ('glue', 'qnli'), ('super_glue', 'copa')]

    for corpus in tqdm(datasets, desc='load and save dataset'):
        print(f'> load {corpus}')
        dataset_nm, sub_dataset_nm = corpus if len(corpus) == 2 else (corpus[0], 'None')

        dataset = load_dataset(dataset_nm, sub_dataset_nm)
        print("> exmaple of the dataset")
        print(f"> {dataset['train'][0]}")
        print(dataset)
        splited_dataset = []
        for key, values in dataset.items():
            exec(f"{key} = dataset['{key}']")
            splited_dataset.append((key, values))

        # make a directory
        dir_nm = sub_dataset_nm if sub_dataset_nm is not None else dataset_nm
        os.makedirs(f"./datasets/{dir_nm}", exist_ok=True)

        # save dataset
        for dataset_nm, dataset in splited_dataset:
            save_jsonl([dataset[idx] for idx in range(len(dataset))], f"./datasets/{dir_nm}/{dataset_nm}.jsonl")
