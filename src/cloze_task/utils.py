import os
from os import path
from transformers import BasicTokenizer


def preprocess_instance(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')

    return text


def load_data(file_path, num_instances=None):
    data = []
    counter = 0
    with open(file_path) as f:
        for line in f:
            instance = line.strip()
            instance = preprocess_instance(instance)

            text = BasicTokenizer(instance)

            if instance:
                data.append(instance)
                counter += 1

            if num_instances is not None:
                if counter == num_instances:
                    break

    print(f"Loaded {len(data)} instances")
    return data


def load_lambada_data(data_dir, num_train_docs=None, filt_train=False):
    assert (path.exists(data_dir))

    train_file = ("filt_train.txt" if filt_train else "train.txt")
    train_data = load_data(path.join(data_dir, train_file), num_instances=num_train_docs)

    dev_data = load_data(path.join(data_dir, "val.txt"))
    test_data = load_data(path.join(data_dir, "test.txt"))

    return train_data, dev_data, test_data


