from os import path


stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
             'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
             'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',
             'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
             'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself',
             'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
             'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
             'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
             'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
             'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
             'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
             'was', 'here', 'than'}


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
            split_text = instance.split("\t")
            assert (len(split_text) == 2)
            prefix, last_word = split_text

            data.append((prefix, last_word))
            counter += 1

            if num_instances is not None:
                if counter == num_instances:
                    break

    print(f"Loaded {len(data)} instances")
    return data


def load_lambada_data(data_dir, num_train_docs=None, filt_train=False):
    assert (path.exists(data_dir))

    train_file = "proc_train.txt"
    train_data = load_data(path.join(data_dir, train_file), num_instances=num_train_docs)

    dev_data = load_data(path.join(data_dir, "proc_val.txt"))
    test_data = load_data(path.join(data_dir, "proc_test.txt"))

    return train_data, dev_data, test_data


