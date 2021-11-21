from datasets import load_dataset
from os import path
import json
import logging
from summarization.data_prep.constants import DATASET_TO_ATTRIBUTES


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def load_coref_model(model_loc, source_loc):
    import sys
    sys.path.append(source_loc)

    from inference.model_inference import Inference

    model = Inference(model_loc)
    model.model.max_span_width = 7

    return model


def process_dataset(output_dir, dataset_name, model_loc, source_loc, max_docs=None, split=None,
                    index_start=None, index_end=None, **kwargs):
    # Load spacy and coref model
    coref_model = load_coref_model(model_loc, source_loc)

    if dataset_name == 'cnn_dailymail':
        dataset = load_dataset(dataset_name, '3.0.0')
    else:
        dataset = load_dataset(dataset_name)

    def process_split(split):
        suffix = ''
        if index_start is not None:
            suffix += f'_{index_start}'
        if index_end is not None:
            suffix += f'_{index_end}'
        if max_docs is not None:
            suffix += f'_max_{max_docs}'
        output_file = path.join(output_dir, f"{dataset_name}_{split}{suffix}.jsonlines")
        print(output_file)
        with open(output_file, 'w') as f:
            split_data = dataset[split]
            if index_start is not None and index_end is not None:
                data_idx_range = range(index_start, index_end)
            elif index_start is not None:
                data_idx_range = range(index_start, len(split_data))
            elif index_end is not None:
                data_idx_range = range(0, index_end)

            start_idx = 0 if index_start is None else index_start
            for idx in data_idx_range:
                instance = split_data[idx]
                if max_docs is not None and (idx - start_idx) >= max_docs:
                    break

                output_dict = {}

                summary = instance[DATASET_TO_ATTRIBUTES[dataset_name]['summary']]
                summary = summary.replace('\\n', ' ')
                summary = summary.replace('\n', ' ')

                document = instance[DATASET_TO_ATTRIBUTES[dataset_name]['document']]
                document = document.replace('\\n', ' ')
                document = document.replace('\n', ' ')

                output_dict['idx'] = idx
                output_dict['orig_idx'] = instance['id']

                coref_output_dict = coref_model.perform_coreference([summary, document])
                output_dict['sentences'] = coref_output_dict['tokenized_doc']['sentences']
                output_dict['part_lens'] = coref_output_dict['tokenized_doc']['part_lens']
                output_dict['coref_clusters'] = coref_output_dict['clusters']

                f.write(json.dumps(output_dict) + "\n")

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} docs")

    if split is None:
        for split in dataset.keys():
            process_split(split)
    else:
        process_split(split)
