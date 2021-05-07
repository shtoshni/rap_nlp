from os import path
import json
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def load_coref_model(model_loc, source_loc):
    import sys
    sys.path.append(source_loc)

    from inference.model_inference import Inference

    model = Inference(model_loc)
    model.model.max_span_width = 7

    return model


def load_data(input_file):
    data = []
    with open(input_file) as f:
        for line in f:
            instance = line.strip().split("\t")
            assert len(instance) == 2
            data.append(instance)
    return data


def flatten(l):
    return [item for sublist in l for item in sublist]


def process_dataset(input_dir, output_dir, dataset_name, model_loc, source_loc,  split='train', max_docs=None,
                    index_start=None, index_end=None, **kwargs):
    # Load spacy and coref model
    coref_model = load_coref_model(model_loc, source_loc)
    tokenizer = coref_model.tokenizer

    suffix = ''
    if index_start is not None:
        suffix += f'_{index_start}'
    if index_end is not None:
        suffix += f'_{index_end}'
    if max_docs is not None:
        suffix += f'_max_{max_docs}'
    output_file = path.join(output_dir, f"{dataset_name}_{split}{suffix}.jsonlines")
    input_file = path.join(input_dir, f"proc_{split}.txt")
    print(output_file)
    with open(output_file, 'w') as f:
        split_data = load_data(input_file)
        if index_start is not None and index_end is not None:
            data_idx_range = range(index_start, index_end)
        elif index_start is not None:
            data_idx_range = range(index_start, len(split_data))
        elif index_end is not None:
            data_idx_range = range(0, index_end)
        else:
            data_idx_range = range(len(split_data))

        start_idx = 0 if index_start is None else index_start
        for idx in data_idx_range:
            instance = split_data[idx]
            if max_docs is not None and (idx - start_idx) >= max_docs:
                break

            story, continuation = instance
            output_dict = {'idx': idx}
            coref_output_dict = coref_model.perform_coreference(story)
            output_dict['input'] = flatten(coref_output_dict['tokenized_doc']['sentences_indices'])
            output_dict['output'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(continuation))
            output_dict['coref_clusters'] = coref_output_dict['clusters']

            f.write(json.dumps(output_dict) + "\n")

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} docs")

