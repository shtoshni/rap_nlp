from datasets import load_dataset
import spacy
from os import path
import json
import sys
import logging
from summarization.constants import DATASET_TO_ATTRIBUTES, SPACY_NER_LABELS
from summarization.utils import entity_match


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def load_coref_model(model_loc, source_loc):
    import sys
    sys.path.append(source_loc)

    from inference.inference import Inference

    model = Inference(model_loc)
    model.model.max_span_width = 7

    return model


def process_dataset(output_dir, dataset_name, model_loc, source_loc, max_docs=None, split=None,
                    index_start=None, index_end=None, **kwargs):
    # Load spacy and coref model
    ner_model = spacy.load('en_core_web_trf')
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
                ner_extra = False
                coref_extra = False

                summary = instance[DATASET_TO_ATTRIBUTES[dataset_name]['summary']]
                document = instance[DATASET_TO_ATTRIBUTES[dataset_name]['document']]

                output_dict['idx'] = idx
                output_dict['orig_idx'] = instance['id']

                # NER way
                spacy_summary = ner_model(summary)
                summary_ents = [ent for ent in spacy_summary.ents if ent.label_ in SPACY_NER_LABELS]
                summary_ents_str = [str(ent) for ent in summary_ents]
                output_dict['summary_ents'] = summary_ents_str

                doc_ents = ner_model(document).ents
                doc_ents = set([str(ent) for ent in doc_ents])

                set_diff = set(summary_ents_str).difference(doc_ents)
                if len(set_diff):
                    ner_extra = True
                    output_dict['raw_ner_singletons'] = list(set_diff)

                ent_diff = []
                if set_diff:
                    ent_diff = set()
                    for ent in set_diff:
                        if not entity_match(ent, document):
                            ent_diff.add(ent)

                output_dict['spacy_doc_entities'] = list(doc_ents)

                if ent_diff:
                    output_dict['spacy_summary'] = [str(word) for word in spacy_summary.doc]
                    output_dict['spacy_singletons'] = []
                    for ent in spacy_summary.ents:
                        if (ent.label_ in SPACY_NER_LABELS) and (str(ent) in ent_diff):
                            output_dict['spacy_singletons'].append([[ent.start, ent.end], str(ent)])

                # Coref way
                coref_output_dict = coref_model.perform_coreference([summary, document])
                summary_len = coref_output_dict['tokenized_doc']['part_lens'][0]

                # Get singletons in summary
                singletons = [cluster[0] for cluster in coref_output_dict['clusters']
                              if len(cluster) == 1 and cluster[0][0][0] < summary_len]

                # Filtering Stage 1
                # Check that the mention is not a super mention of a smaller mention
                filtered_singletons = []
                for singleton in singletons:
                    singleton_start, singleton_end = singleton[0]

                    super_string = False
                    for ment_start, ment_end in coref_output_dict['mentions']:
                        if singleton_start <= ment_start and singleton_end >= ment_end:
                            # Check it's not the same mention
                            if singleton_start == ment_start and singleton_end == ment_end:
                                continue
                            else:
                                super_string = True
                                break
                    if not super_string:
                        filtered_singletons.append(singleton)

                # Filtering Stage 2
                filtered_singletons_2 = filtered_singletons
                # filtered_singletons_2 = []
                # for singleton in filtered_singletons:
                #     if singleton[1] == 'It':
                #         continue
                #     singleton_added = False
                #     words = singleton[1].split()
                #     for word in words:
                #         if word[0].isupper():
                #             filtered_singletons_2.append(singleton)
                #             singleton_added = True
                #             break
                #
                #     if (singleton[1] in summary_ents_str) and (not singleton_added):
                #         filtered_singletons_2.append(singleton)

                filtered_singletons_3 = []
                for singleton in filtered_singletons_2:
                    if not entity_match(' '.join(map(lambda x: str(x), list(ner_model(singleton[1]).doc))), document):
                        filtered_singletons_3.append(singleton)

                output_dict['coref_clusters'] = coref_output_dict['clusters']

                if len(filtered_singletons_3):
                    coref_extra = True
                    # output_dict['coref_clusters'] = coref_output_dict['clusters']
                    output_dict['coref_singletons'] = filtered_singletons_3
                    output_dict['coref_doc'] = {
                        'sentences': coref_output_dict['tokenized_doc']['sentences'],
                        'part_lens': coref_output_dict['tokenized_doc']['part_lens'],
                    }

                # if ner_extra or coref_extra:
                f.write(json.dumps(output_dict) + "\n")

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} docs")

    if split is None:
        for split in dataset.keys():
            process_split(split)
    else:
        process_split(split)







