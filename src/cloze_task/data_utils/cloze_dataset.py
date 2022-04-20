import logging
import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import json
from collections import defaultdict
import random
from cloze_task.data_utils.constants import MENT_END, MENT_START, COREF_START, COREF_END

logger = logging.getLogger(__name__)


class LambadaDataset(Dataset):

    def __init__(self, tokenizer, file_path, max_instances=None, ment_prob=0.0, chain_rep='canonical',
                 max_mention_len=None, coref_len=None, denote_mentions=False, include_singletons=False,
                 split="train"):
        # print(file_path)
        assert os.path.isfile(file_path)
        self.ment_prob = ment_prob
        self.chain_rep = chain_rep

        self.max_mention_len = max_mention_len
        self.coref_len = coref_len

        self.denote_mentions = denote_mentions
        self.include_singletons = include_singletons
        self.tokenizer = tokenizer
        self.split = split

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            if max_instances:
                random.shuffle(self.lines)
                self.lines = self.lines[:max_instances]

        batch_data = self.process_data(self.lines)
        self.input_ids = batch_data["input_ids"]
        self.coref_clusters = batch_data["coref_clusters"]
        self.output_ids = batch_data["output_ids"]

    # @staticmethod
    def process_data(self, lines):
        batch_data = {"input_ids": [], "coref_clusters": [], "output_ids": []}
        for line in lines:
            instance = json.loads(line.strip())
            batch_data["input_ids"].append(instance["input"])
            if self.include_singletons:
                batch_data["coref_clusters"].append(instance["coref_clusters"])
            else:
                batch_data["coref_clusters"].append(
                    [cluster for cluster in instance["coref_clusters"] if len(cluster) > 1])

            if self.split != "train":
                batch_data["output_ids"].append(instance["output"])
        return batch_data

    def __len__(self):
        return len(self.input_ids)

    def get_seq_len(self):
        return [(len(example), idx) for idx, example in enumerate(self.input_ids)]

    def __getitem__(self, i):
        output_dict = {}

        if self.ment_prob:
            input_ids = self._process_instance(self.input_ids[i], self.coref_clusters[i])
        else:
            input_ids = self.input_ids[i]

        output_dict['input_ids'] = self.tokenizer.decode(input_ids)
        if self.split != "train":
            output_dict['output_ids'] = self.tokenizer.decode(self.output_ids[i])

        # print(output_dict)

        return output_dict

    def _process_instance(self, input_ids, coref_clusters):

        clusters_picked = coref_clusters
        if not self.include_singletons:
            clusters_picked = [cluster for cluster in coref_clusters if len(cluster) > 1]

        all_mentions = []
        for idx, cluster in enumerate(clusters_picked):
            for (ment_start, ment_end), _ in cluster:
                if self.max_mention_len is None or (ment_end - ment_start + 1) <= self.max_mention_len:
                    all_mentions.append(((ment_start, ment_end), idx, random.random() < self.ment_prob))

        token_start_to_cluster_idx = defaultdict(list)
        token_end_to_cluster_idx = defaultdict(list)

        if all_mentions:
            # Sort mentions by their positions in the document
            all_mentions = sorted(all_mentions, key=lambda x: x[0][0] - 1e-5 * x[0][1])

            for (ment_start, ment_end), cluster_idx, chosen in all_mentions:
                token_start_to_cluster_idx[ment_start].append((cluster_idx, chosen))
                token_end_to_cluster_idx[ment_end].append((cluster_idx, ment_start, chosen))

        clusters_seen = dict()
        mod_input_ids = []
        for token_idx, input_id in enumerate(input_ids):
            if token_idx in token_start_to_cluster_idx:
                for cluster_idx, chosen in token_start_to_cluster_idx[token_idx]:
                    if (self.denote_mentions and cluster_idx in clusters_seen) or self.include_singletons:
                        # Head of the chain and singletons are represented in the same way
                        if chosen:
                            mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(MENT_START))

            mod_input_ids.append(input_id)
            if token_idx in token_end_to_cluster_idx:
                for cluster_idx, ment_start, chosen in token_end_to_cluster_idx[token_idx]:
                    if (self.denote_mentions and cluster_idx in clusters_seen) or self.include_singletons:
                        # Head of the chain and singletons are represented in the same way
                        if chosen:
                            mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(MENT_END))
                    if cluster_idx in clusters_seen:
                        if chosen:
                            mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(COREF_START))
                            if self.coref_len is None:
                                mod_input_ids.extend(clusters_seen[cluster_idx])
                            else:
                                mod_input_ids.extend(clusters_seen[cluster_idx][:self.coref_len])

                            mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(COREF_END))
                        if self.chain_rep == 'antecedent':
                            # Update cluster representation
                            clusters_seen[cluster_idx] = input_ids[ment_start: token_idx + 1]
                    else:
                        clusters_seen[cluster_idx] = input_ids[ment_start: token_idx + 1]

        # if random.random() < 0.1:
        #     print(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(mod_input_ids)))
        return mod_input_ids

