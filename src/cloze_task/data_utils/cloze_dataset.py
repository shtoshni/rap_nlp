import logging
import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import json
from collections import defaultdict
from cloze_task.data_utils.constants import MENT_END, MENT_START, COREF

logger = logging.getLogger(__name__)


class LambadaDataset(Dataset):

    def __init__(self, tokenizer, file_path, max_instances=None, chain_prob=0.0):
        # print(file_path)
        assert os.path.isfile(file_path)
        self.chain_prob = chain_prob
        self.tokenizer = tokenizer
        # self.special_token_to_id = {}
        # for token in [MENT_END, MENT_START, COREF]:
        #     self.special_token_to_id[token] = self.tokenizer.convert_tokens_to_ids(token)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            if max_instances:
                self.lines = self.lines[:max_instances]

        batch_data = self.process_data(self.lines)
        self.input_ids = batch_data["input_ids"]
        self.coref_clusters = batch_data["coref_clusters"]
        self.output_ids = batch_data["output_ids"]

    @staticmethod
    def process_data(lines):
        batch_data = {"input_ids": [], "coref_clusters": [], "output_ids": []}
        for line in lines:
            instance = json.loads(line.strip())
            batch_data["input_ids"].append(instance["input"])
            batch_data["coref_clusters"].append(instance["coref_clusters"])
            batch_data["output_ids"].append(instance["output"])
        return batch_data

    def __len__(self):
        return len(self.input_ids)

    def get_seq_len(self):
        return [(len(example), idx) for idx, example in enumerate(self.input_ids)]

    def __getitem__(self, i):
        output_dict = {}

        if self.chain_prob:
            input_ids = self._process_instance(self.input_ids[i], self.coref_clusters[i])
        else:
            input_ids = self.input_ids[i]

        output_dict['input_ids'] = self.tokenizer.decode(input_ids)
        output_dict['output_ids'] = self.tokenizer.decode(self.output_ids[i])

        return output_dict

    def _process_instance(self, input_ids, coref_clusters):
        use_coref_chain_list = np.random.choice([0, 1], size=(len(coref_clusters),),
                                                p=[1 - self.chain_prob, self.chain_prob])
        clusters_picked = [cluster for idx, cluster in enumerate(coref_clusters) if use_coref_chain_list[idx]]

        mentions_chosen = []
        for idx, cluster in enumerate(clusters_picked):
            for mention, _ in cluster:
                mentions_chosen.append((mention, idx))

        token_start_to_cluster_idx = defaultdict(list)
        token_end_to_cluster_idx = defaultdict(list)

        if mentions_chosen:
            # Sort mentions by their positions in the document
            mentions_chosen = sorted(mentions_chosen, key=lambda x: x[0][0] - 1e-5 * x[0][1])

            for (ment_start, ment_end), cluster_idx in mentions_chosen:
                token_start_to_cluster_idx[ment_start].append(cluster_idx)
                token_end_to_cluster_idx[ment_end].append((cluster_idx, ment_start))

        clusters_seen = dict()
        mod_input_ids = []
        for token_idx, input_id in input_ids:
            if token_idx in token_start_to_cluster_idx:
                for _ in token_start_to_cluster_idx[token_idx]:
                    mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(MENT_START))

            mod_input_ids.append(input_id)
            if token_idx in token_end_to_cluster_idx:
                for cluster_idx, ment_start in token_end_to_cluster_idx[token_idx]:
                    mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(MENT_END))
                    if cluster_idx in clusters_seen:
                        mod_input_ids.append(self.tokenizer.convert_tokens_to_ids(COREF))
                        mod_input_ids.extend(clusters_seen[cluster_idx])
                        # If using antecedent update the cluster representation here
                    else:
                        clusters_seen[cluster_idx] = input_ids[ment_start: token_idx + 1]

        return mod_input_ids

