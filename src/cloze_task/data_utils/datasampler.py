from cloze_task.data_utils.cloze_dataset import LambadaDataset
import random


class SmartSampler:
    data_source: LambadaDataset

    def __init__(self, data_source):
        self.data_source = data_source
        seq_lens = self.data_source.get_seq_len()
        seq_lens = sorted(seq_lens, key=lambda x: x[0])
        chunk_size = 100
        self.chunks = [seq_lens[i: i + chunk_size] for i in range(0, len(self.data_source), chunk_size)]
        self.seq_order = []
        self.shuffle_chunks()

    def shuffle_chunks(self):
        """Shuffle intra-chunk examples and inter-chunk shuffling."""
        random.shuffle(self.chunks)
        self.seq_order = []
        for chunk in self.chunks:
            random.shuffle(chunk)
            self.seq_order.extend([(example_len, example_idx) for example_len, example_idx in chunk])

    def __iter__(self):
        return iter(self.seq_order)


class SmartBatchSampler:
    def __init__(self, sampler: SmartSampler, max_token_limit: int) -> None:
        self.sampler = sampler
        self.max_token_limit = max_token_limit

    def __iter__(self):
        batch = []
        num_tokens = 0
        for example_len, idx in self.sampler:
            # expected_rap_tokens = int(self.rap_prob * (example_len // 2))
            # instance_len = example_len + expected_rap_tokens
            instance_len = example_len
            if (num_tokens + instance_len) > self.max_token_limit:
                yield batch
                batch = [idx]
                num_tokens = instance_len
            else:
                batch.append(idx)
                num_tokens += instance_len

        if len(batch) > 0:
            yield batch
