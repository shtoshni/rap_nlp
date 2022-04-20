import torch
from transformers import GPT2Tokenizer, LongformerTokenizer
from os import path
from pytorch_lightning.core.datamodule import LightningDataModule
from cloze_task.data_utils.cloze_dataset import LambadaDataset
from cloze_task.data_utils.datasampler import SmartBatchSampler, SmartSampler
from cloze_task.data_utils.data_collator import DataCollatorForClozeGeneration
from cloze_task.data_utils.constants import MENT_START, MENT_END, COREF_START, COREF_END


class ClozeTaskDataModule(LightningDataModule):
    def __init__(self,  data_dir=None, train_batch_size=16, batch_size=1,
                 max_token_limit=1000, num_workers=1,
                 train_size=1e6, val_size=500, model_size='base',
                 ment_prob=0.0, oracle=False,  chain_rep='canonical',
                 coref_len=None, max_mention_len=None,
                 include_singletons=False, denote_mentions=False,
                 **kwargs):
        super().__init__()

        # Set Other eval
        self.data_dir = data_dir

        self.train_size = train_size
        self.val_size = val_size

        # Additional model settings
        self.ment_prob = ment_prob
        self.oracle = oracle
        if self.oracle:
            self.ment_prob = 1.00

        self.chain_rep = chain_rep
        self.max_mention_len = max_mention_len
        self.coref_len = coref_len
        self.denote_mentions = denote_mentions
        self.include_singletons = include_singletons

        self.batch_size = batch_size
        self.train_batch_size = train_batch_size
        self.max_token_limit = max_token_limit
        self.num_workers = num_workers

        # Original tokenizer is used by the coreference model
        self.orig_tokenizer = LongformerTokenizer.from_pretrained(f'allenai/longformer-large-4096')
        # Tokenizer used by the language model
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if ment_prob:
            special_tokens = [MENT_START, MENT_END, COREF_START, COREF_END]
            self.orig_tokenizer.add_special_tokens({
                'additional_special_tokens': special_tokens,
            })
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': special_tokens,
            })

        self.train_data_collator = DataCollatorForClozeGeneration(
            tokenizer=self.tokenizer, training=True)
        self.inference_data_collator = DataCollatorForClozeGeneration(
            tokenizer=self.tokenizer, training=False)

        self.dataset_dict = {}
        for split in ["train", "val", "test"]:
            self.dataset_dict[split] = self._load_dataset(split)

    def estimate_train_batches(self):
        batch_counter = 0
        for _ in self.train_dataloader():
            batch_counter += 1
        example_counter = len(self.dataset_dict['train'].input_ids)
        return example_counter, batch_counter

    def get_tokenizer(self) -> GPT2Tokenizer:
        return self.tokenizer

    def _load_dataset(self, split="train"):
        if self.oracle:
            ment_prob = 1.00
        else:
            ment_prob = (self.ment_prob if split == "train" else 0.0)

        max_instances = None
        if split == 'train':
            max_instances = self.train_size
        elif split == 'val':
            max_instances = self.val_size

        return LambadaDataset(
            tokenizer=self.orig_tokenizer, file_path=path.join(self.data_dir, f"{split}.jsonlines"),
            max_instances=max_instances,
            ment_prob=ment_prob, chain_rep=self.chain_rep,
            max_mention_len=self.max_mention_len, coref_len=self.coref_len,
            include_singletons=self.include_singletons, split=split, denote_mentions=self.denote_mentions,
        )

    def train_dataloader(self):
        batch_sampler = SmartBatchSampler(
            SmartSampler(self.dataset_dict['train']), max_token_limit=self.max_token_limit, batch_size=self.train_batch_size)
        train_loader = torch.utils.data.DataLoader(
            self.dataset_dict['train'], num_workers=2,
            collate_fn=self.train_data_collator, pin_memory=True, drop_last=False,
            batch_sampler=batch_sampler,
        )

        return train_loader

    def val_dataloader(self):
        dev_loader = torch.utils.data.DataLoader(
            self.dataset_dict['val'], num_workers=0,
            batch_size=self.batch_size,
            shuffle=False, collate_fn=self.inference_data_collator,
            drop_last=False, pin_memory=True)

        return dev_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.dataset_dict['test'], batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.inference_data_collator, drop_last=False,
            pin_memory=True)

        return test_loader
