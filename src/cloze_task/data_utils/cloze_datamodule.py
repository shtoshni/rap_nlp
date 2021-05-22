import torch
from transformers import GPT2TokenizerFast, LongformerTokenizerFast
from os import path
from pytorch_lightning.core.datamodule import LightningDataModule
from cloze_task.data_utils.cloze_dataset import LambadaDataset
from cloze_task.data_utils.datasampler import SmartBatchSampler, SmartSampler
from cloze_task.data_utils.data_collator import DataCollatorForClozeGeneration
from cloze_task.data_utils.constants import MENT_START, MENT_END, COREF_START, COREF_END


class ClozeTaskDataModule(LightningDataModule):
    def __init__(self,  data_dir=None, batch_size=8, max_token_limit=1000, num_workers=1,
                 train_size=1e6, val_size=500, model_size='base',
                 chain_prob=0.0, oracle=False,  chain_rep='canonical', coref_len=None,
                 include_singletons=False,
                 **kwargs):
        super().__init__()

        # Set Other eval
        self.data_dir = data_dir

        self.train_size = train_size
        self.val_size = val_size

        # Additional model settings
        self.chain_prob = chain_prob
        self.oracle = oracle
        if self.oracle:
            self.chain_prob = 1.00

        self.chain_rep = chain_rep
        self.coref_len = coref_len
        self.include_singletons = include_singletons

        self.batch_size = batch_size
        self.max_token_limit = max_token_limit
        self.num_workers = num_workers

        # Original tokenizer is used by the coreference model
        self.orig_tokenizer = LongformerTokenizerFast.from_pretrained(f'allenai/longformer-large-4096')
        # Tokenizer used by the language model
        self.tokenizer = GPT2TokenizerFast.from_pretrained(f'gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if chain_prob:
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
        counter = 0
        for _ in self.train_dataloader():
            counter += 1
        return counter

    def get_tokenizer(self) -> GPT2TokenizerFast:
        return self.tokenizer

    def _load_dataset(self, split="train"):
        if self.oracle:
            chain_prob = 1.00
        else:
            chain_prob = (self.chain_prob if split == "train" else 0.0)

        return LambadaDataset(
            tokenizer=self.orig_tokenizer, file_path=path.join(self.data_dir, f"{split}.jsonlines"),
            max_instances=(self.train_size if split == 'train' else self.val_size),
            chain_prob=chain_prob, chain_rep=self.chain_rep, coref_len=self.coref_len,
            include_singletons=self.include_singletons,
        )

    def train_dataloader(self):
        batch_sampler = SmartBatchSampler(
            SmartSampler(self.dataset_dict['train']), max_token_limit=self.max_token_limit)
        train_loader = torch.utils.data.DataLoader(
            self.dataset_dict['train'], num_workers=self.num_workers,
            collate_fn=self.train_data_collator, pin_memory=True, drop_last=False,
            batch_sampler=batch_sampler,
        )

        return train_loader

    def val_dataloader(self):
        dev_loader = torch.utils.data.DataLoader(
            self.dataset_dict['val'], batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.inference_data_collator,
            drop_last=False, pin_memory=True)

        return dev_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.dataset_dict['test'], batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=self.inference_data_collator, drop_last=False,
            pin_memory=True)

        return test_loader
