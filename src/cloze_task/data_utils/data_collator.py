import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForClozeGeneration:

    def __init__(self, tokenizer, training=False):
        self.tokenizer = tokenizer
        self.training = training

    def __call__(self, examples):
        if self.training:
            sentences = [example["input_ids"] + example["output_ids"] for example in examples]
            sentences = [self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
                         for sentence in sentences]
            output_dict = self.tokenizer(sentences, padding='longest', return_tensors="pt", truncation=True,
                                         max_length=1024, add_special_tokens=True)
            labels = output_dict['input_ids'].clone().detach()
            # Remove pad tokens and start tokens from loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            output_dict['labels'] = labels

            return output_dict
        else:
            # assert (len(examples) == 1)
            prefixes = [self.tokenizer.bos_token + example["input_ids"] for example in examples]
            output_dict = self.tokenizer(prefixes, padding='longest', return_tensors="pt")
            # print(tokenized_dict)
            output_dict['output_ids'] = self.tokenizer(
                [example["output_ids"] for example in examples], return_tensors="pt")['input_ids']
            # output_dict['input_length'] = output_dict['input_ids'].shape[1]
            # print(output_dict)
            return output_dict



