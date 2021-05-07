import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForClozeGeneration:

    def __init__(self, tokenizer, training=False):
        self.tokenizer = tokenizer
        self.training = training

    def __call__(self, examples):
        if self.training:
            sentences = [example["input_ids"] + example["output_ids"] for example in examples]
            output_dict = self.tokenizer(sentences, padding=True, return_tensors="pt")
            labels = output_dict['input_ids'].clone().detach()
            # Remove pad tokens and start tokens from loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            output_dict['labels'] = labels
            return output_dict
        else:
            assert (len(examples) == 1)
            tokenized_dict = self.tokenizer(examples[0]["input_ids"], padding=False, add_special_tokens=True)
            # print(tokenized_dict)
            output_dict = {'input_ids': torch.tensor([tokenized_dict['input_ids'][:-1]]),  # Remove BOS tag
                           'output_ids': self.tokenizer.encode(examples[0]["output_ids"], add_special_tokens=False),
                           'input_length': len(tokenized_dict['input_ids'][:-1])}
            return output_dict



