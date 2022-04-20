import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForClozeGeneration:

    def __init__(self, tokenizer, training=False):
        self.tokenizer = tokenizer
        self.training = training

    def __call__(self, examples):
        if self.training:
            sentences = [example["input_ids"] for example in examples]
            sentences = [self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
                         for sentence in sentences]
            output_dict = self.tokenizer(sentences, padding='longest', return_tensors="pt", truncation=True,
                                         max_length=1024, add_special_tokens=False)
            labels = output_dict['input_ids'].clone().detach()
            # Remove pad tokens and start tokens from loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            output_dict['labels'] = labels

            return output_dict
        else:
            # Data for next token prediction
            prefixes = [self.tokenizer.bos_token + example["input_ids"] for example in examples]
            cloze_output_dict = self.tokenizer(prefixes, padding='longest', return_tensors="pt")
            # print(tokenized_dict)
            cloze_output_dict['output_ids'] = self.tokenizer(
                [example["output_ids"] for example in examples], return_tensors="pt", padding='longest')['input_ids']

            # Data for perplexity calculation
            # Perplexity is only calculated for the last token

            # Get prefix lengths: First tokenize the prefix
            prefixes = [self.tokenizer.bos_token + example["input_ids"] for example in examples]
            prefix_lens = [len(input_ids) for input_ids in self.tokenizer(prefixes)["input_ids"]]
            prefix_lens = torch.tensor(prefix_lens, device=cloze_output_dict['output_ids'].device)

            sentences = [example["input_ids"] + " " + example["output_ids"] for example in examples]
            sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]
            perp_output_dict = self.tokenizer(sentences, padding='longest', return_tensors="pt", truncation=True,
                                              max_length=1024, add_special_tokens=False)
            labels = perp_output_dict['input_ids'].clone().detach()

            # Mask out prefix
            batch_size, max_len = labels.shape[0], labels.shape[1]
            tmp = torch.arange(max_len, device=prefix_lens.device).expand(batch_size, max_len)
            labels[tmp < prefix_lens] = -100
            perp_output_dict['labels'] = labels

            # print(perp_output_dict)
            # print(self.tokenizer(examples[0]["output_ids"]))
            # import sys
            # sys.exit()
            return {"cloze": cloze_output_dict, "perp": perp_output_dict}
