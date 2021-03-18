import torch
import torch.nn as nn

from utils import stopwords
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast


class ClozeModel(nn.Module):

    def __init__(self, finetune=True, model_size='small', **kwargs):
        super(ClozeModel, self).__init__()
        model_name = 'gpt2' if model_size == 'small' else f'gpt2-{model_size}'

        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=False,
            gradient_checkpointing=(True if finetune else False)
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        if not finetune:
            for param in self.model.parameters():
                # Don't update model params
                param.requires_grad = False

    def forward(self, example, max_predictions=6):
        prefix_ids = self.tokenizer(example[0])['input_ids']
        cloze_ids = self.tokenizer(example[1])['input_ids']

        if self.training:
            all_ids = torch.unsqueeze(torch.cat([prefix_ids, cloze_ids], dim=0), dim=0)
            loss = self.model(all_ids, labels=all_ids.detach())[0]
            return loss
        else:
            enc = self.tokenizer
            line = example[0]
            line_encoded = enc.encode(line)['input_ids']
            line_encoded = torch.tensor(line_encoded)
            line_encoded = line_encoded.unsqueeze_(0)  # batch of size 1
            line_encoded_list = list(line_encoded[0].numpy())
            line_encoded = prefix_ids
            state = None

            for i in range(max_predictions):
                logits, state = self.model(line_encoded, past=state)

                #        predicted = argmax(logits[0,-1,:])

                # [[idx1, idx2, ...]]
                _, line_encoded_candidates = torch.topk(logits[:, -1, :], k=128, dim=-1)

                # determine which candidates are stopwords by decoding them and
                # comparing against NLTK stopword list

                line_encoded_candidates = line_encoded_candidates[0].tolist()
                is_stopword = []
                for s in line_encoded_candidates:
                    is_stopword.append(enc.decode([s.item()]).strip() in stopwords)

                # find first prediction which is not a stopword
                predicted = None
                for (idx, candidate) in enumerate(line_encoded_candidates):
                    if is_stopword[idx]:
                        #                print('skipping stopword ', idx)
                        continue
                    else:
                        predicted = candidate
                        break
                assert predicted is not None
                line_encoded = torch.tensor([[predicted]]).to("cuda")
                line_encoded_list.append(predicted)

            return enc.decode(line_encoded_list)



