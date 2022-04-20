import math
import re
import torch
from argparse import ArgumentParser

from utils import stopwords
from pytorch_lightning.core.lightning import LightningModule
from transformers import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
from pytorch_utils.optimization import get_inverse_square_root_decay, get_polynomial_decay_schedule_with_warmup
import json
from os import path
from data_utils.stopwords import stopwords


class ClozeModel(LightningModule):
    def __init__(self, args, tokenizer):
        super(ClozeModel, self).__init__()
        self.save_hyperparameters('args')
        self.__dict__.update(**vars(args))

        if args.model_size == 'base':
            model_name = f'gpt2'
        else:
            model_name = f'gpt2-{args.model_size}'

        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=False,
            use_cache=False,
        )
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer

        if self.ment_prob:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.token_mask = (torch.arange(self.model.config.vocab_size) >= 50257).float()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--model_size', default='base', type=str,
                            choices=['base', 'medium', 'large'],
                            help='GPT2 model size.')
        # Chain probability
        parser.add_argument('--ment_prob', type=float, default=0.0)
        # Chain representation
        parser.add_argument('--chain_rep', type=str, default='canonical', choices=['canonical', 'antecedent'])
        # Denote mentions
        parser.add_argument('--denote_mentions', default=False, action="store_true",
                            help="Whether to represent mentions in text or not.")
        parser.add_argument('--max_mention_len', type=int, default=None, help="Max length of mentions allowed.")
        # Coref length
        parser.add_argument('--coref_len', type=int, default=None, help="Max length of coref mention.")

        parser.add_argument('--reduce_redundancy', default=False, action="store_true",
                            help="Try to reduce redundancy in coref.")

        # Include singletons
        parser.add_argument('--include_singletons', default=False, action="store_true",
                            help="Whether to represent singletons in text or not.")
        # (1) Oracle
        parser.add_argument('--oracle', default=False, action="store_true")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=0.0)
        if self.lr_decay == 'square_root':
            scheduler = get_inverse_square_root_decay(
                optimizer, num_warmup_steps=0.1 * self.max_steps,
            )
        elif self.lr_decay == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0.1 * self.max_steps,
                num_training_steps=self.max_steps
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps=0.1 * self.max_steps,
                num_training_steps=self.max_steps,
                lr_init=self.init_lr, lr_end=1e-7, power=1.0, last_epoch=-1
            )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def forward(self, batch, split="train"):
        if self.training:
            return self.model(**batch, return_dict=False)[0]
        else:
            # Perplexity calculation
            perp_batch = batch["perp"]
            num_terms = torch.sum(perp_batch["input_ids"] != self.tokenizer.pad_token_id).item()

            lm_logits = self.model(input_ids=perp_batch['input_ids'], return_dict=True)['logits']
            self.token_mask = self.token_mask.to(lm_logits.device)
            lm_logits = lm_logits * (1 - self.token_mask) + self.token_mask * (-1e10)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = perp_batch['labels'][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

            # Generate last token
            # Function to ignore special added tokens
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                return list(range(1, 50256))

            cloze_batch = batch["cloze"]
            if split == "test":
                num_return_sequences = 4
                do_sample = True
            else:
                num_return_sequences = 1
                do_sample = False
            gen_output = self.model.generate(
                input_ids=cloze_batch['input_ids'],
                num_beams=4, num_return_sequences=num_return_sequences, do_sample=do_sample,
                max_length=cloze_batch['input_ids'].shape[1] + 4, early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                # bad_words_ids=self.bad_word_ids,
                return_dict_in_generate=True
            )

            suffix_ids = gen_output.sequences[:, cloze_batch['input_ids'].shape[1]:].tolist()

            gt_suffix = self.tokenizer.decode(
                cloze_batch['output_ids'][0].tolist(),
                clean_up_tokenization_spaces=True).split(" ")[0].strip()

            pred_suffix = ''
            pred_suffix_list = []
            for suffix_id in suffix_ids:
                pred_suffix = self.tokenizer.decode(suffix_id, clean_up_tokenization_spaces=True).strip().split(" ")[0]
                pred_suffix = re.sub(r'[^\w\s]', '', pred_suffix)
                # pred_suffixes = [re.sub(r'[^\w\s]', '', pred_suffix) for pred_suffix in pred_suffixes]
                pred_suffix_list.append(pred_suffix)
                if pred_suffix in stopwords or pred_suffix == '':
                    continue
                else:
                    break

            corr = int(pred_suffix == gt_suffix)

            return {
                # Cloze task output
                'pred': pred_suffix,
                'all_pred': pred_suffix_list,
                'gt': gt_suffix,
                'corr': corr,
                'total': 1,
                'prefix': self.tokenizer.decode(cloze_batch['input_ids'].tolist()[0][1:]).strip(),

                # Perplexity output
                'loss': loss,
                'num_terms': num_terms,
            }

    def training_step(self, batch, batch_ids):
        """
        @type batch: Dict [str, torch.IntTensor]
        """
        loss = self(batch)
        self.log('loss/step_train_loss', loss.detach())
        return {'loss': loss}

    def validation_step(self, batch, batch_ids, split="val"):
        output = self(batch, split=split)
        return output

    def test_step(self, batch, batch_ids, split="test"):
        # split = "val" if self.final_eval_val else "test"
        output = self(batch, split=split)
        return output

    def validation_epoch_end(self, outputs, split='val'):
        if len(outputs) >= 10:
            log_file = path.join(self.model_dirpath, f"{split}_log.jsonl")
        else:
            log_file = "/dev/null"

        with open(log_file, 'w') as f:
            total_corr = sum([batch_output['corr'] for batch_output in outputs])
            total = sum([batch_output['total'] for batch_output in outputs])
            cloze_acc = total_corr/total

            perp_num = sum([batch_output['loss'] * batch_output['num_terms'] for batch_output in outputs])
            perp_den = sum([batch_output['num_terms'] for batch_output in outputs])
            perp = perp_num/perp_den
            perp = math.exp(perp)

            if len(outputs) >= 10:
                print(total_corr, total, cloze_acc)
                print(f"Perplexity: {perp:.2f}")

                # Avoid logging validation checks
                self.log(f'{split}_acc', cloze_acc)
                self.log(f'{split}_perp', perp)

            for batch_output in outputs:
                f.write(json.dumps(batch_output) + "\n")

        print(f"Logs at: {log_file}")

    def test_epoch_end(self, outputs):
        # split = "val" if self.final_eval_val else "test"
        self.validation_epoch_end(outputs, split="test")


