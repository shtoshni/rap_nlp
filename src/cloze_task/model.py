import torch
from argparse import ArgumentParser

from utils import stopwords
from pytorch_lightning.core.lightning import LightningModule
from transformers import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
from pytorch_utils.optimization import get_inverse_square_root_decay


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
            gradient_checkpointing=True, use_cache=False,
        )

        # print(self.model.config)
        # print(self.model.config.vocab_size)

        self.tokenizer = tokenizer

        self.num_training_steps = args.num_training_steps

        if self.chain_prob:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.token_mask = (torch.arange(self.model.config.vocab_size) >= 50257).float()
        print(self.token_mask)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--model_size', default='base', type=str,
                            choices=['base', 'medium', 'large'],
                            help='BART model size.')
        # Chain probability
        parser.add_argument('--chain_prob', type=float, default=0.0)
        # Chain representation
        parser.add_argument('--chain_rep', type=str, default='canonical', choices=['canonical', 'antecedent'])
        # Coref length
        parser.add_argument('--coref_len', type=int, default=None, help="Max length of coref mention.")
        # Include singletons
        parser.add_argument('--include_singletons', default=False, action="store_true",
                            help="Whether to represent singletons in text or not.")
        # (1) Oracle
        parser.add_argument('--oracle', default=False, action="store_true")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        # if self.lr_decay == 'square_root':
        scheduler = get_inverse_square_root_decay(
            optimizer, num_warmup_steps=0.1 * self.num_training_steps,
        )

        # else:
        #     scheduler = get_linear_schedule_with_warmup(
        #         optimizer, num_warmup_steps=0.1 * self.num_training_steps,
        #         num_training_steps=self.num_training_steps
        #     )

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def forward(self, batch, labels=None):
        if self.training:
            return self.model(**batch, return_dict=False)[0]
        else:
            # output_ids = self.model.generate(
            #     input_ids=batch['input_ids'],
            #     num_beams=4, max_length=batch['input_ids'].shape[1] + 1, early_stopping=True,
            #     pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            # )

            logits = self.model(input_ids=batch['input_ids']).logits
            self.token_mask = self.token_mask.to(device=logits.device)
            # print(logits)
            # print(logits.shape)

            last_token_logit = logits[0, -1, :] * (1 - self.token_mask) + self.token_mask * (-1e10)
            suffix_ids = [[torch.argmax(last_token_logit, dim=0).item()]]

            # suffix_ids = output_ids[:, batch['input_ids'].shape[1]:].tolist()
            corr = 0
            for suffix_id, output_id in zip(suffix_ids, batch['output_ids']):
                # print(suffix_id, output_id[:1])
                corr += int(suffix_id == output_id[:1])
            return {
                'pred': suffix_ids,
                'gt': batch['output_ids'],
                'corr': corr,
                'total': len(suffix_ids)
            }

    def training_step(self, batch, batch_ids):
        """
        @type batch: Dict [str, torch.IntTensor]
        """
        loss = self(batch)
        self.log('loss/step_train_loss', loss.detach())
        # print(batch["input_ids"].shape[0])
        return {'loss': loss}

    # def training_epoch_end(self, outputs):
    #     print("Training Steps:", self.current_epoch_steps)

    def validation_step(self, batch, batch_ids, split="val"):
        output = self(batch)
        return output

    def test_step(self, batch, batch_ids, split="test"):
        output = self(batch)
        return output

    def validation_epoch_end(self, outputs, split='val'):
        total_corr = sum([batch_output['corr'] for batch_output in outputs])
        total = sum([batch_output['total'] for batch_output in outputs])
        val_acc = total_corr/total
        print(total_corr, total, val_acc)
        self.log(f'{split}_acc', val_acc)
        return {'val_acc': val_acc}

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, split='test')


