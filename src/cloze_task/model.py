import torch
from argparse import ArgumentParser

from utils import stopwords
from pytorch_lightning.core.lightning import LightningModule
from transformers import BartForCausalLM
from transformers import get_linear_schedule_with_warmup
from pytorch_utils.optimization import get_inverse_square_root_decay


class ClozeModel(LightningModule):
    def __init__(self, args, tokenizer):
        super(ClozeModel, self).__init__()
        self.save_hyperparameters('args')
        self.__dict__.update(**vars(args))

        model_name = f'facebook/bart-{args.model_size}'
        self.model = BartForCausalLM.from_pretrained(
            model_name, output_hidden_states=False,
            # gradient_checkpointing=True,
            is_decoder=True, is_encoder_decoder=False,
        )
        self.tokenizer = tokenizer

        self.num_training_steps = args.num_training_steps

        if self.chain_prob:
            self.model.resize_token_embeddings(len(self.tokenizer))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--model_size', default='base', type=str,
                            choices=['base', 'large'],
                            help='BART model size.')
        # Chain probability
        parser.add_argument('--chain_prob', type=float, default=0.0)
        # (1) Oracle
        parser.add_argument('--oracle', default=False, action="store_true")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        if self.lr_decay == 'square_root':
            scheduler = get_inverse_square_root_decay(
                optimizer, num_warmup_steps=0.1 * self.num_training_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0.1 * self.num_training_steps,
                num_training_steps=self.num_training_steps
            )

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def forward(self, batch, labels=None):
        if self.training:
            return self.model(**batch, return_dict=False)[0]
        else:
            output_ids = self.model.generate(
                input_ids=batch['input_ids'],
                num_beams=1, max_length=batch['input_ids'].shape[1] + 5, early_stopping=True)[0]
            suffix_ids = output_ids[batch['input_ids'].shape[1]:].tolist()[:-1]
            print(suffix_ids, batch['output_ids'])
            return suffix_ids, tuple(suffix_ids) == tuple(batch['output_ids'])

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
        suffix_ids, corr = output
        return {f'{split}_corr': corr}

    def validation_epoch_end(self, outputs, split='val'):
        with torch.no_grad():
            val_acc = sum([batch_output[f'{split}_corr'] for batch_output in outputs])/len(outputs)
        self.log(f'{split}_acc', val_acc)
        return {'val_acc': val_acc}

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, split='test')



