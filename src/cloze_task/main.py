import torch
import math
from os import path
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from collections import OrderedDict

from cloze_task.model import ClozeModel
from cloze_task.experiment import experiment


def get_model_name(args, argparser):
    arg_to_short_names = OrderedDict(
        [("train_size", "ts"), ('model_size', 'size'),
         ("lr_decay", "decay"), ("init_lr", "lr"),
         ("chain_rep", "cr"), ("coref_len", "clen"),
         ]
    )

    str_repr = ""
    for arg_name, short_name in arg_to_short_names.items():
        val = getattr(args, arg_name)
        # if val != argparser.get_default(arg_name):
        if val is not None:
            str_repr += short_name + "_" + str(val).replace('/', '-') + "_"

    str_repr = str_repr.strip('_')

    if args.include_singletons:
        str_repr += "_singletons"

    if args.denote_mentions:
        str_repr += "_ments"

    if args.chain_prob:
        str_repr += f"_cp_{int(100 * args.chain_prob)}"

    if args.oracle:
        str_repr += '_oracle'

    str_repr += f'_seed_{args.seed}'

    return f"cloze_{str_repr.strip('_')}"


def main(args, parser):
    seed_everything(args.seed)

    args.chain_prob = 1.0 if args.oracle else args.chain_prob
    # max_memory = int(math.ceil(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))
    # if max_memory >= 24:
    #     # For GPUs with memory >= 24 GB, double the token limit
    #     args.max_token_limit = args.max_token_limit * (max_memory/12)
    #     print(f"Setting max token limit to: {args.max_token_limit}")

    # if args.max_steps is None:
    #     # Changing the default value
    #     args.max_steps = 1e5

    args.save_dir = args.weights_save_path if args.weights_save_path is not None else args.base_model_dir
    args.model_name = get_model_name(args, parser)

    print(f"Model name: {args.model_name}")
    experiment(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add model args
    parser = ClozeModel.add_model_specific_args(parser)
    # Training args
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--real_batch_size', type=int, default=16)
    parser.add_argument('--max_token_limit', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--save_step_frequency', type=int, default=None, help='Save checkpoints every N steps')
    parser.add_argument('--num_save_checkpoint', type=int, default=10,
                        help='Max number of times checkpoints is to be saved')
    parser.add_argument('--slurm_id', type=str, default=None, help='Slurm ID')
    parser.add_argument('--init_lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', default="square_root", type=str, choices=["linear", "square_root"])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='/home/shtoshni/Research/rap_nlp/data/lambada/coref_data')
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=5200)
    parser.add_argument('--base_model_dir', type=str, default="../models/")
    parser.add_argument('--use_wandb', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=42)

    parser = Trainer.add_argparse_args(parser)
    main(parser.parse_args(), parser)
