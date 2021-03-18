import argparse
import os
from os import path
import hashlib
import logging
import subprocess
from collections import OrderedDict

from cloze_task.experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-data_dir', default='/home/shtoshni/Research/rap_nlp/data/lambada/final_data',
        help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir',
                        default='/home/shtoshni/Research/rap_nlp/models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument(
        '-dataset', default='lambada', choices=['lambada'], type=str)

    parser.add_argument('-model_size', default='small', type=str,
                        help='GPT2 model type')

    # Training params
    parser.add_argument('-filt_train', default=False, action="store_true",
                        help='Filtered Training Set.')
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-max_stuck_epochs',
                        help='Maximum number of epochs', default=1, type=int)
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=5, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-no_finetune', dest='finetune', help="Finetune model",
                        default=True, action="store_true")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size',
                # Training params
                'max_epochs', 'seed', 'init_lr', 'finetune', 'num_train_docs',
                ]

    model_name = f"cloze_{args.dataset}_"
    arg_dict = vars(args)
    arg_val_list = sorted(list(arg_dict.items()))
    for key, val in arg_val_list:
        if key in imp_opts:
            if val is not None:
                if isinstance(val, bool):
                    if val:
                        model_name += f"{key}_"
                else:
                    model_name += f"{key}_{val}_"

    model_name = model_name.strip('_')
    logging.info(f"Model name: {model_name}")

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = path.join(model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    Experiment(args, **vars(args))


if __name__ == "__main__":
    main()
