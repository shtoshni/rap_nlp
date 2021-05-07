import argparse
import os
from os import path
import hashlib
import logging
from collections import OrderedDict

from summarization.finetune_lm import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-data_dir', default=None, help='Data directory. Use this when it is specified', type=str)
    parser.add_argument('-base_model_dir', default='../models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('-model_dir', default=None,
                        help='Model directory', type=str)

    parser.add_argument('-dataset', default='xsum', choices=['xsum'], type=str)
    parser.add_argument('-model', default='sshleifer/distilbart-xsum-6-6', type=str, help='Model size')
    # Format input
    parser.add_argument('-format_input', default='', choices=['coref', 'coref_null', ''], type=str,
                        help="Formatting of input.")
    # Format output
    parser.add_argument('-format_output', default='', choices=['', 'coref'], type=str,
                        help="Formatting of output.")

    # Training params
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-num_eval_docs', default=None, type=int,
                        help='Number of evaluation docs.')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=25, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-max_gradient_norm',
                        help='Maximum gradient norm', default=1.0, type=float)
    parser.add_argument('-lr', help="Initial learning rate",
                        default=3e-4, type=float)
    parser.add_argument('-not_save_model', dest='to_save_model', help="Whether to save model during training or not",
                        default=True, action="store_false")
    parser.add_argument('-eval', dest='eval_model', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    if args.dataset == 'litbank':
        args.train_with_singletons = True

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'max_segment_len',  # Encoder params
                'ment_emb', "doc_enc", 'max_span_width', 'top_span_ratio',  # Mention model
                'mem_type', 'entity_rep', 'mlp_size',  # Memory params
                'dropout_rate', 'seed', 'init_lr', 'max_epochs',
                'label_smoothing_wt', 'ment_loss',  # weights & sampling
                'num_train_docs', 'train_with_singletons',  'dataset',  # Dataset params
                ]

    if args.cluster_mlp_size != parser.get_default('cluster_mlp_size'):
        imp_opts.append('cluster_mlp_size')

    if args.singleton_file is not None and path.exists(args.singleton_file):
        imp_opts.append('singleton_file')

    if args.fine_tune_lr is not None:
        imp_opts.append('fine_tune_lr')

    if args.sim_func is not parser.get_default('sim_func'):
        imp_opts.append('sim_func')

    # Adding conditional important options
    if args.mem_type in ['learned', 'lru']:
        # Number of max entities only matters for bounded memory models
        imp_opts.append('max_ents')
    else:
        args.max_ents = None

    if args.dataset == 'litbank':
        # Cross-validation split is only important for litbank
        imp_opts.append('cross_val_split')

    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = f"longformer_{args.dataset}_" + str(hash_idx)

    if args.eval_model:
        args.max_training_segments = None

    if args.model_dir is None:
        model_dir = path.join(args.base_model_dir, model_name)
        args.model_dir = model_dir
        best_model_dir = path.join(model_dir, 'best_models')
        args.best_model_dir = best_model_dir
        if not path.exists(model_dir):
            os.makedirs(model_dir)
        if not path.exists(best_model_dir):
            os.makedirs(best_model_dir)
    else:
        best_model_dir = path.join(args.model_dir, 'best_models')
        if not path.exists(best_model_dir):
            best_model_dir = args.model_dir
        args.best_model_dir = best_model_dir

    print("Model directory:", args.model_dir)

    if args.data_dir is None:
        if args.dataset == 'litbank':
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}/{args.cross_val_split}')
            args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll/{args.cross_val_split}')
        elif args.dataset == 'ontonotes':
            if args.train_with_singletons:
                enc_str = "_singletons"
            else:
                enc_str = ""
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}{enc_str}')
            args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll')
    else:
        if args.dataset == 'litbank':
            args.data_dir = path.join(args.data_dir, f'{args.cross_val_split}')
        elif args.dataset == 'ontonotes':
            args.conll_data_dir = path.join(path.dirname(args.data_dir.rstrip("/")), "conll")

    base_data_dir = path.dirname(path.dirname(args.data_dir))
    if args.dataset == 'litbank':
        args.conll_data_dir = path.join(base_data_dir, f'litbank/conll/{args.cross_val_split}')
    else:
        args.conll_data_dir = path.join(base_data_dir, f'ontonotes/conll')

    print(args.data_dir)
    print(args.conll_data_dir)

    # Get mention model name
    # args.pretrained_mention_model = path.join(
    #     path.join(args.base_model_dir, get_mention_model_name(args)), "best_models/model.pth")
    # print(args.pretrained_mention_model)

    # Log directory for Tensorflow Summary
    log_dir = path.join(args.model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = path.join(args.model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    Experiment(**vars(args))


if __name__ == "__main__":
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    main()
