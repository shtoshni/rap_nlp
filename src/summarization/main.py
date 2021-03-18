import argparse
import os
from os import path
import logging

from summarization.determine_singletons import process_dataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_output_dir', default='../data/summarization', help='Output directory', type=str)
    parser.add_argument('-dataset_name', default='xsum', choices=['xsum', 'cnn_dailymail'],
                        help="Dataset identifier",  type=str)
    parser.add_argument('-model_loc',
                        default="/home/shtoshni/Research/long-doc-coref/models/umem_singleton_90K/model.pth")
    parser.add_argument('-max_docs', default=None, type=int,
                        help="Maximum number of docs to process per split")
    parser.add_argument('-split', default=None, type=str, choices=['train', 'validation', 'test'],
                        help="Maximum number of docs to process per split")
    parser.add_argument('-index_start', default=None, type=int,
                        help="Starting index of doc")
    parser.add_argument('-index_end', default=None, type=int,
                        help="Ending index of doc")

    args = parser.parse_args()

    assert (path.exists(args.model_loc))

    args.output_dir = path.join(args.base_output_dir, args.dataset_name)
    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_dataset(**vars(args))


if __name__ == '__main__':
    main()

