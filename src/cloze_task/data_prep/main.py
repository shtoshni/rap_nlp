import argparse
import os
from os import path
import logging

from cloze_task.data_prep.perform_coreference import process_dataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-input_dir', default='../data/lambada/raw_data', help='Output directory', type=str)
    parser.add_argument(
        '-output_dir', default='../data/lambada/coref_data', help='Output directory', type=str)
    parser.add_argument('-dataset_name', default='lambada',
                        help="Dataset identifier",  type=str)
    parser.add_argument('-model_loc',
                        default="/home/shtoshni/Research/fast-coref/models/longformer_litbank/model.pth")
    parser.add_argument('-source_loc', default='/home/shtoshni/Research/coref_inference/src')
    parser.add_argument('-split', default='train', type=str, choices=['train', 'val', 'test'],
                        help="Maximum number of docs to process per split")
    parser.add_argument('-max_docs', default=None, type=int,
                        help="Maximum number of docs to process per split")
    parser.add_argument('-index_start', default=None, type=int,
                        help="Starting index of doc")
    parser.add_argument('-index_end', default=None, type=int,
                        help="Ending index of doc")

    args = parser.parse_args()

    assert (path.exists(args.model_loc))
    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_dataset(**vars(args))


if __name__ == '__main__':
    main()

