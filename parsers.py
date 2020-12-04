# yapf: disable

import argparse

float_to_int = lambda x: int(float(x))

common_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_parser.add_argument('--lr', type=float, required=False, default=0.00025,
        help='Learning rate')
common_parser.add_argument('--num-gpus', type=int, required=False, default=1,
        help='Number of GPUs to use')
common_parser.add_argument('--data-workers', type=int, required=False, default=8,
        help='Number of CPU workers to use for loading data')
common_parser.add_argument('--batchsize', type=int, required=False, default=64,
        help='Batchsize for minibatch')
common_parser.add_argument('--dataset', type=str, required=True,
        choices=['FMA_L', 'FMA_S', 'GTZAN'],
        help='Dataset type')
