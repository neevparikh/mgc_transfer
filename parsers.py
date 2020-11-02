# yapf: disable

import argparse

float_to_int = lambda x: int(float(x))

common_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_parser.add_argument('--lr', type=float, required=True,
        help='Learning rate')
common_parser.add_argument('--dataset', type=float, required=True, choices=['FMA', 'GTZAN'],
        help='Dataset type')
