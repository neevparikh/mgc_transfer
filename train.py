import pytorch_lightning as pl

from model import GenreNet
from parsers import common_parser
from data import FMA_Large, FMA_Small, GTZAN

args = common_parser.parse_args()

if args.dataset == 'FMA_L':
    data = FMA_Large()
elif args.dataset == 'FMA_S':
    data = FMA_Small()
elif args.dataset == 'GTZAN':
    data = GTZAN()
else:
    raise ValueError("Unrecognized dataset")

net = GenreNet(args, input_shape=data.shape, num_classes=data.labels)
