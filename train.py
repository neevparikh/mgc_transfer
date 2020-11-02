import pytorch_lightning as pl

from model import GenreNet
from parsers import common_parser


IMG_SHAPE = (120, 120)

num_classes = {
        'FMA': 8,
        'GTZAN': 10,
    } 

args = common_parser.parse_args()
net = GenreNet(args, input_shape=IMG_SHAPE, num_classes=num_classes[args.dataset])
