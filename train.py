import pytorch_lightning as pl
import torch

from model import GenreNet, _YAMNET_LAYER_DEFS
from parsers import common_parser
from data import FMA_Large, FMA_Small, GTZAN

args = common_parser.parse_args()

if args.dataset == 'FMA_L':
    data = FMA_Large(args)
elif args.dataset == 'FMA_S':
    data = FMA_Small(args)
elif args.dataset == 'GTZAN':
    data = GTZAN(args)
else:
    raise ValueError("Unrecognized dataset")
data.prepare_data()
data.setup('fit')
if args.pretrained:
    net = GenreNet.load_from_checkpoint(args.pretrained, args=args, input_shape=data.shape,
            num_classes=16)
    net.head = torch.nn.Sequential(
            net.head[0],
            net.head[1],
            torch.nn.Linear(_YAMNET_LAYER_DEFS[-1][-1], data.num_labels),
            net.head[3],
        )
    net.num_classes = data.num_labels
else:
    net = GenreNet(args, input_shape=data.shape, num_classes=data.num_labels)
trainer = pl.Trainer(gpus=args.num_gpus,
                     precision=16 if args.num_gpus > 0 else 32,
                     accelerator='ddp' if args.num_gpus > 0 else None,
                     limit_train_batches=1.0,
                     limit_val_batches=1.0,
                     limit_test_batches=1.0)
trainer.fit(net, datamodule=data)
data.setup('test')
trainer.test(net, datamodule=data)
