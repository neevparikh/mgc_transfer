import pytorch_lightning as pl

from model import GenreNet
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
data.setup()
if args.pretrained:
    net = GenreNet.load_from_checkpoint(args.pretrained)
else:
    net = GenreNet(args, input_shape=data.shape, num_classes=data.num_labels)
trainer = pl.Trainer(gpus=args.num_gpus,
                     precision=16 if args.num_gpus > 0 else 32,
                     accelerator='ddp' if args.num_gpus > 0 else None,
                     limit_train_batches=1.0,
                     limit_val_batches=1.0,
                     limit_test_batches=1.0)
trainer.fit(net, datamodule=data)
trainer.test(net, datamodule=data)
