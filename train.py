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
data.prepare_data()
data.setup('fit')
net = GenreNet(args, input_shape=data.shape, num_classes=data.num_labels)
trainer = pl.Trainer(gpus=args.num_gpus, precision=16, accelerator='ddp', limit_train_batches=0.5, limit_val_batches=0.5, limit_test_batches=0.5)
trainer.fit(net, datamodule=data)
trainer.test(net, datamodule=data)
