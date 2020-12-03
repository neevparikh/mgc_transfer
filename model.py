import pytorch_lightning as pl
import torch

from modules import _conv, _separable_conv, Reshape
from utils import conv2d_size_out
from constants import BATCHSIZE

_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, inc , outc)
    (_conv, [3, 3], 2, 1, 32),
    (_separable_conv, [3, 3], 1, 32, 64),
    (_separable_conv, [3, 3], 2, 64, 128),
    (_separable_conv, [3, 3], 2, 128, 256),
    (_separable_conv, [3, 3], 2, 256, 512),
    (_separable_conv, [3, 3], 1, 512, 512),
]

class GenreNet(pl.LightningModule):
    """ GenreNet model definition - based off YAMnet
    """
    def __init__(self, args, input_shape, num_classes):
        super().__init__()

        self.args = args
        self.lr = args.lr

        final_size = input_shape
        for _, kernel, stride, _, _ in _YAMNET_LAYER_DEFS:
            final_size = conv2d_size_out(final_size, kernel_size=kernel, stride=stride)
        self.body = torch.nn.Sequential(*map(lambda t: t[0](*t[1:]), _YAMNET_LAYER_DEFS))

        self.head = torch.nn.Sequential(
                    torch.nn.AvgPool2d(final_size),
                    Reshape(BATCHSIZE, -1),
                    torch.nn.Linear(_YAMNET_LAYER_DEFS[-1][-1], num_classes),
                    torch.nn.Softmax(-1),
                )

        print(self)
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")


    def forward(self, sterogram):
        output = self.body(sterogram)
        predictions = self.head(output)
        return predictions
    
    def training_step(self, batch, batch_idx):
        sterograms, labels = batch
        predictions = self(sterograms)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        self.log('training_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sterograms, labels = batch
        return self._shared_eval(sterograms, labels, 'val')

    def test_step(self, batch, batch_idx):
        sterograms, labels = batch
        return self._shared_eval(sterograms, labels, 'test')

    def _shared_eval(self, sterograms, labels, prefix):
        predictions = self(sterograms)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        self.log('{}_loss'.format(prefix), loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
