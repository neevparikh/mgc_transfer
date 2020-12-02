import torch

class Reshape(torch.nn.Module):
    """
    Description:
        Module that returns a view of the input which has a different size

    Parameters:
        - args : Int...
            The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, x):
        return x.squeeze().view(*self.shape)


def _conv(kernel, stride, inc, outc):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel, stride=stride),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(),
    )


def _separable_conv(kernel, stride, inc, outc):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=inc,
                        out_channels=inc,
                        kernel_size=kernel,
                        groups=inc,
                        stride=stride),
        torch.nn.BatchNorm2d(inc),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 1)),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(),
    )
