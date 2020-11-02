import torch


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
