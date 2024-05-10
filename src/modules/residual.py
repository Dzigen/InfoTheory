import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, act_fn, use_batchnorm, conv2d_amount=1):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(conv2d_amount):
            layers.append(nn.Conv2d(channels, channels, 
                                    kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(act_fn)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = out + residual
        return out