import torch.nn as nn
from src.modules.residual import ResidualBlock

class Encoder(nn.Module):
    activation_map = {
        'relu': nn.ReLU(),
        'leaky': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh()
    }

    def __init__(self, arch, config):
        super(Encoder, self).__init__()

        conv2d_p = {
            'in': arch['enc']['conv2d']['in'],
            'out': arch['enc']['conv2d']['out'],
            'ks': arch['enc']['conv2d']['ks'],
            's': arch['enc']['conv2d']['s'],
            'p': arch['enc']['conv2d']['p']
        }

        if config['use_maxpool']:
            maxpool_p = {
                'ks': arch['enc']['maxpool']['ks'],
                's': arch['enc']['maxpool']['s'],
                'p': arch['enc']['maxpool']['p']
            }

        enc_layers = []
        for i in range(len(conv2d_p['in'])-1):
            enc_layers.append(nn.Conv2d(conv2d_p['in'][i], conv2d_p['out'][i], 
                                        kernel_size=conv2d_p['ks'][i], stride=conv2d_p['s'][i], padding=conv2d_p['p'][i]))
            
            out_c = conv2d_p['out'][i]
            if config['use_maxpool']:
                enc_layers.append(nn.MaxPool2d(kernel_size=maxpool_p['ks'][i], stride=maxpool_p['s'][i], 
                                               padding=maxpool_p['p'][i]))
                out_c = conv2d_p['in'][i+1]

            if config['use_batchnorm']:
                enc_layers.append(nn.BatchNorm2d(out_c))
            
            enc_layers.append(self.activation_map[config['act_fn']])

            if config['use_residuals']:
                enc_layers.append(ResidualBlock(out_c, self.activation_map[config['act_fn']], 
                                                config['use_batchnorm'], config['residual_nlayers']))
        if config['use_residuals']:
            # remove last residualblock-layer
            enc_layers = enc_layers[:-1]
        self.layers = nn.Sequential(*enc_layers)

    def forward(self, x):
        return self.layers(x)