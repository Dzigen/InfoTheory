import torch.nn as nn
from src.modules.residual import ResidualBlock

class Decoder(nn.Module):
    activation_map = {
        'relu': nn.ReLU(),
        'leaky': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh()
    }
        
    def __init__(self, arch, config):
        super(Decoder, self).__init__()
        
        convtrans2d_p = {
            'in': arch['dec']['transconv2d']['in'],
            'out': arch['dec']['transconv2d']['out'],
            'ks': arch['dec']['transconv2d']['ks'],
            's': arch['dec']['transconv2d']['s'],
            'p': arch['dec']['transconv2d']['p'],
            'out_p': arch['dec']['transconv2d']['out_p'],
        }

        l_conv2d_p = {
            'in': arch['dec']['l_conv2d']['in'],
            'out': arch['dec']['l_conv2d']['out'],
            'ks': arch['dec']['l_conv2d']['ks'],
            's': arch['dec']['l_conv2d']['s'],
            'p': arch['dec']['l_conv2d']['p'],
        }

        dec_layers = []
        for in_c, out_c, ks, strd, pad, out_pad in zip(*list(convtrans2d_p.values())):
            if config['use_residuals']:
                dec_layers.append(ResidualBlock(in_c, self.activation_map[config['act_fn']],
                                                 config['use_batchnorm'], config['residual_nlayers']))
                
            dec_layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=ks, stride=strd, 
                                                 padding=pad, output_padding=out_pad))
            
            if config['use_batchnorm']:
                dec_layers.append(nn.BatchNorm2d(out_c))
            
            dec_layers.append(self.activation_map[config['act_fn']])

        # adding conv2d-layer
        dec_layers.append(nn.Conv2d(l_conv2d_p['in'], l_conv2d_p['out'], kernel_size=l_conv2d_p['ks'], 
                                    stride=l_conv2d_p['s'], padding=l_conv2d_p['p']))
        if config['use_batchnorm']:
                dec_layers.append(nn.BatchNorm2d(l_conv2d_p['out']))
        dec_layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.layers(x)