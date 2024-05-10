import torch.nn as nn
import torch

from src.modules.quantizer import get_quantizer
from src.modules.residual import ResidualBlock

class VQVAE(nn.Module):
    activation_map = {
        'relu': nn.ReLU(),
        'leaky': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU()
    }

    def __init__(self, config):
        super(VQVAE, self).__init__()

        #
        enc_conv2d_params = {
            'in': config['enc']['conv2d']['in'],
            'out': config['enc']['conv2d']['out'],
            'ks': config['enc']['conv2d']['ks'],
            's': config['enc']['conv2d']['s'],
            'p': config['enc']['conv2d']['p']
        }
        enc_layers = []
        for in_c, out_c, ks, strd, pad in zip(*list(enc_conv2d_params.values())):
            enc_layers.append(nn.Conv2d(in_c, out_c, kernel_size=ks, stride=strd, padding=pad))
            
            if config['use_batchnorm']:
                enc_layers.append(nn.BatchNorm2d(out_c))
            
            enc_layers.append(self.activation_map[config['act_fn']])

            if config['use_residuals']:
                enc_layers.append(ResidualBlock(out_c, self.activation_map[config['act_fn']], 
                                                config['use_batchnorm'], config['residual_nlayers']))
        if config['use_residuals']:
            # remove last residualblock-layer
            enc_layers = enc_layers[:-1]
        self.encoder = nn.Sequential(*enc_layers)

        #
        self.quantizer = get_quantizer(config)

        #
        dec_convtrans2d_params = {
            'in': config['dec']['transconv2d']['in'],
            'out': config['dec']['transconv2d']['out'],
            'ks': config['dec']['transconv2d']['ks'],
            's': config['dec']['transconv2d']['s'],
            'p': config['dec']['transconv2d']['p'],
            'out_p': config['dec']['transconv2d']['out_p'],
        }
        dec_layers = []
        for in_c, out_c, ks, strd, pad, out_pad in zip(*list(dec_convtrans2d_params.values())):
            if config['use_residuals']:
                dec_layers.append(ResidualBlock(in_c, self.activation_map[config['act_fn']],
                                                 config['use_batchnorm'], config['residual_nlayers']))
                
            dec_layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=ks, stride=strd, 
                                                 padding=pad, output_padding=out_pad))
            
            if config['use_batchnorm']:
                dec_layers.append(nn.BatchNorm2d(out_c))
            
            dec_layers.append(self.activation_map[config['act_fn']])

        dec_layers.append(nn.Sigmoid())        
        self.decoder = nn.Sequential(*dec_layers)

        #
        self.config = config
        self.criterion = torch.nn.MSELoss()
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if not (isinstance(m, nn.Conv2d) or isinstance(m, nn.Embedding) or 
                    isinstance(m, nn.ConvTranspose2d)):
                continue
            
            if self.config['weights_init'] == 'normal':
                nn.init.normal_(m.weight)

            elif self.config['weights_init'] == 'uniform':
                nn.init.uniform_(m.weight)

            elif self.config['weights_init'] == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)

            elif self.config['weights_init'] == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

            elif self.config['weights_init'] == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

            elif self.config['weights_init'] == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            else:
                raise KeyError
            
    def forward(self, x):
        #print("input: ", x.shape)
        enc = self.encoder(x)
        #print("encoder: ",enc.shape)

        quant_output, quant_loss, quant_idxs = self.quantizer(enc)
        #print("quant out: ", quant_output.shape)

        # Adding noise for quanization-error emulation
        lb = 0
        rb = (torch.max(quant_output) / pow(2, self.config['b_quantization'] + 1))
        noise = ((lb - rb) * torch.rand(quant_output.shape, device=self.config['device']) + rb)
        quant_output = quant_output + noise
        #print("noised quant out: ", quant_output.shape)

        out = self.decoder(quant_output)
        #print("decoder: ", out.shape)

        output = {
            'generated_image' : out,
            'quantized_output' : quant_output,
            'quantized_losses' : quant_loss,
            'quantized_indices' : quant_idxs
        }

        return output
    
    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantizer.quantize_indices(indices)
        dec_input = self.post_quant_conv(quantized_output)
        return self.decoder(dec_input)
    
    def compute_loss(self, output, target_images):
        generated_image = output['generated_image']
        quantize_losses = output['quantized_losses']

        recon_loss = self.criterion(generated_image, target_images)
        loss = (self.config['reconstruction_loss_weight']*recon_loss +
                self.config['codebook_loss_weight']*quantize_losses['codebook_loss'] +
                self.config['commitment_loss_weight']*quantize_losses['commitment_loss'])

        return loss