import torch.nn as nn
import torch

from src.modules.quantizer import get_quantizer
from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from src.utils import generate_noise

class VQVAE(nn.Module):

    def __init__(self, config, arch):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(arch, config)
        self.quantizer = get_quantizer(config)
        self.decoder = Decoder(arch, config)

        #
        self.config = config
        self.arch = arch
        self.criterion = torch.nn.MSELoss()
            
    def forward(self, x):
        #print("input: ", x.shape)
        enc = self.encoder(x)
        #print("encoder: ",enc.shape)

        quant_output, quant_loss, quant_idxs = self.quantizer(enc)
        #print("quant out: ", quant_output.shape)
        
        if self.config['add_noise']:
            # Adding noise for quanization-error emulation
            noise = generate_noise(quant_output, self.config['b_quantization'], 
                               self.config['device'])
            quant_output = quant_output + noise
            #print("noised quant_output: ", quant_output.shape)

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