import torch.nn as nn
import torch
from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from src.utils import generate_noise

class AE(nn.Module):
    def __init__(self, config, arch):
        super(AE, self).__init__()

        self.encoder = Encoder(arch, config)
        self.decoder = Decoder(arch, config)

        self.config = config
        self.arch = arch
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        #print("input: ", x.shape)
        enc_out = self.encoder(x)
        #print("encoder: ",enc_out.shape)

        if self.config['add_noise']:
            # Adding noise for quanization-error emulation
            noise = generate_noise(enc_out, self.config['b_quantization'], 
                               self.config['device'])
            enc_out = enc_out + noise
            #print("noised enc_out: ", enc_out.shape)


        out = self.decoder(enc_out)
        #print("decoder: ", out.shape)

        output = {
            'generated_image' : out,
        }

        return output


    def compute_loss(self, output, target_images):
        generated_image = output['generated_image']
        #print(generated_image.shape, target_images.shape)

        recon_loss = self.criterion(generated_image, target_images)
        loss = self.config['reconstruction_loss_weight']*recon_loss

        return loss