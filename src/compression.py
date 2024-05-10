import os
import torch

from src.architectures.VQVAE import VQVAE
from src.codec.EntropyCodec import *
from src.utils import REFORM
import numpy as np


class Compressor:
    """Реализация алгоритма сжатия/декомпрессии изображений
    """

    def __init__(self, ae_model_path, config, save_dir) -> None:
        self.sc = ScalarQuantizer(config['b_quantization'])
        self.ac = ArithmeticCoder(save_dir, feature_size=[config['latent_dim']]*3)

        self.model = VQVAE(config).to(config['device'])
        self.model.load_state_dict(torch.load(ae_model_path))
        self.model.eval()
        self.config = config

    def compress(self, x, name):
        print("Compression Start")
        print("1. x: ", x.shape)

        # енкодер-часть автоенкодера
        x = x.to(self.config['device'])
        encoder_out = self.model.encoder(x)
        print("2. encoder_out: ", encoder_out.shape)

        # vq-часть автоенкодера
        quant_output, _, _ = self.model.quantizer(encoder_out)
        print("3. quant_output: ", quant_output.shape)

        # скалярное кодирование
        quant_output = quant_output[0].cpu().detach().numpy()  
        sc_x, max_x = self.sc.encode(quant_output)
        print("4. sc_x: ", sc_x.shape)

        # арифметическое кодирование
        save_file, bpp = self.ac.code(sc_x, name)
        print('5. save_file: ', save_file)
        
        return save_file, max_x, bpp

    def decompress(self, name, max_x):
        print("Decompression Start")
        print("1. name: ", name)

        # арифметическое декодирование
        decoded_x = self.ac.decode(name)
        print("2. decoded_x: ", decoded_x.shape)

        # скалярное  декодирование
        dequant_x = self.sc.decode(decoded_x, max_x)
        print("3. dequant_x: ", dequant_x.shape)

        dequant_x = torch.unsqueeze(torch.tensor(dequant_x), 0)
        dequant_x = dequant_x.to(self.config['device'])
        print("4. dequant_x (unsqueeze): ", dequant_x.shape)

        # decoder-часть автоенкодера
        decoder_out = self.model.decoder(dequant_x)
        print("5. decoder_out: ", decoder_out.shape)
        decoder_out = torch.squeeze(decoder_out, 0)
        print("6. decoder_out(squeeze): ", decoder_out.shape)

        return REFORM(decoder_out)

class ScalarQuantizer:
    """Реализация алгоритма для выполнения 
    прямого и обратного скалярного квантования
    """

    def __init__(self, b) -> None:
        self.b = b

    def encode(self, x):
        max_x = np.max(x)
        normalized_x = x / max_x

        quantized_x = np.clip(normalized_x, 0, 0.9999999)
        quantized_x = quantized_x*pow(2, self.b)
        quantized_x = quantized_x.astype(np.int32)

        return quantized_x, max_x

    def decode(self, x, max_x):

        shift = 1.0 / pow(2, self.b+1)
        dequantized_x = x.astype(np.float32) / pow(2, self.b)
        dequantized_x += shift
        dequantized_x = (dequantized_x * max_x).astype(np.float32)

        return dequantized_x
    
class ArithmeticCoder:
    """Реализация алгоритма для выполнения арифметического кодирования/декодирования
    """

    def __init__(self, save_dir, feature_size=(8,8,8), init_size=(128,128), device='cuda') -> None:
        self.f_size = feature_size
        self.init_w, self.init_h = init_size
        self.save_dir = save_dir
        self.device = device

    def code(self, x, name):
        save_file = f"{self.save_dir}/{name}.bin"

        self.EntropyEncoder(save_file, x, *self.f_size)

        bytesize = os.path.getsize(save_file)
        bpp = bytesize * 8 / (self.init_w * self.init_h)

        return save_file, bpp

    def decode(self, name):
        save_file = f"{self.save_dir}/{name}.bin"
        out = self.EntropyDecoder(save_file,  *self.f_size)
        os.remove(save_file)

        return out
    
    #Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
    def EntropyEncoder (self, filename,x,size_z,size_h,size_w):
        temp = np.zeros((size_z, size_h, size_w), np.uint8, 'C')
        for z in range(size_z):
            for h in range(size_h):
                for w in range(size_w):
                    temp[z][h][w] = x[z][h][w]
        maxbinsize = (size_h * size_w * size_z)
        bitstream = np.zeros(maxbinsize, np.uint8, 'C')
        StreamSize = np.zeros(1, np.int32, 'C')

        HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)

        fp = open(filename, 'wb')
        out = bitstream[0:StreamSize[0]]
        out.astype('uint8').tofile(fp)
        fp.close()

    #Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
    def EntropyDecoder (self, filename,size_z,size_h,size_w):
        fp = open(filename, 'rb')
        bitstream = fp.read()
        fp.close()

        bitstream = np.frombuffer(bitstream, dtype=np.uint8)
        declayers = np.zeros((size_z, size_h, size_w), np.uint8, 'C')
        FrameOffset = np.zeros(1, np.int32, 'C')
        FrameOffset[0] = 0

        HiddenLayersDecoder(declayers, size_w, size_h, size_z, bitstream, FrameOffset)
        
        return declayers
