from torchmetrics.image import SpatialCorrelationCoefficient
from torchmetrics.image import SpectralAngleMapper
from skimage.metrics import structural_similarity as ssim

import numpy as np
from src.utils import REFORM
from numpy import log10, sqrt

scc = SpatialCorrelationCoefficient()
sam = SpectralAngleMapper(reduction='none')

def PSNR(original, compressed, mode='norm'): 
    if mode == 'norm':
        mse = np.mean((np.array(REFORM(original)) - np.array(REFORM(compressed))) ** 2) 
    else:
        mse = np.mean((np.array(original) - np.array(compressed)) ** 2) 
    
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 