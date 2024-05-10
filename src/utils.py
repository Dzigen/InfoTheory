import os 
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

from src.metrics import PSNR

TRANSFORM = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    ])

REFORM = T.ToPILImage()

class ImagesDataset(Dataset):
    def __init__(self, part_name, data_dir, processor):
        info_df = pd.read_csv(f"{data_dir}/info.csv", sep=';')
        self._data = info_df.loc[info_df['part'] == part_name, :].reset_index(drop=True)
        self.processor = processor
        self.data_dir = data_dir
        self.data_part = part_name

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        image_path = f"{self.data_dir}/{self.data_part}/{self._data['image_name'][idx]}"
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.processor(image)
        image.close()
        
        label = torch.clone(image_tensor)

        return image_tensor, label
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]
        

def images_collate(data):

    images = torch.cat([torch.unsqueeze(item[0], 0) for item in data], 0)
    labels = torch.cat([torch.unsqueeze(item[1], 0) for item in data], 0)

    return {
        "images": images, 
        "labels": labels
    }

#This function is searching for the JPEG quality factor (QF)
#which provides neares compression to TargetBPP
def JPEGRDSingleImage(image,TargetBPP):
    width, height = image.size
    realbpp, realpsnr, realQ = 0, 0, 0
    save_file = 'test.jpeg'

    for Q in range(101):
        image.save(save_file, "JPEG", quality=Q)
        image_dec = Image.open(save_file)
        bytesize = os.path.getsize(save_file)
        bpp = bytesize*8/(width*height)
        psnr = PSNR(image, image_dec, mode=None)

        if abs(realbpp-TargetBPP)>=abs(bpp-TargetBPP):
            realQ = Q
    
    #
    image.save(save_file, "JPEG", quality=realQ)
    image_dec = Image.open(save_file)
    bytesize = os.path.getsize(save_file)
    realbpp = bytesize*8/(width*height)
    realpsnr = PSNR(image, image_dec, mode=None)
    os.remove(save_file)

    return image_dec, realQ, realbpp, realpsnr