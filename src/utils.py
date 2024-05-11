import os 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn

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

def generate_noise(hidden, b, device):
    lb = 0
    rb = (torch.max(hidden) / pow(2, b + 1))
    noise = ((lb - rb) * torch.rand(hidden.shape, device=device) + rb)

    return noise

def initialize_weights(model, weights_init):
    for m in model.modules():
        if not (isinstance(m, nn.Conv2d) or isinstance(m, nn.Embedding) or 
                isinstance(m, nn.ConvTranspose2d)):
            continue
        
        if weights_init == 'normal':
            nn.init.normal_(m.weight)

        elif weights_init == 'uniform':
            nn.init.uniform_(m.weight)

        elif weights_init == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)

        elif weights_init == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        elif weights_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        elif weights_init == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        else:
            raise KeyError