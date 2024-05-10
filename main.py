import torch
import sys
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

sys.path.insert(0, './')

from src.utils import ImagesDataset, DataLoader, images_collate, TRANSFORM
from src.settings import ParamsConfig
from src.trainloop import run
from src.architectures.VQVAE import VQVAE

##

DATA_FOLDER = './data/cars'
PARAMS_YAML_PATH = "./params.yaml"

## load parameters ##

PARAMS = ParamsConfig.get_params_config(PARAMS_YAML_PATH)

## load data ##

train_dataset = ImagesDataset('train', DATA_FOLDER, TRANSFORM)
train_dataloader = DataLoader(
    train_dataset,batch_size=PARAMS['batch_size'],shuffle=True,
    num_workers=PARAMS['num_workers'], collate_fn=images_collate)

test_dataset = ImagesDataset('test', DATA_FOLDER, TRANSFORM)
test_dataloader = DataLoader(
    test_dataset,batch_size=PARAMS['batch_size'],shuffle=False,
    num_workers=PARAMS['num_workers'], collate_fn=images_collate)

print("train size: ", len(train_dataset))
print("test size: ", len(test_dataset))

## load model ##

model = VQVAE(config=PARAMS).to(PARAMS['device'])

## train model ##

run(PARAMS, model, train_dataloader, test_dataloader)


# https://medium.com/@jaikochhar06/how-to-evaluate-image-quality-in-python-a-comprehensive-guide-e486a0aa1f60
# metric: bad - good 
# psnr  : 0 - 100
# ssim  : -1 - 1
# scc   : ? - 1
# sam   : ? - 1