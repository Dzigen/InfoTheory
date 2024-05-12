import torch
import sys
from src.utils import initialize_weights

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


BASE_DIR = '.'
#BASE_DIR = '/home/ubuntu/IT'
sys.path.insert(0, f'{BASE_DIR}/')

from src.utils import ImagesDataset, DataLoader, images_collate, TRANSFORM
from src.settings import ParamsConfig
from src.trainloop import run
from src.architectures.VQVAE import VQVAE
from src.architectures.AE import AE

##

DATA_FOLDER = f'{BASE_DIR}/data/cars'
PARAMS_YAML_PATH = f"{BASE_DIR}/params.yaml"

## load parameters ##

PARAMS = ParamsConfig.get_params_config(PARAMS_YAML_PATH)
PARAMS['base_dir'] = BASE_DIR
ARCH_STRUCT = ParamsConfig.get_architecture_params(PARAMS['latent_dim'], 
                                                   PARAMS['use_maxpool'], base_dir=BASE_DIR)

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

if PARAMS['model_type'] == 'vqvae':
    model = VQVAE(config=PARAMS, arch=ARCH_STRUCT).to(PARAMS['device'])
elif PARAMS['model_type'] == 'ae':
    model = AE(config=PARAMS, arch=ARCH_STRUCT).to(PARAMS['device'])
else:
    raise KeyError

## initialize weights ##

initialize_weights(model, PARAMS['weights_init'])

## train model ##

run(PARAMS, model, train_dataloader, test_dataloader, 
    metrics_flag=PARAMS['compute_metrics'])


# https://medium.com/@jaikochhar06/how-to-evaluate-image-quality-in-python-a-comprehensive-guide-e486a0aa1f60
# metric: bad - good 
# psnr  : 0 - 100
# ssim  : -1 - 1
# scc   : -1 (?) - 1
# sam   : -1 - 1