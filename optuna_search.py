from src.utils import ImagesDataset, DataLoader, images_collate, TRANSFORM
from src.settings import ParamsConfig
from src.trainloop import run
from src.architectures.VQVAE import VQVAE

import torch
import optuna
import joblib
torch.autograd.set_detect_anomaly(True)

OPTUNA_TRIALS = 30
DATA_FOLDER = './data/cars'
PARAMS_YAML_PATH = "./params.yaml"
PARAMS = ParamsConfig.get_params_config(PARAMS_YAML_PATH)

train_dataset = ImagesDataset('train', DATA_FOLDER, TRANSFORM)
TRAIN_DATALOADER = DataLoader(
    train_dataset,batch_size=PARAMS['batch_size'],shuffle=True,
    num_workers=PARAMS['num_workers'], collate_fn=images_collate)

test_dataset = ImagesDataset('test', DATA_FOLDER, TRANSFORM)
TEST_DATALOADER = DataLoader(
    test_dataset,batch_size=PARAMS['batch_size'],shuffle=False,
    num_workers=PARAMS['num_workers'], collate_fn=images_collate)

print("train size: ", len(train_dataset))
print("test size: ", len(test_dataset))

def objective(trial):
    global TRAIN_DATALOADER
    global TEST_DATALOADER
    global PARAMS

    PARAMS['batch_size'] = trial.suggest_categorical('batch_size', [2,4,8,16,32,64])
    PARAMS['lr'] = trial.suggest_categorical('lr', [0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    PARAMS['weights_init'] = trial.suggest_categorical('weights_init', [
        'normal', 'uniform', 'xavier_normal', 'xavier_uniform', 
        'kaiming_uniform', 'kaiming_normal'])
    PARAMS['act_fn'] = trial.suggest_categorical('act_fn', [
        'relu', 'leaky', 'gelu', 'silu'])
    PARAMS['use_batchnorm'] = trial.suggest_categorical('use_batchnorm', [True, False])
    
    PARAMS['use_residuals'] = trial.suggest_categorical('use_residuals', [True, False])
    if PARAMS['use_residuals']:
        PARAMS['residual_nlayers'] = trial.suggest_categorical('residual_nlayers', [1, 2])

    PARAMS['b_quantization'] = trial.suggest_categorical('b_quantization', [1, 2, 3])

    PARAMS['codebook_size'] = trial.suggest_int('codebook_size', 20, 100, 5)
    PARAMS['reconstruction_loss_weight'] = trial.suggest_float('reconstruction_loss_weight', 1, 20, step=1)
    PARAMS['codebook_loss_weight'] = trial.suggest_float('codebook_loss_weight', 0.2, 20, step=0.2)
    PARAMS['commitment_loss_weight'] = trial.suggest_float('commitment_loss_weight', 0.01, 20, step=0.01)

    model = VQVAE(config=PARAMS).to(PARAMS['device'])
    _, _, _, best_epoch, eval_scores = run(PARAMS, model, 
                                          TRAIN_DATALOADER, TEST_DATALOADER)

    return eval_scores[best_epoch]['median_PSNR'], eval_scores[best_epoch]['median_SSIM'], eval_scores[best_epoch]['median_SCC'], eval_scores[best_epoch]['median_SAM']

study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

joblib.dump(study, f"./optuna_study.pkl")