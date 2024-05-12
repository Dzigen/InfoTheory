import torch
import optuna
import joblib
import gc
import sys
torch.autograd.set_detect_anomaly(True)

#BASE_DIR = '.'
BASE_DIR = '/home/ubuntu/IT'
sys.path.insert(0, f"{BASE_DIR}/")

from src.utils import ImagesDataset, DataLoader, images_collate, TRANSFORM
from src.settings import ParamsConfig
from src.trainloop import run
from src.architectures.VQVAE import VQVAE
from src.architectures.AE import AE
from src.utils import initialize_weights

OPTUNA_TRIALS = 50
DATA_FOLDER = f'{BASE_DIR}/data/cars'
PARAMS_YAML_PATH = f"{BASE_DIR}/params.yaml"
PARAMS = ParamsConfig.get_params_config(PARAMS_YAML_PATH)
PARAMS['base_dir'] = BASE_DIR

train_dataset = ImagesDataset('train', DATA_FOLDER, TRANSFORM)
test_dataset = ImagesDataset('test', DATA_FOLDER, TRANSFORM)

print("train size: ", len(train_dataset))
print("test size: ", len(test_dataset))

def objective(trial):
    global PARAMS
    global train_dataset
    global test_dataset

    # model_type = 'vqvae'
    # use_noise = True

    #
    PARAMS['batch_size'] = trial.suggest_categorical('batch_size', [2,4,8,16,32])
    PARAMS['lr'] = trial.suggest_categorical('lr', [0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    
    #
    PARAMS['weights_init'] = trial.suggest_categorical('weights_init', [
        'normal', 'uniform', 'xavier_normal', 'xavier_uniform', 
        'kaiming_uniform', 'kaiming_normal'])
    PARAMS['act_fn'] = trial.suggest_categorical('act_fn', [
        'relu', 'leaky', 'gelu', 'silu', 'tanh'])
    PARAMS['use_batchnorm'] = trial.suggest_categorical('use_batchnorm', [True, False])
    
    #
    PARAMS['use_maxpool'] = trial.suggest_categorical('use_maxpool', [True, False])

    PARAMS['use_residuals'] = trial.suggest_categorical('use_residuals', [True, False])
    if PARAMS['use_residuals']:
        PARAMS['residual_nlayers'] = trial.suggest_categorical('residual_nlayers', [1, 2])

    ARCH_STRUCT = ParamsConfig.get_architecture_params(PARAMS['latent_dim'], PARAMS['use_maxpool'], 
                                                       base_dir=PARAMS['base_dir'])
    if PARAMS['add_noise']:
        PARAMS['b_quantization'] = trial.suggest_categorical('b_quantization', [2, 3])

    #
    PARAMS['reconstruction_loss_weight'] = trial.suggest_float('reconstruction_loss_weight', 1, 20, step=1)

    if PARAMS['model_type'] == 'vqvae':
        PARAMS['codebook_size'] = trial.suggest_int('codebook_size', 20, 100, 5)
        PARAMS['codebook_loss_weight'] = trial.suggest_float('codebook_loss_weight', 0.2, 20, step=0.2)
        PARAMS['commitment_loss_weight'] = trial.suggest_float('commitment_loss_weight', 0.01, 20, step=0.01)
        model = VQVAE(config=PARAMS, arch=ARCH_STRUCT).to(PARAMS['device'])
    elif PARAMS['model_type'] == 'ae':
        model = AE(config=PARAMS, arch=ARCH_STRUCT).to(PARAMS['device'])
    else:
        raise KeyError

    ## init datalodaers ##
    TRAIN_DATALOADER = DataLoader(
        train_dataset,batch_size=PARAMS['batch_size'],shuffle=True,
        num_workers=PARAMS['num_workers'], collate_fn=images_collate)

    TEST_DATALOADER = DataLoader(
        test_dataset,batch_size=PARAMS['batch_size'],shuffle=False,
        num_workers=PARAMS['num_workers'], collate_fn=images_collate)
    
    ## init model weights ##
    initialize_weights(model, PARAMS['weights_init'])

    ## start model loop ##
    _, _, best_evalloss, best_epoch, _ = run(PARAMS, model, TRAIN_DATALOADER, 
                                             TEST_DATALOADER, metrics_flag=PARAMS['compute_metrics'])

    del model

    gc.collect()
    torch.cuda.empty_cache()

    return best_evalloss, best_epoch

study = optuna.create_study(directions=['minimize', 'minimize'])
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

joblib.dump(study, f"{BASE_DIR}/optuna_study.pkl")