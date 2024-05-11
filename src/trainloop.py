import torch
from tqdm import tqdm
import numpy as np
import gc
import json
from time import time
import os
from src.metrics import scc, sam, ssim, PSNR
import numpy as np
import yaml

from src.utils import REFORM

def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

def train(model, loader, optimizer, device):
    model.train()
    losses = []
    #process = tqdm(loader)
    for batch in loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch['images'])
        loss = model.compute_loss(output, batch['labels'])
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        #process.set_postfix({
        #    "mode": "train", "avg_loss": np.mean(losses)})

    return losses

def evaluate(model, loader, device, mode='median', metrics_flag=True):
    model.eval()
    losses = []
    metrics = {
        'psnr': [],
        'ssim': [],
        'scc': [],
        'sam': []
    }
    generated_images = None

    #process = tqdm(loader)
    for batch in loader:

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        output = model(batch['images'])
        loss = model.compute_loss(output, batch['labels'])
        losses.append(loss.item())

        #
        predicted_labels = output['generated_image'].detach().cpu()
        target_labels = batch['labels'].detach().cpu()

        if generated_images is None:
            generated_images = output['generated_image'].detach().cpu()
        else:
            generated_images = torch.cat((generated_images, output['generated_image'].detach().cpu()))

        if metrics_flag:
            metrics['psnr'] += [PSNR(target, predicted) for predicted, target in zip(predicted_labels, target_labels)]
            metrics['ssim'] += [ssim(target.numpy(), predicted.numpy(), data_range=1, channel_axis=0) for predicted, target in zip(predicted_labels, target_labels)]
            metrics['scc'] += [scc(target=target, preds=predicted) for predicted, target in zip(predicted_labels, target_labels)]
            metrics['sam'] += sam(target=target_labels, preds=predicted_labels).mean(axis=[1,2]).tolist()

        #process.set_postfix({
        #    "mode": "eval", "avg_loss": np.mean(losses), 
        #    "psnr": np.median(psnr_metrics), "ssim": np.median(ssim_metrics),
        #    "sam": np.median(sam_metrics), 'scc': np.median(scc_metrics)})

    if metrics_flag:
        metrics = {
            'psnr': round(np.median(metrics['psnr']),5) if mode == 'median' else metrics['psnr'], 
            'ssim': round(np.median(metrics['ssim']),5) if mode == 'median' else metrics['ssim'],
            'scc': round(np.median(metrics['scc']), 5) if mode == 'median' else metrics['scc'],
            'sam': round(np.median(metrics['sam']), 5) if mode == 'median' else metrics['sam']}

    return losses, metrics, generated_images

def run(config, model, train_loader, eval_loader, metrics_flag=True):

    print("Model parameters count: ",param_count(model))
    print("Used config: ", config)

    if config['save_model']:
        print("Init folder to save")
        run_dir = f"{config['base_dir']}/experiments/{config['task_name']}"
        if os.path.isdir(run_dir):
            print("Error: Директория существует")
            return
        os.mkdir(run_dir)

        logs_file_path = f'{run_dir}/run_logs.txt'
        path_to_best_model_save = f"{run_dir}/bestmodel.pt"
        path_to_last_model_save = f"{run_dir}/lastmodel.pt"

        print("Init folder for saving generated images from best model")
        gen_images_dir = f'{run_dir}/generated_images'
        os.mkdir(gen_images_dir)

        print("Saving used nn-arch")
        with open(f"{run_dir}/used_arch.txt", 'w', encoding='utf-8') as fd:
            fd.write(model.__str__())

        print("Saving used config")
        with open(f"{run_dir}/used_config.yaml", 'w', encoding='utf-8') as fd:
            yaml.dump(config, fd, default_flow_style=False)

    #print("Init train objectives")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    ml_train = []
    ml_eval = []
    eval_scores = []
    best_evalloss = 10e5
    device = config['device']

    #print("===LEARNING START===")
    for i in tqdm(range(config['epochs'])):

        #print(f"Epoch {i+1} start:")
        train_s = time()
        train_losses = train(model, train_loader, 
                             optimizer, device)
        train_e = time()
        eval_losses, eval_metrics, gen_images = evaluate(model, eval_loader, device, metrics_flag=metrics_flag)
        eval_e = time()
        
        torch.cuda.empty_cache()
        gc.collect()

        #
        ml_train.append(np.mean(train_losses))
        ml_eval.append(np.mean(eval_losses))
        eval_scores.append(eval_metrics)
        #print(f"Epoch {i+1} results: tain_loss - {round(ml_train[-1], 5)} | eval_loss - {round(ml_eval[-1],5)}")
        #print(eval_scores[-1])

        #
        if best_evalloss >= ml_eval[-1]:
            if config['save_model']:
                print("Update Best Model")
                torch.save(model.state_dict(), path_to_best_model_save)

                print("Saving generated images")
                for img_idx in range(gen_images.shape[0]):
                    PIL_image = REFORM(gen_images[img_idx])
                    PIL_image.save(f"{gen_images_dir}/{img_idx}.jpg") 

            #print("Update Best Score")
            best_evalloss = ml_eval[-1]
            best_epoch = i + 1

        # Save train/eval info to logs folder
        if config['save_model']:
            epoch_log = {
                'epoch': i+1, 'train_loss': round(ml_train[-1],5),
                'eval_loss': round(ml_eval[-1],5), 'scores': eval_scores[-1],
                'train_time': round(train_e - train_s, 5), 'eval_time': round(eval_e - train_e, 5)
                }
            with open(logs_file_path,'a',encoding='utf-8') as logfd:
                logfd.write(str(epoch_log) + '\n')

    #print("===LEARNING END===")
    #print("Best scores: ", eval_scores[best_epoch])

    if config['save_model']:
        print("Saving last model")
        torch.save(model.state_dict(), path_to_last_model_save)

    return ml_train, ml_eval, best_evalloss, best_epoch, eval_scores