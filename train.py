#pip install pytorch-lightning wandb

import pytorch_lightning as L
from model.gauss_predictor import GaussPredictor
from model.gauss_predictor_with_latent_codes import GaussPredictorWithLatentCodes
from data_processing.data_module import FaceDataModule
from pytorch_lightning.loggers import WandbLogger
import yaml
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import time
import torch
from tqdm import tqdm
import argparse

def get_run_id_from_time():
    current_unix_time = int(time.time())
    return hex(current_unix_time)[2:]

def main(config):
    run_id = get_run_id_from_time()
    
    wandb_logger = WandbLogger(name=f"{config['logging']['name']} [{run_id}]", id=run_id, entity='3DSSL_', project='3DSSL', log_model=True, config=config)
    face_data_module = FaceDataModule(config=config)
    face_data_module.setup('fit')
    
    if config['model']['model_parameters']['latent_codes']['enabled']:
        model_class = GaussPredictorWithLatentCodes
    else:
        model_class = GaussPredictor
    
    # load or create model
    if config['model']['checkpoint']:
        gauss_predictor = model_class.load_from_checkpoint(config['model']['checkpoint'], config=config, data_module=face_data_module)
    else:
        gauss_predictor = model_class(config=config, data_module=face_data_module)

    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'], 
        accelerator="gpu", 
        devices=1, 
        logger=wandb_logger, 
        default_root_dir='./checkpoints/',
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=150),
            ModelCheckpoint(monitor="val/loss", mode="min", filename="{epoch:02d}", save_top_k=200, save_last=True)
        ]
    )
    
    if not config['data'].get('test_only', False):
        trainer.fit(gauss_predictor, datamodule=face_data_module)
    else:
        trainer.validate(gauss_predictor, datamodule=face_data_module)
        while True:
            time.sleep(1)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config yaml')
    args = parser.parse_args()
    main(yaml.safe_load(open(args.config, 'r')))