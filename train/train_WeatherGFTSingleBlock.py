import comet_ml

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import datetime
import os
import sys
import wandb
import random
import string

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.weatherbench_128 import WeatherBench128
from Models.WeatherGFTSingle import GFT

from LitModels.mutiout import MutiOut
from utils.metrics import Metrics

from lightning.pytorch.loggers import CometLogger

def train_model(devices, num_nodes):    
    torch_model = GFT(hidden_dim=256,
                    physics_part_coef=0.1, # None means using learnable matrix C x H x W
                    encoder_layers=[2, 2, 2], # original: [3, 3, 3]
                    edcoder_heads=[2, 4, 4], # original: [3, 6, 6]
                    encoder_scaling_factors=[0.5, 0.5, 1], # [128, 256] --> [64, 128] --> [32, 64] --> [32, 64], that is, patch size = 4 (128/32)
                    encoder_dim_factors=[-1, 2, 2],

                    body_layers=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # A total of 4x6=24 HybridBlock, corresponding to 6 hours (24x15min) of time evolution
                    body_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], # original: [8, 8, 8, 8, 8, 8]
                    body_scaling_factors=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    body_dim_factors=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                    decoder_layers=[2, 2, 2], # original: [3, 3, 3]
                    decoder_heads=[4, 4, 2], # original: [6, 6, 3]
                    decoder_scaling_factors=[1, 2, 1],
                    decoder_dim_factors=[1, 0.5, 1],

                    channels=69,
                    head_dim=128,
                    window_size=[4,8],
                    relative_pos_embedding=False,
                    out_kernel=[2,2],
                    
                    pde_block_depth=3, # 1 HybridBlock contains 3 PDE kernels, corresponding to 15 minutes (3x300s) of time evolution
                    block_dt=300, # One PDE kernel corresponds to 300s of time evolution
                    inverse_time=False, 
                    use_checkpoint=True)
    
    train_start_time = '2000-01-01 00:00:00'
    train_end_time = '2003-12-25 00:00:00' # '2000-01-01 23:00:00' #
    val_start_time = '2004-01-01 00:00:00'
    val_end_time = '2004-12-25 00:00:00' # '2004-01-01 23:00:00' #

    train_data = WeatherBench128(start_time=train_start_time, end_time=train_end_time,
                                include_target=False, lead_time=1, interval=5, muti_target_steps=6)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    valid_data = WeatherBench128(start_time=val_start_time, end_time=val_end_time,
                                include_target=False, lead_time=1, interval=1, muti_target_steps=12)
    valid_loader = DataLoader(valid_data, batch_size=2, shuffle=False, num_workers=4)

    world_size=devices*num_nodes
    lr=1e-4
    eta_min=0.0
    max_epoch=20
    steps_per_epoch=len(train_loader)//world_size

    metrics = Metrics(train_data.data_mean_tensor, train_data.data_std_tensor)
    lit_model = MutiOut(torch_model, lr=lr, eta_min=eta_min, max_epoch=max_epoch, steps_per_epoch=steps_per_epoch,
                        loss_type="MAE", metrics=metrics, muti_out_nums=6)

    EXP_NAME = "train_single_block full world"

    save_path = os.path.join('/home/ebugaev/checkpoints/', EXP_NAME, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)))
    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          monitor='val_loss', save_last=True, save_top_k=1, mode="min",
                                          save_on_train_epoch_end=True,
                                          filename='{epoch:02d}-{val_loss:.4f}')
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, check_finite=True)
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    
    os.environ["COMET_API_KEY"] = "D75wgJ5A8n5yvnTcrdgLGpuYy"
    os.environ["COMET_EXPERIMENT_KEY"] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=50))
    comet_ml.login()

    logger = CometLogger(project_name="WeatherPredictions", experiment_name=EXP_NAME)
    logger.experiment.log_code(file_name='/home/ebugaev/WeatherPredictions/Models/WeatherGFTSingle.py')

    trainer = L.Trainer(default_root_dir="./",
                        log_every_n_steps=5,
                        precision=32, # "16-mixed"
                        max_epochs=max_epoch, 
                        logger=logger,
                        accelerator="gpu", devices=devices, num_nodes=num_nodes, strategy=DDPStrategy(static_graph=True),
                        callbacks=[checkpoint_callback, early_stopping_callback])

    trainer.print("[checkpoint path]", save_path)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    trainer.print("train over")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    devices = args.gpus_per_node
    num_nodes = args.nodes
    train_model(devices=devices, num_nodes=num_nodes)