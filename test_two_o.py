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
from Models.PredFormerTwoO import PredFormer_Model
# from Models.FedorPredFormerGFT import PredFormer_Model
from LitModels.mutiout_fedor import MutiOut
from utils.metrics import Metrics

from lightning.pytorch.loggers import CometLogger


model_config = {
    # image h w c
    'height': 32,
    'width': 64,
    'num_channels': 69,
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 8,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 4,
    # depth
    'depth': 1,
    'Ndepth': 24,
    'path_to_constants': '/home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc',
}

torch_model = PredFormer_Model(model_config)
train_start_time = '2000-01-01 00:00:00'
train_end_time = '2003-12-25 00:00:00' # '2000-01-01 23:00:00' #
val_start_time = '2004-01-01 00:00:00'
val_end_time = '2004-12-25 00:00:00' # '2004-01-01 23:00:00' #

train_data = WeatherBench128(start_time=train_start_time, end_time=train_end_time,
                            include_target=False, lead_time=1, interval=12, muti_target_steps=12, cut=[[128 - 92, 128 - 60], [256 - 131, 256 - 67]])
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
valid_data = WeatherBench128(start_time=val_start_time, end_time=val_end_time,
                            include_target=False, lead_time=1, interval=12, muti_target_steps=12, cut=[[128 - 92, 128 - 60], [256 - 131, 256 - 67]])
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False, num_workers=4)

x, y = next(iter(train_loader))
print('next data (x, y) shapes', x.shape, y.shape)