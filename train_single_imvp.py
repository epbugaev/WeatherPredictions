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
import random
import string
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.weatherbench_128_v3 import WeatherBench128
from Models.imvp_v3 import IAM4VP
from LitModels.mutiout_imvp import MutiOut
from utils.metrics import Metrics

from lightning.pytorch.loggers import CometLogger

def train_model(devices, num_nodes):


    torch_model = IAM4VP(T_data=12, C_data=69, H_data=128, W_data=256, hid_S=64, hid_T=256, N_S=4, N_T=12)
    
    train_start_time = '2000-01-01 00:00:00'
    train_end_time = '2016-12-25 00:00:00' # '2000-01-01 23:00:00' #
    val_start_time = '2017-01-01 00:00:00'
    val_end_time = '2017-12-25 00:00:00' # '2004-01-01 23:00:00' #

    train_data = WeatherBench128(start_time=train_start_time, end_time=train_end_time,
                                include_target=False, 
                                lead_time=1, 
                                interval=1,
                                muti_target_steps=1,
                                start_time_x=0,
                                end_time_x=11,      
                                start_time_y=12,
                                end_time_y=23)  
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)
    valid_data = WeatherBench128(start_time=val_start_time, end_time=val_end_time,
                                include_target=False,
                                lead_time=1, 
                                interval=1,
                                muti_target_steps=1,
                                start_time_x=0,
                                end_time_x=11,      
                                start_time_y=12,
                                end_time_y=23)  
    valid_loader = DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=8)

    world_size=devices*num_nodes
    lr=5e-4
    eta_min=0.0
    max_epoch=20
    steps_per_epoch=len(train_loader)//world_size

    metrics = Metrics(train_data.data_mean_tensor, train_data.data_std_tensor)
    lit_model = MutiOut(torch_model, lr=lr, eta_min=eta_min, max_epoch=max_epoch, steps_per_epoch=steps_per_epoch,
                        loss_type="MAE", metrics=metrics, muti_out_nums=6, time_prediction=6)

    EXP_NAME = "train_imvp_v3_big_model"

    save_path = os.path.join('/home/fa.buzaev/checkpoints/', EXP_NAME, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)))
    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          monitor='val_loss', save_last=True, save_top_k=1, mode="min",
                                          save_on_train_epoch_end=True,
                                          filename='{epoch:02d}-{val_loss:.4f}')
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, check_finite=True)
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    
    os.environ["COMET_API_KEY"] = "3wVTcWCAZ9LFRxOLDsqRDutwt"
    os.environ["COMET_EXPERIMENT_KEY"] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=50))
    comet_ml.login()
    
    trainer = L.Trainer(default_root_dir="./",
                        log_every_n_steps=5,
                        precision=32, # "16-mixed"
                        max_epochs=max_epoch, 
                        logger=CometLogger(project_name="WeatherPredictions", experiment_name=EXP_NAME),
                        accelerator="gpu", devices=devices, num_nodes=num_nodes, strategy=DDPStrategy(static_graph=True),
                        callbacks=[checkpoint_callback, early_stopping_callback])

    trainer.print("[checkpoint path]", save_path)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader, 
    #             ckpt_path='/home/fa.buzaev/checkpoints/train_imvp_v2/2025-04-29-15:38aeboe/last.ckpt')


    trainer.print("train over")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    devices = args.gpus_per_node
    num_nodes = args.nodes
    train_model(devices=devices, num_nodes=num_nodes)