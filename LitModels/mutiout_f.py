import torch
import lightning as L
from utils.metrics import Metrics
import numpy as np
from torch import nn
from LitModels.basemodel import BaseModel

class MutiOut(BaseModel):
    def __init__(self, 
                 model:nn.Module=nn.Identity(), 
                 # training
                 lr=0.0001, 
                 eta_min=0.0, 
                 max_epoch=10, 
                 steps_per_epoch=100, 
                 loss_type:str="MAE",
                 metrics:Metrics=Metrics(),
                 # testing
                 muti_steps:int=1,
                 **kwargs):
        super().__init__(model, lr, eta_min, max_epoch, steps_per_epoch, loss_type, metrics, muti_steps)


    def training_step(self, batch):
        x, y = batch
        
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # self.log("train_loss", loss, prog_bar=True)
        lr_now = self.the_optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": loss, "lr": lr_now}

        self.log_dict(log_dict, prog_bar=True)
        return loss

    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        rmse_first = self.metrics.WRMSE(y_hat[:,0], y[:,0])
        rmse_last = self.metrics.WRMSE(y_hat[:,-1], y[:,-1])

        # Определение индексов для различных переменных
        # На основе порядка в weatherbench_128_v2.py
        index_map = {
            'u10': 1,      # 10m_u_component_of_wind
            'v10': 2,      # 10m_v_component_of_wind
            't2': 0,       # 2m_temperature
            'z500': 11,    # geopotential на уровне 500 hPa
            't500': 20,    # temperature на уровне 500 hPa
            't50': 23,     # temperature на уровне 50 hPa
            't1000': 19,   # temperature на уровне 1000 hPa
            'u500': 32,    # u_component_of_wind на уровне 500 hPa
            'v500': 45,    # v_component_of_wind на уровне 500 hPa
            'u50': 35,     # u_component_of_wind на уровне 50 hPa
            'v50': 48,     # v_component_of_wind на уровне 50 hPa
            'u1000': 31,   # u_component_of_wind на уровне 1000 hPa
            'v1000': 44,   # v_component_of_wind на уровне 1000 hPa
        }
        
        log_dict = {
            "val_loss": val_loss,
        }
        
        # Добавляем логирование для первого и последнего предсказания
        for var_name, idx in index_map.items():
            log_dict[f"RMSE_{var_name}_first"] = rmse_first[idx]
            log_dict[f"RMSE_{var_name}_last"] = rmse_last[idx]
        
        self.log_dict(log_dict, prog_bar=True)