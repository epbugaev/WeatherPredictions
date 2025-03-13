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
        self.out_idx = self.model.out_layer


    def training_step(self, batch):
        x, y = batch
        
        y = y[:, self.out_idx]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # self.log("train_loss", loss, prog_bar=True)
        lr_now = self.the_optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": loss, "lr": lr_now}

        learnable_coef = self.model.get_learnable_physics_importance()
        if isinstance(learnable_coef, float):
            log_dict['average_physics_coef'] = learnable_coef
        else:
            log_dict['average_physics_coef'] = torch.mean(learnable_coef)
            log_dict['coef_z_500'] = torch.mean(learnable_coef[:, 7, :, :]) # 7 is pressure 500, first 13 are for z, [layers, C, H, W]
            log_dict['coef_t_500'] = torch.mean(learnable_coef[:, 13 + 7, :, :]) # 7 is pressure 500, second 13 are for t, [layers, C, H, W]

        self.log_dict(log_dict, prog_bar=True)
        return loss

    
    def validation_step(self, batch):
        x, y = batch
        y = y[:, self.out_idx]
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        rmse_first = self.metrics.WRMSE(y_hat[:,0], y[:,0])
        rmse_last = self.metrics.WRMSE(y_hat[:,-1], y[:,-1])

        log_dict = {"val_loss": val_loss, "RMSE_z500_first": rmse_first[11], "RMSE_z500_last": rmse_last[11]}

        learnable_coef = self.model.get_learnable_physics_importance()
        if isinstance(learnable_coef, float):
            log_dict['average_physics_coef'] = learnable_coef
        else:
            log_dict['average_physics_coef'] = torch.mean(learnable_coef)
            log_dict['coef_z_500'] = torch.mean(learnable_coef[:, 7, :, :]) # 7 is pressure 500, first 13 are for z, [layers, C, H, W]
            log_dict['coef_t_500'] = torch.mean(learnable_coef[:, 13 + 7, :, :]) # 7 is pressure 500, second 13 are for t, [layers, C, H, W]
        
        self.log_dict(log_dict, prog_bar=True)