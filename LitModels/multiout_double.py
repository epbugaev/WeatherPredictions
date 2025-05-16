import torch
import lightning as L
from utils.metrics import Metrics
import numpy as np
from torch import nn
from LitModels.basemodel_no_history import BaseModel
import matplotlib.pyplot as plt 

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

        hours = [0, 2, 5]

        y_hat = self.model(x)[:, hours, ...]
        y = y[:, hours, ...]

        loss = self.loss(y_hat, y)
        # self.log("train_loss", loss, prog_bar=True)
        lr_now = self.the_optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": loss, "lr": lr_now}

        self.log_dict(log_dict, prog_bar=True)

        self.trained = True
        return loss

    
    def validation_step(self, batch, batch_ids=None):
        x, y = batch

        y_hat = self.model(x)
        y_hat_second = self.model(y_hat[:, -1, ...])
        y_hat = torch.concat([y_hat, y_hat_second], axis=1) # Autoregressively predict for hours 1, 3, 6

        val_loss = self.loss(y_hat, y)
        rmse_first = self.metrics.WRMSE(y_hat[:,0], y[:,0])
        rmse_last = self.metrics.WRMSE(y_hat[:,-1], y[:,-1])

        log_dict = {"val_loss": val_loss, "RMSE_z500_first": rmse_first[11], "RMSE_z500_last": rmse_last[11], 
                    "RMSE_t500_first": rmse_first[11 + 13], "RMSE_t500_last": rmse_last[11 + 13], 
                    "RMSE_u500_first": rmse_first[11 + 26], "RMSE_u500_last": rmse_last[11 + 26], 
                    "RMSE_v500_first": rmse_first[11 + 39], "RMSE_v500_last": rmse_last[11 + 39]}

        # Определение индексов для различных переменных
        # На основе порядка в weatherbench_128_v2.py
        index_map = {
            'u10': 1,      # 10m_u_component_of_wind
            'v10': 2,      # 10m_v_component_of_wind
            't2': 0,       # 2m_temperature
            'z500': 4 + 7,    # geopotential на уровне 500 hPa
            't500': 17 + 7,    # temperature на уровне 500 hPa
            't50': 17 + 0,     # temperature на уровне 50 hPa
            't1000': 17 + 12,   # temperature на уровне 1000 hPa
            'u500': 43 + 7,    # u_component_of_wind на уровне 500 hPa
            'v500': 56 + 7,    # v_component_of_wind на уровне 500 hPa
            'u50': 43 + 0,     # u_component_of_wind на уровне 50 hPa
            'v50': 56 + 0,     # v_component_of_wind на уровне 50 hPa
            'u1000': 43 + 12,   # u_component_of_wind на уровне 1000 hPa
            'v1000': 56 + 12,   # v_component_of_wind на уровне 1000 hPa
        }
        
        log_dict = {
            "val_loss": val_loss,
        }
        
        # Добавляем логирование для первого и последнего предсказания
        for var_name, idx in index_map.items():
            log_dict[f"f RMSE_{var_name}_first"] = rmse_first[idx]
            log_dict[f"f RMSE_{var_name}_last"] = rmse_last[idx]
        
        self.log_dict(log_dict, prog_bar=True)
        
        # Добавляем визуализации в comet для указанных переменных
        if self.trained and self.logger is not None and hasattr(self.logger, "experiment"):
            print('HERE!!!')

            # Список переменных для визуализации
            vis_vars = ['u50', 'v50', 'z500', 'u500', 'v500']
            
            # Визуализируем только последний прогноз
            last_pred = y_hat[:, -1, ...]
            last_true = y[:, -1]
            
            for var_name in vis_vars:
                if var_name in index_map:
                    var_idx = index_map[var_name]
                    
                    # Берем первый образец из батча
                    pred_map = last_pred[0, var_idx].cpu().numpy()
                    true_map = last_true[0, var_idx].cpu().numpy()
                    diff_map = pred_map - true_map
                    
                    # Создаем и логируем изображения
                    fig_pred, ax_pred = plt.subplots(figsize=(10, 8))
                    im_pred = ax_pred.imshow(pred_map, cmap='viridis')
                    plt.colorbar(im_pred, ax=ax_pred)
                    ax_pred.set_title(f'{var_name} Prediction')
                    self.logger.experiment.log_figure(figure=fig_pred, figure_name=f'{var_name}_prediction')
                    plt.close(fig_pred)
                    
                    fig_true, ax_true = plt.subplots(figsize=(10, 8))
                    im_true = ax_true.imshow(true_map, cmap='viridis')
                    plt.colorbar(im_true, ax=ax_true)
                    ax_true.set_title(f'{var_name} Ground Truth')
                    self.logger.experiment.log_figure(figure=fig_true, figure_name=f'{var_name}_true')
                    plt.close(fig_true)
                    
                    fig_diff, ax_diff = plt.subplots(figsize=(10, 8))
                    im_diff = ax_diff.imshow(diff_map, cmap='RdBu_r')
                    plt.colorbar(im_diff, ax=ax_diff)
                    ax_diff.set_title(f'{var_name} Prediction - True')
                    self.logger.experiment.log_figure(figure=fig_diff, figure_name=f'{var_name}_diff')
                    plt.close(fig_diff)
        
        self.log_dict(log_dict, prog_bar=True)
        self.trained = False