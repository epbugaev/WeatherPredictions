import torch
import lightning as L
from WeatherPredictions.utils.metrics import Metrics
import numpy as np
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity
from LitModels.basemodel_predrnn import BaseModel
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
        #self.out_idx = self.model.out_layer


    def training_step(self, batch):
        x, y = batch

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        #logger.debug(f'x, y shapes: {x.shape} {y.shape}')
        inp = torch.cat([
            x, 
            y
        ], dim=1)
        
        inp = inp.permute(0, 1, 3, 4, 2).contiguous()
        #print('inp shape in training:', inp.shape)
        y_hat, _ = self.model(inp, torch.zeros(1, 12, 1, 1, 1).to(x.device))
        y_hat = y_hat.permute(0, 1, 4, 2, 3)
        inp = inp.permute(0, 1, 4, 2, 3)
        #print('final shape:', y_hat.shape)

        #print('Forward passed succesfully')
        loss = self.loss(inp[:, 1:, ...], y_hat)

        #print('Calculated loss:', loss)

        # self.log("train_loss", loss, prog_bar=True)
        lr_now = self.the_optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": loss, "lr": lr_now}

        self.log_dict(log_dict, prog_bar=True)
        return loss

    
    def validation_step(self, batch):
        x, y = batch

        inp = torch.cat([
            x, 
            y
        ], dim=1)
        
        inp = inp.permute(0, 1, 3, 4, 2).contiguous()
        #print('inp shape in val:', inp.shape)
        y_hat, _ = self.model(inp, torch.zeros(1, 12, 1, 1, 1).to(x.device))
        y_hat = y_hat.permute(0, 1, 4, 2, 3)[:, 11:, ...]
    
        val_loss = self.loss(y_hat, y)
        rmse_first = self.metrics.WRMSE(y_hat[:,0], y[:,0])
        rmse_last = self.metrics.WRMSE(y_hat[:,-1], y[:,-1])

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
        
        return val_loss