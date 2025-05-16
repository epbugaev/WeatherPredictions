import torch
import lightning as L
from utils.metrics import Metrics
import numpy as np
from torch import nn
from LitModels.basemodel import BaseModel
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
                 time_prediction:int=6,  # Number of time steps to predict
                 **kwargs):
        super().__init__(model, lr, eta_min, max_epoch, steps_per_epoch, loss_type, metrics, muti_steps)
        self.example_input_array = torch.Tensor(1, 6, 69, 32, 64)
        self.time_prediction = time_prediction
        self.automatic_optimization = False  # Disable automatic optimization for manual optimization

    

    def forward(self, x, y_raw=None, t=None):
        if y_raw is None:
            y_raw = []  # Empty list as default
            t = torch.zeros(x.shape[0], device=x.device)  # Default timestamp of zeros
        return self.model(x, y_raw, t)


    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        total_loss = 0
        pred_list = []
        
        for idx_time in range(self.time_prediction):
            # Generate timestep tensor
            t = torch.tensor((idx_time + 1) * 100).repeat(x.shape[0]).to(self.device)
            
            # Make prediction using current input and previous predictions
            prediction = self.model(x, pred_list, t)
            
            # Store prediction for next iteration
            pred_list.append(prediction.detach())
            
            # Calculate loss for this timestep
            step_loss = self.loss(prediction, y[:, idx_time])
            total_loss += step_loss
            
            # Manual backpropagation for this step
            self.manual_backward(step_loss)
            
        # Update weights after all timesteps
        optimizer.step()
        
        # Log metrics
        lr_now = optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": total_loss / self.time_prediction, "lr": lr_now}
        self.log_dict(log_dict, prog_bar=True)
        
        return total_loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        total_loss = 0
        pred_list = []
        
        # Perform iterative prediction
        for idx_time in range(self.time_prediction):
            t = torch.tensor((idx_time + 1) * 100).repeat(x.shape[0]).to(self.device)
            prediction = self.model(x, pred_list, t)
            pred_list.append(prediction.detach())
            
            # Calculate loss for this timestep
            step_loss = self.loss(prediction, y[:, idx_time])
            total_loss += step_loss
        
        # Calculate average loss across all timesteps
        val_loss = total_loss / self.time_prediction
        
        # Calculate RMSE metrics for first and last prediction
        rmse_first = self.metrics.WRMSE(pred_list[0], y[:, 0])
        rmse_last = self.metrics.WRMSE(pred_list[-1], y[:, -1])
        
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
            last_pred = pred_list[-1]
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