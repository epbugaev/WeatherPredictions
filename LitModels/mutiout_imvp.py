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
                 time_prediction:int=12,  # Number of time steps to predict
                 **kwargs):
        super().__init__(model, lr, eta_min, max_epoch, steps_per_epoch, loss_type, metrics, muti_steps)
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
        
        log_dict = {
            "val_loss": val_loss, 
            "RMSE_z500_first": rmse_first[11], 
            "RMSE_z500_last": rmse_last[11]
        }
        
        self.log_dict(log_dict, prog_bar=True)
        
        return val_loss