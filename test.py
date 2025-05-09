import os
import torch
from torch import nn, einsum
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
from Models.imvp_v3 import IAM4VP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

x_train = torch.randn(2, 12, 69, 128, 256).to(device)
model = IAM4VP().to(device)

# Calculate and print the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params:,}")

# Optionally, print parameters by module
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: {params:,} parameters ({params/total_params:.2%} of total)")


pred_list = [torch.randn(2, 69, 128, 256).to(device)]
with torch.no_grad():
    t = torch.tensor(1 * 100).repeat(x_train.shape[0]).to(device)
    y_pred = model(x_train, pred_list, t)
    print(y_pred.shape)