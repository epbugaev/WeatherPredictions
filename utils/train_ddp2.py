import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.WeatherGFT import GFT
from utils.dataloader import load_data
# from utils.dataloader_ddp import load_data

import time

import argparse

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--data_root', type=str, default='/home/fratnikov/weather_bench/', help='Data root')

parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
parser.add_argument('--data_split', type=str, default='1_40625', help='Data split')
# parser.add_argument('--data_split', type=str, default='5_625', help='Data split')
parser.add_argument('--data_name', type=str, default='mv_gft', help='Data name')
parser.add_argument('--train_time', type=list, default=['1980', '2015'], help='Train time')
parser.add_argument('--val_time', type=list, default=['2016', '2016'], help='Validation time')
parser.add_argument('--test_time', type=list, default=None, help='Test time')
parser.add_argument('--idx_in', type=list, default=[0], help='Index in')
parser.add_argument('--idx_out', type=list, default=[1, 3, 6], help='Index out')
parser.add_argument('--step', type=int, default=1, help='Step')
parser.add_argument('--levels', type=str, default='all', help='Levels')
parser.add_argument('--distributed', type=bool, default=False, help='Distributed')
parser.add_argument('--use_augment', type=bool, default=False, help='Use augment')
parser.add_argument('--use_prefetcher', type=bool, default=False, help='Use prefetcher')

parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
parser.add_argument('--encoder_layers', type=list, default=[2, 2, 2], help='Encoder layers')
parser.add_argument('--edcoder_heads', type=list, default=[3, 6, 6], help='Encoder heads')
parser.add_argument('--encoder_scaling_factors', type=list, default=[0.5, 0.5, 1], help='Encoder scaling factors')
parser.add_argument('--encoder_dim_factors', type=list, default=[-1, 2, 2], help='Encoder dim factors')
parser.add_argument('--body_layers', type=list, default=[4, 4, 4, 4, 4, 4], help='Body layers')
parser.add_argument('--body_heads', type=list, default=[8, 8, 8, 8, 8, 8], help='Body heads')
parser.add_argument('--body_scaling_factors', type=list, default=[1, 1, 1, 1, 1, 1], help='Body scaling factors')
parser.add_argument('--body_dim_factors', type=list, default=[1, 1, 1, 1, 1, 1], help='Body dim factors')
parser.add_argument('--decoder_layers', type=list, default=[2, 2, 2], help='Decoder layers')
parser.add_argument('--decoder_heads', type=list, default=[6, 6, 3], help='Decoder heads')
parser.add_argument('--decoder_scaling_factors', type=list, default=[1, 2, 1], help='Decoder scaling factors')
parser.add_argument('--decoder_dim_factors', type=list, default=[1, 0.5, 1], help='Decoder dim factors')
parser.add_argument('--channels', type=int, default=69, help='Channels')
parser.add_argument('--head_dim', type=int, default=128, help='Head dim')
parser.add_argument('--window_size', type=list, default=[4,8], help='Window size')
parser.add_argument('--relative_pos_embedding', type=bool, default=False, help='Relative pos embedding')
parser.add_argument('--out_kernel', type=list, default=[2,2], help='Out kernel')
parser.add_argument('--pde_block_depth', type=int, default=3, help='PDE block depth')
parser.add_argument('--block_dt', type=int, default=300, help='Block dt')
parser.add_argument('--inverse_time', type=bool, default=False, help='Inverse time')
parser.add_argument('--drop_last', type=bool, default=True, help='Drop last incomplete batch')

args_for_train = parser.parse_args()



def train(device, model, optimizer, scheduler, dataloader_train, 
          loss_function, save_model_interval=5, model_save_dir='checkpoints/', epochs=3, 
          name_experiment='model_test', start_epoch = 0):
    train_regression_list = []

    grad_list = []

    for epoch_number in range(start_epoch, epochs):
        time_start = time.time()
        train_regression_full = 0
        model.train()
        total_grad = 0
        for batch_idx, (x_train, y_train) in enumerate(dataloader_train):
            x_train = x_train.to(device).squeeze(1)
            y_train = y_train.to(device)
            print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
            if torch.isnan(x_train).any() or torch.isnan(y_train).any():
                print(f"NaN detected in input data! Batch {batch_idx}")
                continue
            optimizer.zero_grad()

            loss = loss_function(model(x_train), y_train)
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            train_regression_full += loss.item()

            scheduler.step()
            for tag, value in model.named_parameters():
                if value.grad is not None:
                    grad = value.grad.norm()
                    total_grad += grad

        grad_list.append(total_grad.cpu().item())
        grad_by_batch = total_grad.cpu().item() / len(dataloader_train)

        train_regression_full = train_regression_full / len(dataloader_train)
        train_regression_list.append(train_regression_full)

        # Save model weights at specified intervals
        if epoch_number % save_model_interval == 0:
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            model_path = os.path.join(model_save_dir, f"{name_experiment}_{epoch_number}.pth")
            state_dict_path = os.path.join(model_save_dir, f"state_dict_{name_experiment}_{epoch_number}.pth")
            
            torch.save(model.state_dict(), model_path)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, state_dict_path)
            
        end_time = time.time()

        print(f"Epoch : {epoch_number}\n",
              f"Time : {round((end_time - time_start), 5)}\n",
              f"Train Pred loss : {round((float(train_regression_full)), 5)}\n",
              f"Grad : {round((float(total_grad)), 5)}\n",
              f"grad_by_batch : {round((float(grad_by_batch)), 5)}\n",
              )

    return train_regression_list


if __name__ == "__main__":

    model = GFT(hidden_dim=args_for_train.hidden_dim,
        encoder_layers=args_for_train.encoder_layers,
        edcoder_heads=args_for_train.edcoder_heads,
        encoder_scaling_factors=args_for_train.encoder_scaling_factors,
        encoder_dim_factors=args_for_train.encoder_dim_factors,

        body_layers=args_for_train.body_layers,
        body_heads=args_for_train.body_heads,
        body_scaling_factors=args_for_train.body_scaling_factors,
        body_dim_factors=args_for_train.body_dim_factors,

        decoder_layers=args_for_train.decoder_layers,
        decoder_heads=args_for_train.decoder_heads,
        decoder_scaling_factors=args_for_train.decoder_scaling_factors,
        decoder_dim_factors=args_for_train.decoder_dim_factors,

        channels=args_for_train.channels,
        head_dim=args_for_train.head_dim,
        window_size=args_for_train.window_size,
        relative_pos_embedding=args_for_train.relative_pos_embedding,
        out_kernel=args_for_train.out_kernel,

        pde_block_depth=args_for_train.pde_block_depth,
        block_dt=args_for_train.block_dt,
        inverse_time=args_for_train.inverse_time)
    
    print(f"Model created")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = args_for_train.batch_size
    data_root = args_for_train.data_root

    dataloader_train, dataloader_vali, dataloader_test, mean, std = load_data(batch_size=batch_size,
                                                                        val_batch_size=batch_size,
                                                                        data_root=data_root,
                                                                        num_workers=args_for_train.num_workers,
                                                                        data_split=args_for_train.data_split,
                                                                        data_name=args_for_train.data_name,
                                                                        train_time=args_for_train.train_time,
                                                                        val_time=args_for_train.val_time,
                                                                        test_time=args_for_train.test_time,
                                                                        idx_in=args_for_train.idx_in,
                                                                        idx_out=args_for_train.idx_out,
                                                                        step=args_for_train.step,
                                                                        levels=args_for_train.levels,
                                                                        distributed=args_for_train.distributed, 
                                                                        use_augment=args_for_train.use_augment,
                                                                        use_prefetcher=args_for_train.use_prefetcher, 
                                                                        drop_last=args_for_train.drop_last)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args_for_train.train_epochs, eta_min=1e-5)
    loss_function = nn.L1Loss()

    train(device, model, optimizer, scheduler, dataloader_train, loss_function, 
          save_model_interval=5, model_save_dir='checkpoints/', epochs=3, 
          name_experiment='model_test', start_epoch = 0)