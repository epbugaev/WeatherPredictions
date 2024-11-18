import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import logging
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import time
import sys

import pandas as pd
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.WeatherGFT import GFT
from utils.dataloader import load_data
# from utils.dataloader_ddp import load_data

import argparse

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--data_root', type=str, default='/home/fratnikov/weather_bench/', help='Data root')

parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
# parser.add_argument('--data_split', type=str, default='1_40625', help='Data split')
parser.add_argument('--data_split', type=str, default='5_625', help='Data split')
parser.add_argument('--data_name', type=str, default='mv_gft', help='Data name')
parser.add_argument('--train_time', type=list, default=['2015', '2015'], help='Train time')
parser.add_argument('--val_time', type=list, default=None, help='Validation time')
parser.add_argument('--test_time', type=list, default=None, help='Test time')
parser.add_argument('--idx_in', type=list, default=[0], help='Index in')
parser.add_argument('--idx_out', type=list, default=[1, 3, 6], help='Index out')
parser.add_argument('--step', type=int, default=1, help='Step')
parser.add_argument('--levels', type=str, default='all', help='Levels')
parser.add_argument('--distributed', type=bool, default=True, help='Distributed')
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



def setup_logging(rank):
    """Настройка логирования"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'training_rank_{rank}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup(rank, world_size):
    """Инициализация DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Очистка DDP"""
    dist.destroy_process_group()

def get_warmup_scheduler(optimizer, warmup_epochs, steps_per_epoch, last_epoch=-1):
    """Создание планировщика только с warmup"""
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        return float(step) / float(max(1, warmup_steps))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def validate(model, dataloader, criterion, device, epoch, rank):
    """Функция валидации модели"""
    model.eval()
    total_loss = 0
    val_losses_df = pd.DataFrame(columns=['epoch', 'batch', 'loss'])
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Собираем статистику со всех процессов
            loss_item = loss.item()
            dist.all_reduce(torch.tensor([loss_item]).to(device))
            total_loss += loss_item
            
            # Сохраняем статистику
            if rank == 0:
                val_losses_df = pd.concat([val_losses_df, pd.DataFrame([{
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss_item,
                }])], ignore_index=True)
    
    avg_loss = total_loss / len(dataloader)
    
    # Сохраняем CSV с результатами валидации
    if rank == 0:
        val_losses_df.to_csv(f'validation_losses_epoch_{epoch}.csv', index=False)
    
    return avg_loss

def train(rank, world_size, args_for_train):
    # logger = setup_logging(rank)
    setup(rank, world_size)
    
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
    

    print(f"Dataloaders created")
    x_train, y_train = next(iter(dataloader_train))
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")


    if rank == 0:
        losses_df = pd.DataFrame(columns=['epoch', 'batch', 'loss', 'lr'])
    
    # Перемещаем модель на GPU
    # device = torch.device(f"cuda:{rank}")
    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    print(f"Model moved to device: {device_id}")
    
    # Оборачиваем модель в DDP
    model = DDP(model, device_ids=[device_id])
    print(f"Model wrapped in DDP")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    steps_per_epoch = len(dataloader_train)
    
    warmup_scheduler = get_warmup_scheduler(
        optimizer, 
        warmup_epochs=args_for_train.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args_for_train.train_epochs * steps_per_epoch,
        eta_min=0
    )
    
    # Для работы с FP16
    scaler = GradScaler()
    
    # лосс
    criterion = torch.nn.L1Loss()
    print(f"Criterion set to L1Loss")
    
    global_step = 0
    total_epochs = args_for_train.warmup_epochs + args_for_train.train_epochs
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        print(f"Epoch {epoch} started")
        
        avg_loss = float('inf')
        
        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            inputs = inputs.to(device_id).squeeze(1)
            targets = targets.to(device_id)


            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                print(f"NaN detected in input data! Batch {batch_idx}")
                continue

            print(f"Inputs and targets moved to device")
            optimizer.zero_grad()
            print(f"Optimizer zero grad")

            with autocast(dtype=torch.float32):
                outputs = model(inputs)

                if torch.isnan(outputs).any():
                    print(f"NaN detected in model outputs! Batch {batch_idx}")
                    continue

                print(f"Outputs computed")
                loss = criterion(outputs, targets)
                print(f"Loss computed")
            
            print(f"Loss: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Собираем статистику со всех процессов
            loss_item = loss.item()
            dist.all_reduce(torch.tensor([loss_item]).to(device_id))
            total_loss += loss_item

            # Вычисляем среднюю потерю после каждого батча
            avg_loss = total_loss / (batch_idx + 1)  # Изменено с len(dataloader_train)
            epoch_time = time.time() - epoch_start_time

            # Сохраняем статистику
            if rank == 0:
                losses_df = pd.concat([losses_df, pd.DataFrame([{
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss_item,
                    'avg_loss': avg_loss,
                    'epoch_time': epoch_time,
                    'lr': optimizer.param_groups[0]['lr']
                }])], ignore_index=True)
            
            # Выбираем планировщик в зависимости от эпохи
            if epoch < args_for_train.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            global_step += 1
                # Сохраняем CSV после каждой эпохи
        if rank == 0:
            losses_df.to_csv(f'training_losses.csv', index=False)
        
        # Сохранение чекпоинта каждые 5 эпох
        if rank == 0 and epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f'Сохранен чекпоинт: {checkpoint_path}')
        
        if dataloader_vali is not None:
            # Добавляем валидацию каждые 5 эпох
            if epoch % 5 == 0:
                val_loss = validate(model, dataloader_vali, criterion, device_id, epoch, rank)
                if rank == 0:
                    print(f'Epoch {epoch}, Validation Loss: {val_loss:.6f}')
                
                    # Добавляем результаты валидации в основной DataFrame
                    if rank == 0:
                        losses_df = pd.concat([losses_df, pd.DataFrame([{
                            'epoch': epoch,
                            'batch': 'validation',
                            'loss': val_loss,
                            'lr': optimizer.param_groups[0]['lr']
                        }])], ignore_index=True)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    torch.multiprocessing.spawn(
        train,
        args=(world_size,args_for_train),
        nprocs=world_size,
        join=True
    )