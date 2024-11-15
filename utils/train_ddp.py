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

def train(rank, world_size, warmup_epochs=5, train_epochs=50):
    # logger = setup_logging(rank)
    setup(rank, world_size)


    batch_size = 8
    data_root = '/home/fratnikov/weather_bench/'

    dataloader_train, dataloader_vali, dataloader_test, mean, std = load_data(batch_size=batch_size,
                                                                        val_batch_size=batch_size,
                                                                        data_root=data_root,
                                                                        num_workers=10,
                                                                        data_split='1_40625',
                                                                        # data_split='5_625',
                                                                        data_name='mv_gft',
                                                                        # train_time=['2015', '2018'],
                                                                        train_time=['2015', '2015'],
                                                                        # val_time=['2018', '2018'],
                                                                        # test_time=['2018', '2018'],
                                                                        # val_time=None,
                                                                        test_time=None,
                                                                        idx_in=[0],
                                                                        idx_out=[1, 3, 6],
                                                                        step=1,
                                                                        levels='all', 
                                                                        distributed=True, use_augment=False,
                                                                        use_prefetcher=False, drop_last=False)
    
    model = GFT(hidden_dim=256,
            encoder_layers=[2, 2, 2],
            edcoder_heads=[3, 6, 6],
            encoder_scaling_factors=[0.5, 0.5, 1], # [128, 256] --> [64, 128] --> [32, 64] --> [32, 64], that is, patch size = 4 (128/32)
            encoder_dim_factors=[-1, 2, 2],

            body_layers=[4, 4, 4, 4, 4, 4], # A total of 4x6=24 HybridBlock, corresponding to 6 hours (24x15min) of time evolution
            body_heads=[8, 8, 8, 8, 8, 8],
            body_scaling_factors=[1, 1, 1, 1, 1, 1],
            body_dim_factors=[1, 1, 1, 1, 1, 1],

            decoder_layers=[2, 2, 2],
            decoder_heads=[6, 6, 3],
            decoder_scaling_factors=[1, 2, 1],
            decoder_dim_factors=[1, 0.5, 1],

            channels=69,
            head_dim=128,
            window_size=[4,8],
            relative_pos_embedding=False,
            out_kernel=[2,2],

            pde_block_depth=3, # 1 HybridBlock contains 3 PDE kernels, corresponding to 15 minutes (3x300s) of time evolution
            block_dt=300, # One PDE kernel corresponds to 300s of time evolution
            inverse_time=False)
    
    print(f"Model created")

    # Инициализация DataFrame для сохранения лоссов
    if rank == 0:
        losses_df = pd.DataFrame(columns=['epoch', 'batch', 'loss', 'lr'])
    
    # Перемещаем модель на GPU
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Оборачиваем модель в DDP
    model = DDP(model, device_ids=[rank])
    print(f"Model wrapped in DDP")
    
    # Оптимизатор и планировщики
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Вычисляем количество шагов на эпоху для планировщика
    steps_per_epoch = len(dataloader_train)
    
    # Создаем два планировщика
    warmup_scheduler = get_warmup_scheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_epochs * steps_per_epoch,
        eta_min=0
    )
    
    # Для работы с FP16
    scaler = GradScaler()
    
    # лосс
    criterion = torch.nn.L1Loss()
    print(f"Criterion set to L1Loss")
    
    global_step = 0
    total_epochs = warmup_epochs + train_epochs
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        print(f"Epoch {epoch} started")
        
        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(f"Inputs and targets moved to device")
            optimizer.zero_grad()
            print(f"Optimizer zero grad")
            # Используем autocast для FP16
            with autocast():
                outputs = model(inputs)
                print(f"Outputs computed")
                loss = criterion(outputs, targets)
                print(f"Loss computed")
            
            # Обратное распространение с FP16
            print(f"Loss: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Собираем статистику со всех процессов
            loss_item = loss.item()
            dist.all_reduce(torch.tensor([loss_item]).to(device))
            total_loss += loss_item
            
            # Сохраняем статистику
            if rank == 0:
                losses_df = losses_df.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss_item,
                    'lr': optimizer.param_groups[0]['lr']
                }, ignore_index=True)
            
            # Выбираем планировщик в зависимости от эпохи
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            global_step += 1
        
        # Сохраняем CSV после каждой эпохи
        if rank == 0:
            losses_df.to_csv(f'training_losses.csv', index=False)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(dataloader_train)
        
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
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )