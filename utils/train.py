import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import wandb



def setup_ddp(rank, world_size):
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    except Exception as e:
        print(f"Failed to initialize DDP: {e}")
        raise


def get_warmup_scheduler(optimizer, warmup_steps, learning_rate):
    class WarmupScheduler:
        def __init__(self, optimizer, warmup_steps, initial_lr):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.initial_lr = initial_lr
            self.current_step = 0

        def step(self):
            self.current_step += 1
            if self.current_step <= self.warmup_steps:
                lr = self.initial_lr * (self.current_step / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        def state_dict(self):
            return {
                'current_step': self.current_step,
            }

        def load_state_dict(self, state_dict):
            self.current_step = state_dict['current_step']

    return WarmupScheduler(optimizer, warmup_steps, learning_rate)


def validate(model, val_dataset, criterion, device, config):
    val_sampler = DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers
    )
    
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    # Синхронизация метрик между процессами
    total_loss = torch.tensor(total_loss).to(device)
    total_samples = torch.tensor(total_samples).to(device)
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss.item() / total_samples.item()
    return avg_loss

def train_ddp(rank, world_size, model, train_dataset, val_dataset, config):
    setup_ddp(rank, world_size)
    
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Добавляем scaler для fp16
    scaler = GradScaler()
    
    # Настраиваем warmup и cosine annealing
    warmup_steps = config.warmup_epochs * len(train_loader)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, config.learning_rate)
    
    # Косинусный scheduler начнет работать после warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(config.epochs - config.warmup_epochs) * len(train_loader),
        eta_min=1e-7
    )
    
    # Инициализируем wandb только для основного процесса
    if rank == 0:
        wandb.init(
            project="weather-predictions",
            config={
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "warmup_epochs": config.warmup_epochs,
            }
        )
    
    for epoch in range(config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(rank)
            targets = targets.to(rank)
            
            optimizer.zero_grad()
            
            # Используем fp16
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Обратное распространение с fp16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Обновляем learning rate
            if epoch < config.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % config.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    "batch": batch_idx,
                })
        
        if rank == 0:
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = validate(model, val_dataset, criterion, rank, config)
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            })
    
    if rank == 0:
        wandb.finish()
    
    dist.destroy_process_group()


# Обновляем конфигурацию
class Config:
    batch_size = 32
    epochs = 50 
    learning_rate = 5e-4
    num_workers = 4
    log_interval = 10
    warmup_epochs = 5  # Добавлен параметр warmup
