import comet_ml

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import datetime
import os
import sys
import random
import string
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.dataloader_weather import WeatherBenchDataset2 
from Data.weatherbench_128_v2 import WeatherBench128
from Models.imvp_v2 import IAM4VP
from LitModels.mutiout_imvp import MutiOut
from utils.metrics import Metrics

from lightning.pytorch.loggers import CometLogger


train_data = WeatherBenchDataset2(data_root='/home/fratnikov/weather_bench/1.40625deg',
                            data_name='gft',
                            training_time=['2015', '2015'],
                            idx_in=[*range(0, 6)],
                            idx_out=[*range(6, 12)],
                            step=120,
                            levels='all',
                            data_split='1_40625',
                            use_augment=False)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

x_train, y_train = next(iter(train_loader))
print(x_train.shape, y_train.shape)
print("train_data.std.shape", train_data.std.shape)

def prepare_mean_or_std(data):
    """
    Подготавливает стандартное отклонение для использования в модели.
    
    Args:
        std_data (numpy.ndarray): Исходные данные стандартного отклонения
        
    Returns:
        torch.Tensor: Подготовленный тензор стандартного отклонения
    """
    data = torch.from_numpy(data).squeeze()
    
    # Разделяем первые 4 канала на 5 частей
    z_old, t_old, q_old, u_old, v_old = data[4:].chunk(5, dim=0)
    
    # Повторяем каждую часть 13 раз
    z_old = z_old.repeat(13)
    t_old = t_old.repeat(13)
    q_old = q_old.repeat(13)
    u_old = u_old.repeat(13)
    v_old = v_old.repeat(13)
    
    # Объединяем все части в один тензор
    return torch.cat([data[:4], z_old, t_old, q_old, u_old, v_old], dim=0)

train_data_std = prepare_mean_or_std(train_data.std)
train_data_mean = prepare_mean_or_std(train_data.mean)

# train_data.std = train_data.std[:4] + 
metrics = Metrics(train_data_mean, train_data_std)

train_start_time = '2000-01-01 00:00:00'
train_end_time = '2003-12-25 00:00:00' # '2000-01-01 23:00:00' #
val_start_time = '2004-01-01 00:00:00'
val_end_time = '2004-12-25 00:00:00' # '2004-01-01 23:00:00' #

train_data = WeatherBench128(start_time=train_start_time, end_time=train_end_time,
                            include_target=False, 
                            lead_time=1, 
                            interval=120,
                            muti_target_steps=1,
                            start_time_x=0,
                            end_time_x=5,      
                            start_time_y=6,
                            end_time_y=11)  


print("train_data.data_std_tensor", train_data.data_std_tensor)
print("train_data.data_std_tensor", train_data.data_std_tensor.shape)


#     world_size=devices*num_nodes
#     lr=5e-4
#     eta_min=0.0
#     max_epoch=20
#     steps_per_epoch=len(train_loader)//world_size

#     metrics = Metrics(train_data.mean, train_data.std)
#     lit_model = MutiOut(torch_model, lr=lr, eta_min=eta_min, max_epoch=max_epoch, steps_per_epoch=steps_per_epoch,
#                         loss_type="MAE", metrics=metrics, muti_out_nums=6, time_prediction=6)

#     EXP_NAME = "train_imvp_mini_gft_v2"

#     save_path = os.path.join('/home/fa.buzaev/checkpoints/', EXP_NAME, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)))
#     checkpoint_callback = ModelCheckpoint(dirpath=save_path,
#                                           monitor='val_loss', save_last=True, save_top_k=1, mode="min",
#                                           save_on_train_epoch_end=True,
#                                           filename='{epoch:02d}-{val_loss:.4f}')
#     early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, check_finite=True)
#     # lr_monitor = LearningRateMonitor(logging_interval='step')
    
#     os.environ["COMET_API_KEY"] = "3wVTcWCAZ9LFRxOLDsqRDutwt"
#     os.environ["COMET_EXPERIMENT_KEY"] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=50))
#     comet_ml.login()
    
#     trainer = L.Trainer(default_root_dir="./",
#                         log_every_n_steps=5,
#                         precision=32, # "16-mixed"
#                         max_epochs=max_epoch, 
#                         logger=CometLogger(project_name="WeatherPredictions", experiment_name=EXP_NAME),
#                         accelerator="gpu", devices=devices, num_nodes=num_nodes, strategy=DDPStrategy(static_graph=True),
#                         callbacks=[checkpoint_callback, early_stopping_callback])

#     trainer.print("[checkpoint path]", save_path)

#     trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

#     trainer.print("train over")


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--gpus_per_node", type=int, default=1)
#     parser.add_argument("--nodes", type=int, default=1)
#     args = parser.parse_args()

#     devices = args.gpus_per_node
#     num_nodes = args.nodes
#     train_model(devices=devices, num_nodes=num_nodes)