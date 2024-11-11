import torch
import torch.nn as nn


def denorm(item, std, mean, idx=0):
    mean = mean[idx]
    std = std[idx]
    item_denorm = item * std + mean
    return item_denorm




# def weighted_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     """
#     Вычисляет взвешенный по широте RMSE
#     Args:
#         pred: предсказанные значения [bs, t, c, h, w]
#         target: целевые значения [bs, t, c, h, w]
#     Returns:
#         взвешенное значение RMSE для каждого канала и временного шага [bs, t, c]
#     """
    
#     bs, t, c, num_lat, num_long = pred.shape
    
#     # Создаем массив широт и вычисляем веса
#     lats = torch.arange(0, num_lat, device=pred.device)
#     lat_radians = torch.pi / 180.0 * lats
#     cos_lats = torch.cos(lat_radians)
#     s = torch.sum(cos_lats)
    
#     # Вычисляем веса по широте [lat, 1]
#     weights = (cos_lats / s).view(-1, 1)
    
#     # Расширяем веса для бродкастинга [1, 1, 1, lat, 1]
#     weights = weights.view(1, 1, 1, num_lat, 1)
    
#     # Вычисляем квадрат разности
#     # squared_diff = (pred - target) ** 2  # [bs, t, c, lat, lon]
#     squared_diff = (pred - target) ** 2  # [bs, t, c, lat, lon]
    
#     # Применяем веса и усредняем по широте и долготе
#     weighted_squared_diff = squared_diff * weights  # [bs, t, c, lat, lon]
#     mean_weighted_diff = torch.mean(weighted_squared_diff)  # [bs, t, c]
    
#     # Вычисляем RMSE
#     rmse = torch.sqrt(mean_weighted_diff * num_lat)
    
#     return rmse


# def weighted_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     """
#     Вычисляет взвешенный по широте RMSE
#     Args:
#         pred: предсказанные значения [bs, t, c, h, w]
#         target: целевые значения [bs, t, c, h, w]
#     Returns:
#         взвешенное значение RMSE для каждого канала и временного шага [bs, t, c]
#     """
#     bs, t, c, num_lat, num_long = pred.shape
    
#     # Создаем массив широт и вычисляем веса
#     lats = torch.arange(0, num_lat, device=pred.device)
#     lat_radians = torch.pi / 180.0 * lats
#     cos_lats = torch.cos(lat_radians)
#     s = torch.sum(cos_lats)
    
#     # Вычисляем веса по широте [lat, 1]
#     weights = (cos_lats / s).view(-1, 1)
    
#     # Расширяем веса для бродкастинга [1, 1, 1, lat, 1]
#     weights = weights.view(1, 1, 1, num_lat, 1)
    
#     # Вычисляем квадрат разности
#     squared_diff = torch.abs(pred - target) # [bs, t, c, lat, lon]
    
#     # Применяем веса и усредняем по широте и долготе
#     weighted_squared_diff = squared_diff * weights  # [bs, t, c, lat, lon]
#     mae = torch.mean(weighted_squared_diff)  # [bs, t, c]
    
#     return mae

# torch version for rmse comp
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, t, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result)

@torch.jit.script
def weighted_mae_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, t, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * torch.abs(pred - target), dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_mae_torch_channels(pred, target)
    return torch.mean(result)


def calculate_metrics(outputs, test, device, dict_with_std_mean, j=72):
    u_10_wrmse = []
    v_10_wrmse = []
    t2_wrmse = []
    tp_wrmse = []
    
    u_10_wmae = []
    v_10_wmae = []
    t2_wmae = []
    tp_wmae = []
    
    u_10_mae = []
    v_10_mae = []
    t2_mae = []
    tp_mae = []
    
    u_10_rmse = []
    v_10_rmse = []
    t2_rmse = []
    tp_rmse = []
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    for j in range(72):
        # wrmse
        t2_wrmse.append(weighted_rmse(denorm(torch.Tensor(outputs[:, j:j+1, 0, :1]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0), 
                                           denorm(test[:, j:j+1, :1], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0)).cpu().detach().numpy())
        u_10_wrmse.append(weighted_rmse(denorm(torch.Tensor(outputs[:, j:j+1, 0, 1:2]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1), 
                                           denorm(test[:, j:j+1, 1:2], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1)).cpu().detach().numpy())
        v_10_wrmse.append(weighted_rmse(denorm(torch.Tensor(outputs[:, j:j+1, 0, 2:3]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2), 
                                           denorm(test[:, j:j+1, 2:3], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2)).cpu().detach().numpy())
        tp_wrmse.append(weighted_rmse(denorm(torch.Tensor(outputs[:, j:j+1, 0, 3:4]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3), 
                                           denorm(test[:, j:j+1, 3:4], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3)).cpu().detach().numpy())
        
        # wmae
        t2_wmae.append(weighted_mae(denorm(torch.Tensor(outputs[:, j:j+1, 0, :1]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0), 
                                        denorm(test[:, j:j+1, :1], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0)).cpu().detach().numpy())
        u_10_wmae.append(weighted_mae(denorm(torch.Tensor(outputs[:, j:j+1, 0, 1:2]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1), 
                                        denorm(test[:, j:j+1, 1:2], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1)).cpu().detach().numpy())
        v_10_wmae.append(weighted_mae(denorm(torch.Tensor(outputs[:, j:j+1, 0, 2:3]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2), 
                                        denorm(test[:, j:j+1, 2:3], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2)).cpu().detach().numpy())
        tp_wmae.append(weighted_mae(denorm(torch.Tensor(outputs[:, j:j+1, 0, 3:4]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3), 
                                        denorm(test[:, j:j+1, 3:4], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3)).cpu().detach().numpy())
        
        # rmse
        t2_rmse.append(torch.sqrt(mse_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, :1]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0),
                                         denorm(test[:, j:j+1, :1], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0))).cpu().detach().numpy())
        u_10_rmse.append(torch.sqrt(mse_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 1:2]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1),
                                         denorm(test[:, j:j+1, 1:2], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1))).cpu().detach().numpy())
        v_10_rmse.append(torch.sqrt(mse_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 2:3]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2),
                                         denorm(test[:, j:j+1, 2:3], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2))).cpu().detach().numpy())
        tp_rmse.append(torch.sqrt(mse_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 3:4]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3),
                                         denorm(test[:, j:j+1, 3:4], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3))).cpu().detach().numpy())
        
        # mae
        t2_mae.append(mae_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, :1]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0),
                             denorm(test[:, j:j+1, :1], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=0)).cpu().detach().numpy())
        u_10_mae.append(mae_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 1:2]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1),
                             denorm(test[:, j:j+1, 1:2], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=1)).cpu().detach().numpy())
        v_10_mae.append(mae_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 2:3]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2),
                             denorm(test[:, j:j+1, 2:3], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=2)).cpu().detach().numpy())
        tp_mae.append(mae_loss(denorm(torch.Tensor(outputs[:, j:j+1, 0, 3:4]).to(device), dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3),
                             denorm(test[:, j:j+1, 3:4], dict_with_std_mean['std'], dict_with_std_mean['mean'], idx=3)).cpu().detach().numpy())

    metrics_dict = {
        't2_wrmse': t2_wrmse,
        'u_10_wrmse': u_10_wrmse,
        'v_10_wrmse': v_10_wrmse,
        'tp_wrmse': tp_wrmse,
        't2_wmae': t2_wmae,
        'u_10_wmae': u_10_wmae,
        'v_10_wmae': v_10_wmae,
        'tp_wmae': tp_wmae,
        't2_rmse': t2_rmse,
        'u_10_rmse': u_10_rmse,
        'v_10_rmse': v_10_rmse,
        'tp_rmse': tp_rmse,
        't2_mae': t2_mae,
        'u_10_mae': u_10_mae,
        'v_10_mae': v_10_mae,
        'tp_mae': tp_mae
    }
    
    return metrics_dict
        
        


