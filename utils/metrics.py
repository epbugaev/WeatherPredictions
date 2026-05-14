import torch

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

# @torch.jit.script
def type_weighted_bias_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)


    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    if len(pred.shape) == 5:
        weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, 1, -1, 1))
    else:
        weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))

    result = torch.mean(weight * pred, dim=(-1, -2))

    # result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
    return result

# @torch.jit.script
def type_weighted_bias_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_bias_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)

# @torch.jit.script
def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    weight = _lat_weight(pred)
    result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
    return result

def type_weighted_activity_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_activity_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

@torch.jit.script
def _lat_weight(pred: torch.Tensor) -> torch.Tensor:
    # cos(lat)-нормированные веса, готовые к broadcast по pred:
    # форма (1,1,1,-1,1) для (B,T,C,H,W) и (1,1,-1,1) для (B,C,H,W).
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    factor = latitude_weighting_factor_torch(lat_t, num_lat, s)
    if pred.dim() == 5:
        return torch.reshape(factor, (1, 1, 1, -1, 1))
    return torch.reshape(factor, (1, 1, -1, 1))

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w] or [n, t, c, h, w] and returns latitude-weighted rmse for each chann
    weight = _lat_weight(pred)
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    weight = _lat_weight(pred)
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 50
    qlim = 4
    qcut = 1
    n, c, h, w = pred.size()
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype)
    P_tar = torch.quantile(target.view(n,c,h*w), q=qtile, dim=-1)
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype)
    P_pred = torch.quantile(pred.view(n,c,h*w), q=qtile, dim=-1)
    return torch.mean(torch.mean((P_pred - P_tar)/P_tar, dim=0), dim=0)
    # return torch.mean((P_pred - P_tar)/P_tar, dim=0).view(c)


class Metrics():
    def __init__(self, data_mean=None, data_std=None):
        self.data_mean = data_mean
        self.data_std = data_std
        
    def MSE(self, pred, gt):
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()

    def Bias(self, pred, gt):
        data_std = self.data_std.to(gt.device)
        return (type_weighted_bias_torch(pred - gt, metric_type="all") * data_std).tolist()

    def Activity(self, pred, clim_time_mean_daily):
        clim_time_mean_daily = clim_time_mean_daily.to(pred.device)
        data_std = self.data_std.to(pred.device)
        return (type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="all") * data_std).tolist()

    def WRMSE(self, pred, gt):
        data_std = self.data_std.to(gt.device)
        return (weighted_rmse_torch(pred, gt) * data_std).tolist()

    def WACC(self, pred, gt, clim_time_mean_daily):
        clim_time_mean_daily = clim_time_mean_daily.to(gt.device)
        return (weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)).tolist()


    def RQE(self, pred, gt):
        data_mean = self.data_mean.to(gt.device)
        data_std = self.data_std.to(gt.device)
        pred_real = pred * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        gt_real = gt * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        return (top_quantiles_error_torch(pred_real[:,[37,24,0,11,2,66],:,:], gt_real[:,[37,24,0,11,2,66],:,:])).tolist()