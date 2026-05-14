"""Метрики качества прогноза погоды: WRMSE, WACC, Bias, Activity и др.

Все функции работают с тензорами PyTorch и предполагают, что предпоследняя
ось — широта (H), последняя — долгота (W). Веса по широте — нормированный
`cos(lat)` (равномерное распределение `lat` от 90° до -90°).
"""

import torch


@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """Перевести индекс широты в градусы (от 90° к -90°).

    Args:
        j: индексы вдоль оси широты, любая форма.
        num_lat: общее число точек по широте.

    Returns:
        Широты в градусах той же формы, что и `j`.
    """
    return 90.0 - j * 180.0 / float(num_lat - 1)


# @torch.jit.script
def type_weighted_bias_torch_channels(pred: torch.Tensor) -> torch.Tensor:
    """Bias по каналам, взвешенный по широте.

    Args:
        pred: разность `pred - gt`, форма `(B, C, H, W)` или `(B, T, C, H, W)`.

    Returns:
        Тензор формы `(B, C)` или `(B, T, C)`: среднее по spatial-осям, взвешенное по широте.
    """
    weight = _lat_weight(pred)
    return torch.mean(weight * pred, dim=(-1, -2))


# @torch.jit.script
def type_weighted_bias_torch(pred: torch.Tensor) -> torch.Tensor:
    """Bias по каналам, усреднённый по батчу.

    Args:
        pred: разность `pred - gt`, форма `(B, C, H, W)`.

    Returns:
        Тензор формы `(C,)`.
    """
    result = type_weighted_bias_torch_channels(pred)
    return torch.mean(result, dim=0)


# @torch.jit.script
def type_weighted_activity_torch_channels(pred: torch.Tensor) -> torch.Tensor:
    """Activity (взвешенный RMS вокруг среднего) по каналам.

    Args:
        pred: тензор формы `(B, C, H, W)` или `(B, T, C, H, W)`.

    Returns:
        Тензор формы `(B, C)` или `(B, T, C)`.
    """
    weight = _lat_weight(pred)
    result = torch.sqrt(
        torch.mean(
            weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2,
            dim=(-1, -2),
        )
    )
    return result


def type_weighted_activity_torch(pred: torch.Tensor) -> torch.Tensor:
    """Activity по каналам, усреднённый по батчу.

    Args:
        pred: тензор формы `(B, C, H, W)` или `(B, T, C, H, W)`.

    Returns:
        Тензор формы `(C,)` или `(T, C)`.
    """
    result = type_weighted_activity_torch_channels(pred)
    return torch.mean(result, dim=0)


@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    """Cos(lat)-вес с множителем `num_lat`, отнормированный на `s`.

    Args:
        j: индексы широты.
        num_lat: число точек по широте.
        s: сумма `cos(lat)`.

    Returns:
        Тензор весов формы `j`.
    """
    return num_lat * torch.cos(3.1416 / 180.0 * lat(j, num_lat)) / s


@torch.jit.script
def _lat_weight(pred: torch.Tensor) -> torch.Tensor:
    """Готовые к broadcast веса cos(lat) для spatial-осей `pred`.

    Args:
        pred: тензор формы `(B, C, H, W)` или `(B, T, C, H, W)`.

    Returns:
        Веса формы `(1, 1, -1, 1)` для 4D и `(1, 1, 1, -1, 1)` для 5D.
    """
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    factor = latitude_weighting_factor_torch(lat_t, num_lat, s)
    if pred.dim() == 5:
        return torch.reshape(factor, (1, 1, 1, -1, 1))
    return torch.reshape(factor, (1, 1, -1, 1))


@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lat-weighted RMSE по каналам.

    Args:
        pred: предсказание, форма `(B, C, H, W)` или `(B, T, C, H, W)`.
        target: эталон той же формы.

    Returns:
        Тензор формы `(B, C)` или `(B, T, C)`.
    """
    weight = _lat_weight(pred)
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))
    return result


@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lat-weighted RMSE по каналам, усреднённый по батчу.

    Args:
        pred: форма `(B, C, H, W)` или `(B, T, C, H, W)`.
        target: эталон той же формы.

    Returns:
        Тензор формы `(C,)` или `(T, C)`.
    """
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lat-weighted Anomaly Correlation Coefficient (ACC) по каналам.

    `pred` и `target` должны быть аномалиями (значение минус климатология).

    Args:
        pred: аномалия предсказания, форма `(B, C, H, W)` или `(B, T, C, H, W)`.
        target: аномалия эталона той же формы.

    Returns:
        Тензор формы `(B, C)` или `(B, T, C)`; значения в `[-1, 1]`.
    """
    weight = _lat_weight(pred)
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(weight * pred * pred, dim=(-1, -2))
        * torch.sum(weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lat-weighted ACC по каналам, усреднённый по батчу.

    Args:
        pred: аномалия предсказания.
        target: аномалия эталона.

    Returns:
        Тензор формы `(C,)` или `(T, C)`.
    """
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Относительная ошибка на верхних квантилях (RQE).

    Считает разность квантилей `pred` и `target` для 50 уровней
    `1 - logspace(-4, -1, 50)` (т.е. от ~0.9 до ~0.9999) и возвращает
    среднюю относительную ошибку по батчу и каналам.

    Args:
        pred: предсказание `(n, c, h, w)` в физических единицах.
        target: эталон той же формы.

    Returns:
        Тензор формы `(quantiles,)` — относительная ошибка по уровням.
    """
    qs = 50
    qlim = 4
    qcut = 1
    n, c, h, w = pred.size()
    qtile = 1.0 - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype)
    P_tar = torch.quantile(target.view(n, c, h * w), q=qtile, dim=-1)
    qtile = 1.0 - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype)
    P_pred = torch.quantile(pred.view(n, c, h * w), q=qtile, dim=-1)
    return torch.mean(torch.mean((P_pred - P_tar) / P_tar, dim=0), dim=0)


class Metrics:
    """Фасад над per-channel метриками для использования из тренинг-цикла.

    Хранит per-channel `data_mean` и `data_std` (обычно из train-датасета) и
    возвращает метрики в физических единицах: помноженные на `data_std` для
    обратного выхода из z-score нормализации.

    Args:
        data_mean: 1-D тензор per-channel среднего, форма `(C,)`. Может быть None.
        data_std: 1-D тензор per-channel std, форма `(C,)`. Может быть None.
    """

    def __init__(self, data_mean=None, data_std=None):
        self.data_mean = data_mean
        self.data_std = data_std

    def MSE(self, pred, gt):
        """Скалярный MSE по всему тензору.

        Args:
            pred: предсказание.
            gt: эталон.

        Returns:
            Float — `torch.mean((pred - gt) ** 2).item()`.
        """
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()

    def Bias(self, pred, gt):
        """Bias `(pred - gt)`, взвешенный по широте, в физических единицах.

        Args:
            pred: предсказание `(B, C, H, W)`.
            gt: эталон той же формы.

        Returns:
            Список длины `C`.
        """
        data_std = self.data_std.to(gt.device)
        return (type_weighted_bias_torch(pred - gt) * data_std).tolist()

    def Activity(self, pred, clim_time_mean_daily):
        """Activity предсказания относительно климатологии, в физических единицах.

        Args:
            pred: предсказание `(B, C, H, W)`.
            clim_time_mean_daily: климатологическое среднее той же формы.

        Returns:
            Список длины `C`.
        """
        clim_time_mean_daily = clim_time_mean_daily.to(pred.device)
        data_std = self.data_std.to(pred.device)
        return (
            type_weighted_activity_torch(pred - clim_time_mean_daily) * data_std
        ).tolist()

    def WRMSE(self, pred, gt):
        """Lat-weighted RMSE по каналам, в физических единицах.

        Args:
            pred: предсказание `(B, C, H, W)` или `(B, T, C, H, W)`.
            gt: эталон той же формы.

        Returns:
            Список (или вложенный список для 5D) длины `C`.
        """
        data_std = self.data_std.to(gt.device)
        return (weighted_rmse_torch(pred, gt) * data_std).tolist()

    def WACC(self, pred, gt, clim_time_mean_daily):
        """Lat-weighted ACC по каналам относительно климатологии.

        Args:
            pred: предсказание `(B, C, H, W)`.
            gt: эталон той же формы.
            clim_time_mean_daily: климатология той же формы.

        Returns:
            Список длины `C`; значения в `[-1, 1]`.
        """
        clim_time_mean_daily = clim_time_mean_daily.to(gt.device)
        return (weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)).tolist()

    def RQE(self, pred, gt):
        """Relative Quantile Error на хвостах распределений по выбранным каналам.

        Берёт `pred` и `gt` в z-score, обратно-нормализует через `data_mean/std`,
        затем считает `top_quantiles_error_torch` на каналах `[37, 24, 0, 11, 2, 66]`
        (захардкоженный набор: z500, u10, t2m, q700, v10, t850 в текущем layout’е).

        Args:
            pred: z-score предсказание `(B, C, H, W)`.
            gt: z-score эталон той же формы.

        Returns:
            Список `quantile_levels × len(selected_channels)`.
        """
        data_mean = self.data_mean.to(gt.device)
        data_std = self.data_std.to(gt.device)
        pred_real = pred * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(
            1, gt.shape[1], 1, 1
        )
        gt_real = gt * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        return (
            top_quantiles_error_torch(
                pred_real[:, [37, 24, 0, 11, 2, 66], :, :], gt_real[:, [37, 24, 0, 11, 2, 66], :, :]
            )
        ).tolist()
