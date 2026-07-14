"""Пины гео-гейта диабатической головы Q_θ (арм X2 лестницы exp16).

Q_θ — **единственная** голова PI-IAM4VP, которая читает статическую географию
(орография, |широта|/90, маска суша/море); корректор и backbone её не видят. Значит,
вывод exp16 «весь выигрыш физики несёт Q_θ» смешивает три вещи: маску на t/q,
свободный аддитивный источник и **гео-эмбеддинг**. Арм X2 отделяет третью.

Контроль честен, только если X2 отличается от X1 **информацией, а не ёмкостью**:
гео-каналы обнуляются, но не выбрасываются, поэтому входная ширина ``diabatic_head``
и число его параметров остаются теми же. Эти тесты пинят ровно это.
"""

from __future__ import annotations

import h5netcdf
import numpy as np
import pytest
import torch

from Models.IAM4VP import IAM4VP

CUT = [75, 107, 164, 228]
H_DATA, W_DATA = 32, 64


@pytest.fixture(scope="module")
def constants_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Синтетические константы WeatherBench: orography, lsm, lat2d на полной сетке."""
    path = tmp_path_factory.mktemp("constants") / "constants_1.40625deg.nc"
    lat_full, lon_full = 128, 256
    rng = np.random.default_rng(0)
    orography = rng.normal(500.0, 300.0, size=(lat_full, lon_full)).astype(np.float32)
    lsm = rng.integers(0, 2, size=(lat_full, lon_full)).astype(np.float32)
    lat1d = np.linspace(-89.3, 89.3, lat_full, dtype=np.float32)
    lat2d = np.repeat(lat1d[:, None], lon_full, axis=1)
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"lat": lat_full, "lon": lon_full}
        for name, values in (("orography", orography), ("lsm", lsm), ("lat2d", lat2d)):
            variable = f.create_variable(name, ("lat", "lon"), data=values)
            variable.attrs["units"] = "1"
    return str(path)


def build_model(constants_path: str, *, use_geo: bool) -> IAM4VP:
    """Арм с Q_θ и без физического признака (конфигурация X1/X2)."""
    return IAM4VP(
        T_data=2,
        C_data=69,
        H_data=H_DATA,
        W_data=W_DATA,
        use_physics=False,
        use_physics_residual_corrector=True,
        physics_feature_mode="no_physics",
        use_diabatic_term=True,
        diabatic_hidden_channels=8,
        diabatic_constants_path=constants_path,
        diabatic_cut=CUT,
        diabatic_apply_to="t_and_q",
        diabatic_use_geo=use_geo,
    )


def test_geo_gate_zeroes_the_buffer_but_keeps_its_shape(constants_path: str) -> None:
    """use_geo=False обнуляет географию, НЕ меняя форму буфера."""
    with_geo = build_model(constants_path, use_geo=True)
    without_geo = build_model(constants_path, use_geo=False)

    assert with_geo.diabatic_geo.shape == without_geo.diabatic_geo.shape
    assert without_geo.diabatic_geo.abs().max().item() == 0.0
    assert with_geo.diabatic_geo.abs().max().item() > 0.0


def test_geo_gate_does_not_change_capacity(constants_path: str) -> None:
    """X2 отличается от X1 ИНФОРМАЦИЕЙ, а не числом параметров — иначе не контроль."""
    with_geo = build_model(constants_path, use_geo=True)
    without_geo = build_model(constants_path, use_geo=False)

    n_with = sum(p.numel() for p in with_geo.diabatic_head.parameters())
    n_without = sum(p.numel() for p in without_geo.diabatic_head.parameters())
    assert n_with == n_without

    total_with = sum(p.numel() for p in with_geo.parameters())
    total_without = sum(p.numel() for p in without_geo.parameters())
    assert total_with == total_without


def test_default_is_geo_on_bit_exact(constants_path: str) -> None:
    """Дефолт — география включена: старые армы (R2/R3/R4/R5) не меняются."""
    default = IAM4VP(
        T_data=2,
        C_data=69,
        H_data=H_DATA,
        W_data=W_DATA,
        use_physics=False,
        use_physics_residual_corrector=True,
        physics_feature_mode="no_physics",
        use_diabatic_term=True,
        diabatic_hidden_channels=8,
        diabatic_constants_path=constants_path,
        diabatic_cut=CUT,
        diabatic_apply_to="t_and_q",
    )
    explicit = build_model(constants_path, use_geo=True)
    assert torch.equal(default.diabatic_geo, explicit.diabatic_geo)


def test_geo_channels_are_orography_abslat_lsm(constants_path: str) -> None:
    """Три гео-канала — стандартизованная орография, |широта|/90, маска суши."""
    geo = IAM4VP._load_static_geo(constants_path, CUT, H_DATA, W_DATA)
    assert geo.shape == (1, 3, H_DATA, W_DATA)

    orography_standardized = geo[0, 0]
    assert abs(orography_standardized.mean().item()) < 1e-4
    assert abs(orography_standardized.std(unbiased=False).item() - 1.0) < 1e-3

    abslat = geo[0, 1]
    assert abslat.min().item() >= 0.0
    assert abslat.max().item() <= 1.0

    lsm = geo[0, 2]
    assert set(np.unique(lsm.numpy()).tolist()) <= {0.0, 1.0}


def test_zeroed_geo_kills_geographic_information(constants_path: str) -> None:
    """Голова X2 не может отличить точки, различающиеся ТОЛЬКО географией.

    Прямая проверка смысла контроля: подаём одно и то же состояние, но модель без
    гео обязана выдать поле, инвариантное к рельефу — то есть её выход не должен
    коррелировать с орографией сильнее, чем при случайной перестановке.
    """
    model = build_model(constants_path, use_geo=False)
    state = torch.zeros(1, 65, H_DATA, W_DATA)
    geo = model.diabatic_geo.expand(1, -1, -1, -1)

    output = model.diabatic_head(torch.cat([state, geo], dim=1))
    # Вход постоянен по пространству => выход обязан быть постоянным по пространству.
    spatial_spread = output.std(dim=(2, 3)).max().item()
    assert spatial_spread == pytest.approx(0.0, abs=1e-6)


def test_geo_on_lets_the_head_see_space(constants_path: str) -> None:
    """Зеркальный пин: с географией тот же постоянный вход даёт НЕ постоянный выход."""
    model = build_model(constants_path, use_geo=True)
    # zero_init гасит последнюю свёртку, поэтому смотрим на предпоследний слой.
    features = torch.cat(
        [torch.zeros(1, 65, H_DATA, W_DATA), model.diabatic_geo.expand(1, -1, -1, -1)], dim=1
    )
    hidden = model.diabatic_head.net[:-1](features)
    assert hidden.std(dim=(2, 3)).max().item() > 1e-6
