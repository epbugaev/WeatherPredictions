"""Static (time-invariant) input fields: orography and land-sea mask (exp 24).

Константы из WeatherBench ``constants_1.40625deg.nc`` подаются моделям как
дополнительные входные каналы. Чтение/кроп общие с географическими входами
диабатического блока (``PhysicsResidualMixin._load_static_geo``).
"""

from __future__ import annotations

import h5netcdf
import numpy as np
import torch

STATIC_INPUT_FIELDS = ("orography", "lsm")


def read_constant_fields(path: str, names: list[str], cut: list[int]) -> dict[str, np.ndarray]:
    """Прочитать и кропнуть 2-D константные поля из constants-NetCDF.

    Args:
        path: путь к ``constants_1.40625deg.nc``.
        names: имена переменных (например, ``["orography", "lsm", "lat2d"]``).
        cut: окно кропа ``[lat0, lat1, lon0, lon1]`` на нативной сетке.

    Returns:
        Dict ``name -> np.ndarray`` формы ``(lat1-lat0, lon1-lon0)``, float32.
    """
    la0, la1, lo0, lo1 = cut
    fields: dict[str, np.ndarray] = {}
    with h5netcdf.File(path, "r") as f:
        for name in names:
            fields[name] = np.asarray(f.variables[name], dtype=np.float32)[la0:la1, lo0:lo1]
    return fields


def standardize_field(field: np.ndarray) -> np.ndarray:
    """Z-score a 2-D field over its own crop: ``(x - mean) / (std + 1e-6)``.

    Args:
        field: 2-D array to standardize (e.g. orography, cropped).

    Returns:
        ``np.ndarray`` of the same shape and dtype, zero-mean, unit-variance
        (up to the ``1e-6`` stabilizer added to the denominator).
    """
    return (field - field.mean()) / (field.std() + 1e-6)


def load_static_input_fields(
    path: str, fields: list[str], cut: list[int], H: int, W: int
) -> torch.Tensor:
    """Загрузить нормированные статические входные каналы.

    Нормировка: орография — z-score по кропу; lsm — как есть (уже 0..1).
    Порядок каналов повторяет порядок ``fields``.

    Args:
        fields: подмножество :data:`STATIC_INPUT_FIELDS`, непустое.
        cut: окно кропа ``[lat0, lat1, lon0, lon1]``.
        H, W: ожидаемый пространственный shape (валидация кропа).

    Returns:
        ``torch.Tensor`` формы ``(S, H, W)``, float32, ``S = len(fields)``.

    Raises:
        ValueError: пустой/неизвестный список полей или кроп не равен ``(H, W)``.
    """
    if not fields:
        raise ValueError("static_input_fields must be a non-empty list")
    unknown = [name for name in fields if name not in STATIC_INPUT_FIELDS]
    if unknown:
        raise ValueError(f"Unknown static input fields {unknown}; supported: {STATIC_INPUT_FIELDS}")
    raw = read_constant_fields(path, list(fields), cut)
    layers = []
    for name in fields:
        field = raw[name]
        if field.shape != (H, W):
            raise ValueError(
                f"static field {name!r} crop {field.shape} != ({H}, {W}); check cut {cut}"
            )
        if name == "orography":
            field = standardize_field(field)
        layers.append(field)
    return torch.from_numpy(np.stack(layers, axis=0)).float()


class StaticInputMixin:
    """Опциональные статические входные каналы для ``nn.Module``-хоста.

    Вызвать :meth:`init_static_input` в ``__init__`` хоста (RNG не потребляется,
    инициализация весов не сдвигается), расширить первый слой на возвращённое
    число каналов и оборачивать кадры в :meth:`append_static_input` в forward.
    """

    static_input: torch.Tensor | None

    def init_static_input(
        self,
        static_input_fields: list[str] | None,
        static_constants_path: str | None,
        static_cut: list[int] | None,
        H: int,
        W: int,
    ) -> int:
        """Построить и зарегистрировать буфер ``(1, S, H, W)``.

        Args:
            static_input_fields: список полей или ``None``/пустой = выключено.
            static_constants_path: путь к constants-NetCDF (обязателен при включении).
            static_cut: окно кропа ``[lat0, lat1, lon0, lon1]`` (обязателен при включении).
            H, W: пространственный shape кадра модели.

        Returns:
            Число статических каналов S (0 = выключено, буфер не создаётся).

        Raises:
            ValueError: включено, но нет пути или кропа.
        """
        if not static_input_fields:
            self.static_input = None
            return 0
        if static_constants_path is None:
            raise ValueError("static_input_fields requires static_constants_path")
        if static_cut is None:
            raise ValueError("static_input_fields requires static_cut")
        static = load_static_input_fields(
            static_constants_path, static_input_fields, static_cut, H, W
        )
        self.register_buffer("static_input", static.unsqueeze(0))
        return static.shape[0]

    def append_static_input(self, frames: torch.Tensor) -> torch.Tensor:
        """Приклеить статические каналы к батчу кадров.

        Args:
            frames: ``torch.Tensor`` формы ``(N, C, H, W)``.

        Returns:
            ``(N, C+S, H, W)``; при выключенной статике — ``frames`` как есть.
        """
        if self.static_input is None:
            return frames
        return torch.cat([frames, self.static_input.expand(frames.shape[0], -1, -1, -1)], dim=1)
