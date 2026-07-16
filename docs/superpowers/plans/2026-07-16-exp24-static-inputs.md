# Exp 24 — Static Inputs (orography + lsm) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Дать трём семействам моделей (PI-IAM4VP, PI-PredRNNv2, PI-SimVPv2) опциональные статические входные каналы — орографию и маску суша/море — и сгенерировать 6 конфигов эксперимента 24 (USA, армы no_physics и A2-exp13).

**Architecture:** Opt-in на уровне модели: общий загрузчик + `StaticInputMixin` в новом `utils/static_input.py`; каждая модель регистрирует буфер `(1, S, H, W)` и конкатенирует его к кадрам на входе первого слоя; выход остаётся 69 каналов, физический residual-путь и даталоадер не меняются. Spec: `docs/superpowers/specs/2026-07-16-exp24-static-inputs-design.md`.

**Tech Stack:** Python 3.10+, PyTorch, h5netcdf, yaml, pytest (unittest-style, как в существующих тестах).

## Global Constraints

- Никаких `try/except` — только LBYL-проверки (`if ...: raise ValueError`). (CLAUDE.md §2)
- Никаких импортов внутри функций; stdlib → сторонние → локальные. (CLAUDE.md §2)
- Docstrings Google-стиля с shape тензоров у всех публичных функций/классов. (CLAUDE.md §4)
- Type hints PEP 604 (`str | None`, `list[int]`). (CLAUDE.md §4)
- После каждой задачи: `ruff check .` и `ruff format .` — ноль предупреждений.
- Коммиты на английском, conventional commits.
- **В рабочем дереве есть чужие незакоммиченные правки** (`tests/test_physics_sign_conventions.py`, `utils/physics.py`, `docs/planKDD.md`, `example/`) — НИКОГДА не `git add -A`; добавлять только файлы своей задачи поимённо.
- При `static_input_fields=None` (дефолт) поведение всех моделей бит-в-бит прежнее: буфер не создаётся, RNG-порядок не сдвигается (загрузка буфера не потребляет RNG).
- Выход всех моделей — прежние 69 каналов; статика только на входе.

---

### Task 1: `utils/static_input.py` — загрузчик и `StaticInputMixin`

**Files:**
- Create: `utils/static_input.py`
- Test: `tests/test_static_input.py`

**Interfaces:**
- Produces: `read_constant_fields(path: str, names: list[str], cut: list[int]) -> dict[str, np.ndarray]` — сырые кропнутые поля `(H, W)` float32.
- Produces: `load_static_input_fields(path: str, fields: list[str], cut: list[int], H: int, W: int) -> torch.Tensor` — нормированный `(S, H, W)`.
- Produces: `StaticInputMixin` c методами `init_static_input(static_input_fields, static_constants_path, static_cut, H, W) -> int` (возвращает S; 0 = выключено) и `append_static_input(frames: (N, C, H, W)) -> (N, C+S, H, W)`.
- Produces: константа `STATIC_INPUT_FIELDS = ("orography", "lsm")`.

- [ ] **Step 1: Написать падающий тест**

```python
"""Static input channels (exp 24): loader, mixin, model integration."""

from __future__ import annotations

import os
import tempfile
import unittest

import h5netcdf
import numpy as np
import torch
from torch import nn

from utils.static_input import (
    STATIC_INPUT_FIELDS,
    StaticInputMixin,
    load_static_input_fields,
    read_constant_fields,
)

GRID_H, GRID_W = 32, 64
FULL_CUT = [0, GRID_H, 0, GRID_W]


def _write_constants_nc(path: str) -> dict[str, np.ndarray]:
    """Записать синтетический constants-файл; вернуть исходные массивы.

    Args:
        path: куда писать .nc (h5netcdf).

    Returns:
        Dict полей ``orography`` / ``lsm`` / ``lat2d`` формы ``(GRID_H, GRID_W)``.
    """
    rng = np.random.default_rng(0)
    orog = (rng.random((GRID_H, GRID_W)) * 3000.0).astype(np.float32)
    lsm = (rng.random((GRID_H, GRID_W)) > 0.5).astype(np.float32)
    lat_1d = np.linspace(-90.0, 90.0, GRID_H, dtype=np.float32)
    lat2d = np.repeat(lat_1d[:, None], GRID_W, axis=1)
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"lat": GRID_H, "lon": GRID_W}
        for name, data in {"orography": orog, "lsm": lsm, "lat2d": lat2d}.items():
            var = f.create_variable(name, ("lat", "lon"), np.float32)
            var[...] = data
    return {"orography": orog, "lsm": lsm, "lat2d": lat2d}


class StaticInputFileMixin(unittest.TestCase):
    """База: синтетический constants.nc на класс (пишется один раз)."""

    nc_path: str
    fields: dict[str, np.ndarray]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.nc_path = os.path.join(cls._tmpdir.name, "constants.nc")
        cls.fields = _write_constants_nc(cls.nc_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()


class TestReadConstantFields(StaticInputFileMixin):
    def test_reads_and_crops(self) -> None:
        cut = [2, 10, 4, 20]
        raw = read_constant_fields(self.nc_path, ["orography", "lsm"], cut)
        self.assertEqual(set(raw), {"orography", "lsm"})
        self.assertEqual(raw["orography"].shape, (8, 16))
        np.testing.assert_array_equal(
            raw["orography"], self.fields["orography"][2:10, 4:20]
        )
        self.assertEqual(raw["lsm"].dtype, np.float32)


class TestLoadStaticInputFields(StaticInputFileMixin):
    def test_shape_and_normalization(self) -> None:
        static = load_static_input_fields(
            self.nc_path, ["orography", "lsm"], FULL_CUT, GRID_H, GRID_W
        )
        self.assertEqual(static.shape, (2, GRID_H, GRID_W))
        self.assertEqual(static.dtype, torch.float32)
        # Орография — z-score по кропу.
        self.assertAlmostEqual(static[0].mean().item(), 0.0, places=4)
        self.assertAlmostEqual(static[0].std(correction=0).item(), 1.0, places=3)
        # lsm — как есть (0..1).
        torch.testing.assert_close(
            static[1], torch.from_numpy(self.fields["lsm"]), rtol=0, atol=0
        )

    def test_field_order_follows_request(self) -> None:
        ab = load_static_input_fields(
            self.nc_path, ["orography", "lsm"], FULL_CUT, GRID_H, GRID_W
        )
        ba = load_static_input_fields(
            self.nc_path, ["lsm", "orography"], FULL_CUT, GRID_H, GRID_W
        )
        torch.testing.assert_close(ab[0], ba[1], rtol=0, atol=0)

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            load_static_input_fields(
                self.nc_path, ["orography", "slt"], FULL_CUT, GRID_H, GRID_W
            )

    def test_rejects_empty_fields(self) -> None:
        with self.assertRaises(ValueError):
            load_static_input_fields(self.nc_path, [], FULL_CUT, GRID_H, GRID_W)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            load_static_input_fields(
                self.nc_path, ["orography"], [0, 8, 0, 16], GRID_H, GRID_W
            )


class _ToyModel(StaticInputMixin, nn.Module):
    """Минимальный носитель миксина для юнит-тестов."""

    def __init__(self, nc_path: str | None, fields: list[str] | None) -> None:
        super().__init__()
        self.num_static = self.init_static_input(
            fields, nc_path, FULL_CUT, GRID_H, GRID_W
        )


class TestStaticInputMixin(StaticInputFileMixin):
    def test_disabled_registers_nothing(self) -> None:
        model = _ToyModel(None, None)
        self.assertEqual(model.num_static, 0)
        self.assertNotIn("static_input", model.state_dict())
        frames = torch.randn(3, 5, GRID_H, GRID_W)
        self.assertIs(model.append_static_input(frames), frames)

    def test_enabled_appends_channels(self) -> None:
        model = _ToyModel(self.nc_path, ["orography", "lsm"])
        self.assertEqual(model.num_static, 2)
        self.assertIn("static_input", model.state_dict())
        frames = torch.randn(3, 5, GRID_H, GRID_W)
        out = model.append_static_input(frames)
        self.assertEqual(out.shape, (3, 7, GRID_H, GRID_W))
        torch.testing.assert_close(out[:, :5], frames, rtol=0, atol=0)
        torch.testing.assert_close(out[1, 5:], out[2, 5:], rtol=0, atol=0)

    def test_enabled_requires_path_and_cut(self) -> None:
        with self.assertRaises(ValueError):
            _ToyModel(None, ["orography"])
        model = nn.Module()
        with self.assertRaises(ValueError):
            StaticInputMixin.init_static_input(
                model, ["orography"], self.nc_path, None, GRID_H, GRID_W
            )

    def test_buffer_consumes_no_rng(self) -> None:
        """Загрузка буфера не должна сдвигать RNG-поток инициализации весов."""
        torch.manual_seed(0)
        _ToyModel(self.nc_path, ["orography", "lsm"])
        state_after_enabled = torch.get_rng_state()
        torch.manual_seed(0)
        _ToyModel(None, None)
        state_after_disabled = torch.get_rng_state()
        self.assertTrue(torch.equal(state_after_enabled, state_after_disabled))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_static_input.py -v` (если uv недоступен — `python -m pytest`)
Expected: FAIL/ERROR c `ModuleNotFoundError: No module named 'utils.static_input'`

- [ ] **Step 3: Реализовать `utils/static_input.py`**

```python
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


def read_constant_fields(
    path: str, names: list[str], cut: list[int]
) -> dict[str, np.ndarray]:
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
            fields[name] = np.asarray(f.variables[name], dtype=np.float32)[
                la0:la1, lo0:lo1
            ]
    return fields


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
        raise ValueError(
            f"Unknown static input fields {unknown}; supported: {STATIC_INPUT_FIELDS}"
        )
    raw = read_constant_fields(path, list(fields), cut)
    layers = []
    for name in fields:
        field = raw[name]
        if field.shape != (H, W):
            raise ValueError(
                f"static field {name!r} crop {field.shape} != ({H}, {W}); check cut {cut}"
            )
        if name == "orography":
            field = (field - field.mean()) / (field.std() + 1e-6)
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
        return torch.cat(
            [frames, self.static_input.expand(frames.shape[0], -1, -1, -1)], dim=1
        )
```

Замечание: `register_buffer` доступен через хост (`nn.Module`); `self.static_input = None` в выключенном режиме — обычный атрибут, в `state_dict` не попадает.

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_static_input.py -v`
Expected: PASS (все тесты)

- [ ] **Step 5: Линт и коммит**

```bash
ruff check . && ruff format --check utils/static_input.py tests/test_static_input.py
git add utils/static_input.py tests/test_static_input.py
git commit -m "feat(exp24): static input loader and StaticInputMixin (orography + lsm)"
```

---

### Task 2: рефакторинг `_load_static_geo` на общий ридер

**Files:**
- Modify: `utils/physics_residual.py:767-795` (метод `_load_static_geo`) + импорт вверху файла
- Test: `tests/test_static_input.py` (дописать класс)

**Interfaces:**
- Consumes: `read_constant_fields` из Task 1.
- Produces: `PhysicsResidualMixin._load_static_geo` — сигнатура и выход прежние, бит-в-бит.

- [ ] **Step 1: Дописать падающий тест бит-в-бит эквивалентности**

В `tests/test_static_input.py` добавить (импорт `from utils.physics_residual import PhysicsResidualMixin` — вверх файла):

```python
class TestLoadStaticGeoUnchanged(StaticInputFileMixin):
    """Регресс-гард рефакторинга: _load_static_geo бит-в-бит как старая формула."""

    def test_matches_legacy_formula(self) -> None:
        geo = PhysicsResidualMixin._load_static_geo(
            self.nc_path, FULL_CUT, GRID_H, GRID_W
        )
        orog = self.fields["orography"]
        expected = np.stack(
            [
                (orog - orog.mean()) / (orog.std() + 1e-6),
                np.abs(self.fields["lat2d"]) / 90.0,
                self.fields["lsm"],
            ],
            axis=0,
        )[None]
        self.assertEqual(geo.shape, (1, 3, GRID_H, GRID_W))
        torch.testing.assert_close(
            geo, torch.from_numpy(expected).float(), rtol=0, atol=0
        )

    def test_requires_path(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsResidualMixin._load_static_geo(None, FULL_CUT, GRID_H, GRID_W)
```

- [ ] **Step 2: Прогнать — тест уже должен пройти на старом коде (это гард)**

Run: `uv run pytest tests/test_static_input.py::TestLoadStaticGeoUnchanged -v`
Expected: PASS (тест фиксирует текущее поведение ДО рефакторинга)

- [ ] **Step 3: Рефакторинг**

В `utils/physics_residual.py` добавить в блок локальных импортов вверху:

```python
from utils.static_input import read_constant_fields
```

и заменить тело `_load_static_geo` (строки 780-795, docstring оставить):

```python
        if path is None:
            raise ValueError("use_diabatic_term=True requires diabatic_constants_path")
        if cut is None:
            cut = [75, 107, 164, 228]
        raw = read_constant_fields(path, ["orography", "lsm", "lat2d"], cut)
        orog, lsm, lat2d = raw["orography"], raw["lsm"], raw["lat2d"]
        if orog.shape != (H, W):
            raise ValueError(f"geo crop {orog.shape} != ({H},{W}); check diabatic_cut {cut}")
        orog_n = (orog - orog.mean()) / (orog.std() + 1e-6)
        abslat_n = np.abs(lat2d) / 90.0
        geo = np.stack([orog_n, abslat_n, lsm], axis=0)[None]
        return torch.from_numpy(geo).float()
```

Если после замены `h5netcdf` в `physics_residual.py` больше нигде не используется — удалить его импорт (проверить: `grep -n "h5netcdf" utils/physics_residual.py`).

- [ ] **Step 4: Прогнать гард и смежные тесты**

Run: `uv run pytest tests/test_static_input.py tests/test_physics_residual_mixin.py -v`
Expected: PASS (гард по-прежнему зелёный — рефакторинг бит-в-бит)

- [ ] **Step 5: Линт и коммит**

```bash
ruff check .
git add utils/physics_residual.py tests/test_static_input.py
git commit -m "refactor(exp24): _load_static_geo reuses shared read_constant_fields"
```

---

### Task 3: статика в SimVP (v2)

**Files:**
- Modify: `Models/SimVP.py` (классы `SimVP_Model`, `PI_SimVP_Model`)
- Test: `tests/test_static_input.py` (дописать класс)

**Interfaces:**
- Consumes: `StaticInputMixin` из Task 1.
- Produces: `SimVP_Model(..., static_input_fields=None, static_constants_path=None, static_cut=None)` и то же у `PI_SimVP_Model`; forward-контракт `(B, T, C, H, W) -> (B, T, C, H, W)` не меняется.

- [ ] **Step 1: Написать падающий тест**

В `tests/test_static_input.py` (импорт `from Models.SimVP import PI_SimVP_Model, SimVP_Model` — вверх файла):

```python
SIMVP_SHAPE = (2, 69, GRID_H, GRID_W)

PHYSICS_NOPHYS = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_residual_lambda_l1": 1e-4,
    "physics_feature_mode": "no_physics",
    "physics_residual_shuffle": "none",
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
}


class TestSimVPStaticInput(StaticInputFileMixin):
    def _static_kwargs(self) -> dict:
        return {
            "static_input_fields": ["orography", "lsm"],
            "static_constants_path": self.nc_path,
            "static_cut": FULL_CUT,
        }

    def test_backbone_forward_shape_unchanged(self) -> None:
        torch.manual_seed(0)
        model = SimVP_Model(
            in_shape=SIMVP_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2,
            **self._static_kwargs(),
        )
        model.eval()
        x = torch.randn(2, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (2, 2, 69, GRID_H, GRID_W))

    def test_disabled_matches_param_absence_bitexact(self) -> None:
        torch.manual_seed(0)
        plain = SimVP_Model(in_shape=SIMVP_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2)
        torch.manual_seed(0)
        disabled = SimVP_Model(
            in_shape=SIMVP_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2,
            static_input_fields=None,
        )
        self.assertEqual(
            list(plain.state_dict().keys()), list(disabled.state_dict().keys())
        )
        plain.eval()
        disabled.eval()
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            torch.testing.assert_close(plain(x), disabled(x), rtol=0, atol=0)

    def test_pi_model_with_static_and_physics(self) -> None:
        torch.manual_seed(0)
        model = PI_SimVP_Model(
            in_shape=SIMVP_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2,
            **self._static_kwargs(), **PHYSICS_NOPHYS,
        )
        model.eval()
        model.set_physics_normalization(torch.zeros(69), torch.ones(69))
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (1, 2, 69, GRID_H, GRID_W))
        self.assertIn("static_input", model.state_dict())
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_static_input.py::TestSimVPStaticInput -v`
Expected: FAIL c `TypeError: ... unexpected keyword argument 'static_input_fields'` (у `SimVP_Model` его нет; у PI-модели ключи утекли бы в physics_kwargs)

- [ ] **Step 3: Реализация в `Models/SimVP.py`**

Импорт вверху файла: `from utils.static_input import StaticInputMixin`.

`SimVP_Model` — наследование и конструктор:

```python
class SimVP_Model(StaticInputMixin, nn.Module):
```

В `__init__` добавить параметры после `act_inplace=True`:

```python
        static_input_fields=None,
        static_constants_path=None,
        static_cut=None,
```

и в теле — до построения `self.enc` (H, W на строке 61 уже перезаписаны даунсемплом, поэтому статика инициализируется от исходного in_shape):

```python
        T, C, H, W = in_shape  # T is pre_seq_length
        num_static = self.init_static_input(
            static_input_fields, static_constants_path, static_cut, H, W
        )
        H, W = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))
        act_inplace = False
        self.enc = Encoder(C + num_static, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
```

(строка `H, W = int(...)` уже существует — сдвинуть `init_static_input` ДО неё). `self.dec` не меняется (выход `C`).

В `SimVP_Model.forward` после `x = x_raw.view(B * T, C, H, W)`:

```python
        x = self.append_static_input(x)
```

`PI_SimVP_Model.__init__` — добавить те же три параметра в сигнатуру (после `physics_chunk_size`) и пробросить их в `super().__init__(...)` рядом с остальными backbone-аргументами:

```python
            static_input_fields=static_input_fields,
            static_constants_path=static_constants_path,
            static_cut=static_cut,
```

(иначе ключи упадут в `**physics_kwargs` и сломают `init_physics_residual`).

- [ ] **Step 4: Прогнать тесты SimVP (новые + существующие)**

Run: `uv run pytest tests/test_static_input.py tests/test_pi_simvpv2.py -v`
Expected: PASS

- [ ] **Step 5: Линт и коммит**

```bash
ruff check .
git add Models/SimVP.py tests/test_static_input.py
git commit -m "feat(exp24): optional static input channels in SimVP/PI-SimVPv2"
```

---

### Task 4: статика в IAM4VP

**Files:**
- Modify: `Models/IAM4VP.py:229-370` (класс `IAM4VP`)
- Test: `tests/test_static_input.py` (дописать класс)

**Interfaces:**
- Consumes: `StaticInputMixin` из Task 1.
- Produces: `IAM4VP(..., static_input_fields=None, static_constants_path=None, static_cut=None)`; forward-контракт `(x_raw, y_raw, t) -> (B, C, H, W)` не меняется.

- [ ] **Step 1: Написать падающий тест**

В `tests/test_static_input.py` (импорт `from Models.IAM4VP import IAM4VP` — вверх файла):

```python
class TestIAM4VPStaticInput(StaticInputFileMixin):
    def _build(self, **kwargs) -> IAM4VP:
        torch.manual_seed(0)
        model = IAM4VP(
            T_data=2, C_data=69, H_data=GRID_H, W_data=GRID_W,
            hid_S=8, N_S=4, N_T=2, use_physics=False, **kwargs,
        )
        model.eval()
        return model

    def test_forward_shape_with_static(self) -> None:
        model = self._build(
            static_input_fields=["orography", "lsm"],
            static_constants_path=self.nc_path,
            static_cut=FULL_CUT,
        )
        x = torch.randn(2, 2, 69, GRID_H, GRID_W)
        t = torch.full((2,), 100.0)
        with torch.no_grad():
            first = model(x, None, t)
            second = model(x, [first], t)
        self.assertEqual(first.shape, (2, 69, GRID_H, GRID_W))
        self.assertEqual(second.shape, (2, 69, GRID_H, GRID_W))

    def test_disabled_matches_param_absence_bitexact(self) -> None:
        plain = self._build()
        disabled = self._build(static_input_fields=None)
        self.assertEqual(
            list(plain.state_dict().keys()), list(disabled.state_dict().keys())
        )
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        t = torch.full((1,), 100.0)
        with torch.no_grad():
            torch.testing.assert_close(
                plain(x, None, t), disabled(x, None, t), rtol=0, atol=0
            )
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_static_input.py::TestIAM4VPStaticInput -v`
Expected: FAIL — static-ключи проваливаются в `**physics_kwargs` → `TypeError` в `init_physics_residual`

- [ ] **Step 3: Реализация в `Models/IAM4VP.py`**

Импорт вверху: `from utils.static_input import StaticInputMixin`.

Наследование:

```python
class IAM4VP(StaticInputMixin, PhysicsResidualMixin, nn.Module):
```

Сигнатура `__init__` — после `use_physics: bool = True` добавить:

```python
        static_input_fields: list[str] | None = None,
        static_constants_path: str | None = None,
        static_cut: list[int] | None = None,
```

В теле `__init__` — сразу после `super().__init__()` (до `self.time_mlp`; буфер не потребляет RNG, порядок инициализации весов не сдвигается):

```python
        num_static = self.init_static_input(
            static_input_fields, static_constants_path, static_cut, H_data, W_data
        )
```

и расширить все три энкодера кадров:

```python
        self.enc = Encoder(C_data + num_static, hid_S, N_S)
        ...
        self.lp = Encoder(C_data + num_static, hid_S, N_S)
        self.lp_phys = Encoder(C_data + num_static, hid_S, N_S)
```

(`self.readout`, mask-токены, `init_physics_residual` — без изменений).

В `forward` — три точки конкатенации:

1. после `x = x_raw.view(B * T, C, H, W)`:

```python
        x = self.append_static_input(x)
```

2. вход `lp` (строка `embed2, skip_lp, embed_1_lp, embed_2_lp = self.lp(pred)`):

```python
            embed2, skip_lp, embed_1_lp, embed_2_lp = self.lp(self.append_static_input(pred))
```

3. вход `lp_phys` в legacy-ветке (строка `... = self.lp_phys(pred_to_hybrid)`):

```python
                embed2_phys, skip_lp_phys, embed_1_lp_phys, embed_2_lp_phys = self.lp_phys(
                    self.append_static_input(pred_to_hybrid)
                )
```

- [ ] **Step 4: Прогнать тесты (новые + физический голден IAM4VP)**

Run: `uv run pytest tests/test_static_input.py tests/test_physics_residual_mixin.py tests/test_physics_hybrid_exp16_port.py -v`
Expected: PASS (голдены не сдвинулись — при выключенной статике RNG-порядок прежний)

- [ ] **Step 5: Линт и коммит**

```bash
ruff check .
git add Models/IAM4VP.py tests/test_static_input.py
git commit -m "feat(exp24): optional static input channels in IAM4VP"
```

---

### Task 5: статика в PredRNNv2 / PI-PredRNNv2 (с патчингом)

**Files:**
- Modify: `Models/PredRNN.py` (классы `PredRNNv2_Model`, `PI_PredRNNv2_Model`)
- Modify: `Models/__init__.py` (проброс static-ключей в PI-билдере НЕ нужен — они идут верхнеуровневыми kwargs; изменений нет, файл указан для проверки)
- Test: `tests/test_static_input.py` (дописать класс)

**Interfaces:**
- Consumes: `StaticInputMixin` из Task 1; `_reshape_patch` (модульная функция `Models/PredRNN.py:35`).
- Produces: `PredRNNv2_Model(num_layers, num_hidden, configs, static_input_fields=None, static_constants_path=None, static_cut=None)`; то же у `PI_PredRNNv2_Model` (static-параметры — явные, до `**physics_kwargs`). Forward-контракт `(frames_tensor, mask_true) -> (next_frames, aux_losses)` не меняется.

- [ ] **Step 1: Написать падающий тест**

В `tests/test_static_input.py` (импорт `from Models.PredRNN import PI_PredRNNv2_Model, PredRNNv2_Model` — вверх файла):

```python
def _predrnn_configs(patch_size: int) -> dict:
    return {
        "in_shape": (2, 69, GRID_H, GRID_W),
        "patch_size": patch_size,
        "filter_size": 3,
        "stride": 1,
        "layer_norm": True,
        "pre_seq_length": 2,
        "aft_seq_length": 2,
        "reverse_scheduled_sampling": 0,
        "decouple_beta": 0.1,
    }


class TestPredRNNv2StaticInput(StaticInputFileMixin):
    def _static_kwargs(self) -> dict:
        return {
            "static_input_fields": ["orography", "lsm"],
            "static_constants_path": self.nc_path,
            "static_cut": FULL_CUT,
        }

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(1)
        frames = torch.randn(2, 4, GRID_H, GRID_W, 69)
        mask = torch.zeros(1, 1, 1, 1, 1)
        return frames, mask

    def test_forward_shape_patch1(self) -> None:
        torch.manual_seed(0)
        model = PredRNNv2_Model(2, (8, 8), _predrnn_configs(1), **self._static_kwargs())
        model.eval()
        frames, mask = self._inputs()
        with torch.no_grad():
            out, aux = model(frames, mask)
        self.assertEqual(out.shape, (2, 3, GRID_H, GRID_W, 69))
        self.assertIn("decouple", aux)

    def test_forward_shape_patch2_static_is_patched(self) -> None:
        """patch_size=2: буфер статики патчится, канальная арифметика сходится."""
        torch.manual_seed(0)
        model = PredRNNv2_Model(2, (8, 8), _predrnn_configs(2), **self._static_kwargs())
        model.eval()
        # frame_channel = 4*69, статика = 4*2 канала на латентной сетке H/2 x W/2.
        self.assertEqual(model.static_input.shape, (1, 8, GRID_H // 2, GRID_W // 2))
        frames, mask = self._inputs()
        with torch.no_grad():
            out, _ = model(frames, mask)
        self.assertEqual(out.shape, (2, 3, GRID_H, GRID_W, 69))

    def test_disabled_matches_param_absence_bitexact(self) -> None:
        torch.manual_seed(0)
        plain = PredRNNv2_Model(2, (8, 8), _predrnn_configs(1))
        torch.manual_seed(0)
        disabled = PredRNNv2_Model(2, (8, 8), _predrnn_configs(1), static_input_fields=None)
        self.assertEqual(
            list(plain.state_dict().keys()), list(disabled.state_dict().keys())
        )
        plain.eval()
        disabled.eval()
        frames, mask = self._inputs()
        with torch.no_grad():
            torch.testing.assert_close(
                plain(frames, mask)[0], disabled(frames, mask)[0], rtol=0, atol=0
            )

    def test_pi_model_with_static_and_physics(self) -> None:
        torch.manual_seed(0)
        model = PI_PredRNNv2_Model(
            2, (8, 8), _predrnn_configs(1), **self._static_kwargs(), **PHYSICS_NOPHYS
        )
        model.eval()
        model.set_physics_normalization(torch.zeros(69), torch.ones(69))
        frames, mask = self._inputs()
        with torch.no_grad():
            out, aux = model(frames, mask)
        self.assertEqual(out.shape, (2, 3, GRID_H, GRID_W, 69))
        self.assertIn("static_input", model.state_dict())
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_static_input.py::TestPredRNNv2StaticInput -v`
Expected: FAIL c `TypeError: ... unexpected keyword argument 'static_input_fields'`

- [ ] **Step 3: Реализация в `Models/PredRNN.py`**

Импорт вверху: `from utils.static_input import StaticInputMixin`.

`PredRNNv2_Model` — наследование и конструктор:

```python
class PredRNNv2_Model(StaticInputMixin, nn.Module):
```

Сигнатура:

```python
    def __init__(
        self,
        num_layers: int,
        num_hidden,
        configs: dict,
        static_input_fields: list[str] | None = None,
        static_constants_path: str | None = None,
        static_cut: list[int] | None = None,
    ) -> None:
```

В теле — после присвоения `self.num_hidden = list(num_hidden)` и ДО построения `cell_list` (нужен `num_static` для входа первого слоя; RNG не трогается):

```python
        _, _, img_height, img_width = tuple(configs["in_shape"])
        num_static = self.init_static_input(
            static_input_fields, static_constants_path, static_cut, img_height, img_width
        )
        # Кадры патчатся внутри forward, поэтому буфер статики хранится уже
        # пропатченным: (1, S, H, W) -> (1, S*p^2, H/p, W/p).
        self.static_channel = num_static * self.patch_size**2
        if num_static > 0:
            static_channels_last = self.static_input.permute(0, 2, 3, 1).unsqueeze(1)
            patched = _reshape_patch(static_channels_last, self.patch_size)
            self.static_input = patched[0].permute(0, 3, 1, 2).contiguous()
```

и в построении ячеек — вход только первого слоя:

```python
            in_channel = (
                self.frame_channel + self.static_channel if i == 0 else self.num_hidden[i - 1]
            )
```

(`conv_last` и `adapter` — без изменений: выход остаётся `frame_channel`.)

В `PredRNNv2_Model.forward` — единственная точка: вход первой ячейки (строки 324-326). Статика приклеивается ПОСЛЕ scheduled-sampling-смешивания (`net` и `x_gen` оба `frame_channel`-канальные, статика у обоих одна и та же):

```python
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                self.append_static_input(net), h_t[0], c_t[0], memory
            )
```

`_correct_step(x_gen, net)` получает `net` БЕЗ статики — физический prev-state должен остаться сырыми 69 каналами (не менять).

`PI_PredRNNv2_Model.__init__` — добавить явные static-параметры до `**physics_kwargs` и пробросить:

```python
    def __init__(
        self,
        num_layers: int,
        num_hidden,
        configs: dict,
        static_input_fields: list[str] | None = None,
        static_constants_path: str | None = None,
        static_cut: list[int] | None = None,
        **physics_kwargs,
    ) -> None:
        super().__init__(
            num_layers,
            num_hidden,
            configs,
            static_input_fields=static_input_fields,
            static_constants_path=static_constants_path,
            static_cut=static_cut,
        )
```

(`PredRNN_Model` (v1) не трогаем — YAGNI, в exp24 он не участвует.)

`Models/__init__.py`: изменения НЕ нужны — `_build_predrnn_v2`/`_build_pi_predrnn_v2` пробрасывают `**params`, и static-ключи из `model.params` YAML попадут в явные kwargs. Проверить глазами, не коммитить пустую правку.

- [ ] **Step 4: Прогнать тесты (новые + существующие PredRNN)**

Run: `uv run pytest tests/test_static_input.py tests/test_pi_predrnnv2.py tests/test_predrnn_aux_loss.py -v`
Expected: PASS

- [ ] **Step 5: Линт и коммит**

```bash
ruff check .
git add Models/PredRNN.py tests/test_static_input.py
git commit -m "feat(exp24): optional static input channels in PredRNNv2/PI-PredRNNv2"
```

---

### Task 6: генератор конфигов и 6 YAML exp24

**Files:**
- Create: `docs/experiments/24_static_inputs/make_configs.py`
- Create: `configs/exp24/exp24_{iam4vp,predrnnv2,simvpv2}_{nophys,a2}_static_usa.yaml` (генерируются скриптом)
- Test: `tests/test_static_input.py` (дописать класс)

**Interfaces:**
- Consumes: родительские конфиги `configs/abl16_long/abl16_r0_no_physics_t12.yaml`, `configs/abl16_long/abl16_r3_a2_exp13_t12.yaml`, `configs/exp20/exp20_p0_no_physics_s0.yaml`, `configs/exp20/exp20_p3_a2_exp13_s0.yaml`, `configs/exp21_long/exp21_s0_no_physics_t12_s0.yaml`, `configs/exp21_long/exp21_s3_a2_exp13_t12_s0.yaml`.
- Produces: функция `staticize(config: dict, family: str, arm: str) -> dict` (чистая, тестируемая) и `main()`, пишущий 6 YAML.

- [ ] **Step 1: Написать падающий тест**

В `tests/test_static_input.py` (импорты вверх файла: `import sys`, `from pathlib import Path`, `import yaml`; путь к генератору добавить через список модулей нельзя — грузим как данные):

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "docs" / "experiments" / "24_static_inputs"))
import make_configs  # noqa: E402  (генератор лежит вне пакетов, как в exp22)


class TestMakeConfigsExp24(unittest.TestCase):
    def test_staticize_adds_static_block_and_renames(self) -> None:
        base = yaml.safe_load(
            (REPO_ROOT / "configs/abl16_long/abl16_r0_no_physics_t12.yaml").read_text()
        )
        cfg = make_configs.staticize(base, "iam4vp", "nophys")
        params = cfg["model"]["params"]
        self.assertEqual(params["static_input_fields"], ["orography", "lsm"])
        self.assertEqual(params["static_cut"], [75, 107, 164, 228])
        self.assertTrue(params["static_constants_path"].endswith("constants_1.40625deg.nc"))
        self.assertEqual(cfg["experiment"]["name"], "exp24-iam4vp-nophys-static-usa-s0")

    def test_staticize_keeps_budget_and_physics(self) -> None:
        base_path = REPO_ROOT / "configs/exp21_long/exp21_s3_a2_exp13_t12_s0.yaml"
        base = yaml.safe_load(base_path.read_text())
        expected_epoch = base["training"]["max_epoch"]
        expected_mode = base["model"]["params"]["physics_feature_mode"]
        cfg = make_configs.staticize(base, "simvpv2", "a2")
        self.assertEqual(cfg["training"]["max_epoch"], expected_epoch)
        self.assertEqual(cfg["model"]["params"]["physics_feature_mode"], expected_mode)
        self.assertEqual(cfg["data"]["cut"], [[75, 107], [164, 228]])

    def test_staticize_rejects_non_usa_parent(self) -> None:
        base = yaml.safe_load(
            (REPO_ROOT / "configs/exp22/exp22_iam4vp_a2_npac.yaml").read_text()
        )
        with self.assertRaises(ValueError):
            make_configs.staticize(base, "iam4vp", "a2")

    def test_generated_configs_match_generator(self) -> None:
        """Все 6 закоммиченных YAML — ровно то, что выдаёт генератор."""
        for (family, arm), base_rel in make_configs.BASE_CONFIGS.items():
            base = yaml.safe_load((REPO_ROOT / base_rel).read_text())
            expected = make_configs.staticize(base, family, arm)
            out_path = (
                REPO_ROOT / "configs/exp24" / f"exp24_{family}_{arm}_static_usa.yaml"
            )
            self.assertTrue(out_path.exists(), f"нет {out_path} — прогони make_configs")
            self.assertEqual(yaml.safe_load(out_path.read_text()), expected)
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_static_input.py::TestMakeConfigsExp24 -v`
Expected: FAIL c `ModuleNotFoundError: No module named 'make_configs'`

- [ ] **Step 3: Реализовать генератор**

`docs/experiments/24_static_inputs/make_configs.py`:

```python
"""Генератор 6 конфигов exp24 из USA-конфигов баз exp16-long/exp20/exp21-long.

exp24 = 3 семейства × 2 арма (no_physics / A2 exp13), USA, seed 0, + статические
входные каналы (орография, lsm). Родительский конфиг определяет ВСЁ (бюджет
эпох, батч, lr, окна данных): пары «static vs база» сравниваются в одинаковом
харнессе. Скрипт добавляет РОВНО три ключа в ``model.params`` и меняет
``experiment.name`` (spec: docs/superpowers/specs/2026-07-16-exp24-static-inputs-design.md).

Запуск (локально):
    python docs/experiments/24_static_inputs/make_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "configs" / "exp24"

USA_CUT = [[75, 107], [164, 228]]
USA_CUT_FLAT = [75, 107, 164, 228]
STATIC_FIELDS = ["orography", "lsm"]
CONSTANTS_PATH = (
    "/home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc"
)

# (семейство, арм) -> базовый USA-конфиг (он же — бюджет эпох и весь харнесс).
BASE_CONFIGS: dict[tuple[str, str], str] = {
    ("iam4vp", "nophys"): "configs/abl16_long/abl16_r0_no_physics_t12.yaml",
    ("iam4vp", "a2"): "configs/abl16_long/abl16_r3_a2_exp13_t12.yaml",
    ("predrnnv2", "nophys"): "configs/exp20/exp20_p0_no_physics_s0.yaml",
    ("predrnnv2", "a2"): "configs/exp20/exp20_p3_a2_exp13_s0.yaml",
    ("simvpv2", "nophys"): "configs/exp21_long/exp21_s0_no_physics_t12_s0.yaml",
    ("simvpv2", "a2"): "configs/exp21_long/exp21_s3_a2_exp13_t12_s0.yaml",
}


def staticize(config: dict, family: str, arm: str) -> dict:
    """Добавить static-блок в загруженный USA-конфиг (мутирует и возвращает).

    Args:
        config: разобранный YAML родительского USA-конфига.
        family: ``iam4vp`` / ``predrnnv2`` / ``simvpv2``.
        arm: ``nophys`` / ``a2``.

    Returns:
        Тот же ``config`` со static-ключами и новым ``experiment.name``.

    Raises:
        ValueError: родительский конфиг не USA (гард от неверной базы).
    """
    if config["data"]["cut"] != USA_CUT:
        raise ValueError(
            f"Родитель не USA: data.cut={config['data']['cut']}, ожидался {USA_CUT}"
        )
    config["experiment"]["name"] = f"exp24-{family}-{arm}-static-usa-s0"
    params = config["model"]["params"]
    params["static_input_fields"] = list(STATIC_FIELDS)
    params["static_constants_path"] = CONSTANTS_PATH
    params["static_cut"] = list(USA_CUT_FLAT)
    return config


def main() -> None:
    """Сгенерировать 6 конфигов в ``configs/exp24/``. Side effect: пишет YAML."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (family, arm), base_rel in BASE_CONFIGS.items():
        base_path = REPO_ROOT / base_rel
        assert base_path.exists(), f"нет базового конфига {base_path}"
        config = staticize(yaml.safe_load(base_path.read_text()), family, arm)
        out_path = OUT_DIR / f"exp24_{family}_{arm}_static_usa.yaml"
        header = (
            f"# GENERATED by docs/experiments/24_static_inputs/make_configs.py "
            f"from {base_rel}.\n# Do not edit by hand — regenerate. "
            f"Arm={arm}, family={family}, region=usa.\n"
        )
        body = yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True, default_flow_style=False
        )
        out_path.write_text(header + body)
        written += 1
    print(f"[make_configs] {written} конфигов -> {OUT_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Сгенерировать конфиги и прогнать тест**

```bash
python docs/experiments/24_static_inputs/make_configs.py
ls configs/exp24/   # 6 файлов
uv run pytest tests/test_static_input.py::TestMakeConfigsExp24 -v
```

Expected: `[make_configs] 6 конфигов -> .../configs/exp24`, тесты PASS

- [ ] **Step 5: Смоук — модель конструируется из сгенерированного конфига**

Быстрая ручная проверка (не тест — константы в конфиге указывают на кластерный путь; подменяем на синтетику):

```bash
python - <<'EOF'
import yaml, torch, numpy as np, h5netcdf, tempfile, os
import Models  # noqa: F401 — регистрирует модели
from utils.registry import get_model

tmp = tempfile.mkdtemp()
nc = os.path.join(tmp, "constants.nc")
with h5netcdf.File(nc, "w") as f:
    f.dimensions = {"lat": 128, "lon": 256}
    for name in ("orography", "lsm", "lat2d"):
        v = f.create_variable(name, ("lat", "lon"), np.float32)
        v[...] = np.random.rand(128, 256).astype(np.float32)

cfg = yaml.safe_load(open("configs/exp24/exp24_simvpv2_nophys_static_usa.yaml"))
params = cfg["model"]["params"]
params["static_constants_path"] = nc
model = get_model(cfg["model"]["type"])(**params)
print("OK:", cfg["model"]["type"], "static_input" in model.state_dict())
EOF
```

Expected: `OK: PI-SimVPv2 True`

- [ ] **Step 6: Линт и коммит**

```bash
ruff check .
git add docs/experiments/24_static_inputs/make_configs.py configs/exp24/ tests/test_static_input.py
git commit -m "feat(exp24): config generator and 6 USA static-input configs"
```

---

### Task 7: README, launcher, CHANGELOG, финальная проверка

**Files:**
- Create: `docs/experiments/24_static_inputs/README.md`
- Create: `sh_files/exp24_train.sh`
- Modify: `CHANGELOG.md` (секция `## [Unreleased]` → `### Added`)

**Interfaces:**
- Consumes: конфиги из Task 6; шаблон лончера `sh_files/exp22_train.sh`; USA-мемап `predformer_usa_2000_2004.dat`.
- Produces: документация эксперимента и sbatch-волна.

- [ ] **Step 1: README эксперимента**

`docs/experiments/24_static_inputs/README.md`:

```markdown
# Эксперимент 24 — статические входы (орография + маска суша/море)

Spec: [docs/superpowers/specs/2026-07-16-exp24-static-inputs-design.md](../../superpowers/specs/2026-07-16-exp24-static-inputs-design.md)

## Вопрос

Помогают ли модели статические поля (орография, lsm), поданные как
дополнительные входные каналы бэкбона? И меняется ли при этом выигрыш
физического приора A2-exp13?

- **H1.** Статика улучшает обе армы (no_physics и A2) по RMSE/ACC на общей эпохе.
- **H2.** Выигрыш A2 поверх статики сжимается: диабатический блок уже видит
  орографию/lsm/|lat| через `_load_static_geo`, часть его вклада может быть
  чисто географической.

## Матрица (6 ранов, USA, seed 0)

| семейство | nophys+static | a2+static | база без статики (существующие раны) |
| --- | --- | --- | --- |
| PI-IAM4VP | exp24-iam4vp-nophys-static-usa-s0 | exp24-iam4vp-a2-static-usa-s0 | abl16L r0/r3 t12 (500 эпох) |
| PI-PredRNNv2 | exp24-predrnnv2-nophys-static-usa-s0 | exp24-predrnnv2-a2-static-usa-s0 | exp20 p0/p3 (40 эпох) |
| PI-SimVPv2 | exp24-simvpv2-nophys-static-usa-s0 | exp24-simvpv2-a2-static-usa-s0 | exp21L s0/s3 t12 (500 эпох) |

Конфиги: `configs/exp24/` (генерятся `make_configs.py`, руками не править).
Бюджет/батч/lr — бит-в-бит от родителя: пара «static vs база» сравнивается в
одинаковом харнессе, армы ранжируются только на общей эпохе (уроки exp16/18).

## Механизм

`utils/static_input.py`: орография (z-score по кропу) + lsm (0..1) из
`constants_1.40625deg.nc`, буфер `(1, S, H, W)` в модели, конкат к кадрам на
входе первого слоя. Выход моделей — прежние 69 каналов; физпуть не тронут.
У PredRNNv2 статика приклеивается после scheduled-sampling-смешивания;
при patch_size>1 буфер патчится тем же патчем.

## Запуск

```bash
EXP24_JOBS="exp24_simvpv2_nophys_static_usa exp24_simvpv2_a2_static_usa" \
  bash sh_files/exp24_train.sh
# IAM4VP / PredRNNv2 — тяжёлые: NGPU=2
```

Кластерный worktree: не забыть минимальный `.env` (3 COMET_-строки) в корень.

## Протокол сравнения

Общая эпоха; парный бутстрап (`tools/metrics_ladder.py`); ограниченные метрики
(ACC/CSI/FSS) — абсолютной разницей; PSD только по её модулю. Результаты — сюда.
```

- [ ] **Step 2: Лончер**

`sh_files/exp24_train.sh` (по образцу `exp22_train.sh`, регион фиксирован — USA):

```bash
#!/bin/bash
# Волна обучающих джоб exp24 (USA, статические входы). Модель берётся из
# model.type конфига; один скрипт запускает любое из трёх семейств.
# Тяжёлым (IAM4VP/PredRNNv2) можно дать 2 GPU через NGPU=2.
#
#   EXP24_JOBS="exp24_simvpv2_nophys_static_usa exp24_simvpv2_a2_static_usa" \
#     bash sh_files/exp24_train.sh
set -euo pipefail

# Дефолтный env контракта (weatherpred-gft-fix) сломан (GLIBC_2.28); pi-iamvp несёт
# весь train-стек. Экспортим, чтобы sbatch пробросил его в джобу.
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_exp24}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2000_2004.dat}"
ACCOUNT="${ACCOUNT:-proj_1715}"
NGPU="${NGPU:-1}"
read -r -a JOBS <<<"${EXP24_JOBS:?Set EXP24_JOBS to config stems (space-separated)}"

if [[ ! -f "${MEMMAP}" ]]; then
  echo "[exp24-train] нет мемапа ${MEMMAP}" >&2
  exit 1
fi

for stem in "${JOBS[@]}"; do
  cfg="${REPO_ROOT}/configs/exp24/${stem}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[exp24-train] нет конфига ${cfg}" >&2
    exit 1
  fi
  echo "[exp24-train] submit ${stem} ngpu=${NGPU}"
  SKIP_STAGE=1 ORIG_MEMMAP="${MEMMAP}" REPO_ROOT="${REPO_ROOT}" \
    sbatch -A "${ACCOUNT}" -J "${stem}" --gres="gpu:${NGPU}" \
    "${REPO_ROOT}/sh_files/train_v4_memmap.sh" "configs/exp24/${stem}.yaml"
done
```

Проверка синтаксиса: `bash -n sh_files/exp24_train.sh` → без вывода.

- [ ] **Step 3: CHANGELOG**

В `CHANGELOG.md` под `## [Unreleased]` в `### Added` добавить:

```markdown
- exp24: optional static input channels (orography + land-sea mask) for
  IAM4VP, PredRNNv2 and SimVP families (`utils/static_input.py`,
  `static_input_fields` / `static_constants_path` / `static_cut` model params);
  6 USA configs in `configs/exp24/` + generator.
```

(если секции `### Added` под `[Unreleased]` нет — создать; формат Keep a Changelog уже используется в файле).

- [ ] **Step 4: Финальная проверка репозитория**

```bash
ruff check .                          # ноль предупреждений
ruff check --select F401,F841,T201,T100 .
uv run pytest tests/ -q               # ВЕСЬ suite, без регрессий
bash -n sh_files/exp24_train.sh
git status                            # в staged — только файлы exp24
```

Expected: pytest — все зелёные (включая 31 прежний + новые); в дереве остаются только ЧУЖИЕ незакоммиченные файлы (`tests/test_physics_sign_conventions.py`, `utils/physics.py`, `docs/planKDD.md`, `example/`).

- [ ] **Step 5: Коммит**

```bash
git add docs/experiments/24_static_inputs/README.md sh_files/exp24_train.sh CHANGELOG.md
git commit -m "docs(exp24): experiment README, sbatch launcher, changelog"
```

---

## Вне плана (после мержа кода)

- Репак/наличие USA-мемапа на кластере уже есть (`predformer_usa_2000_2004.dat`).
- Запуск волны, снятие метрик (`exp20/21_metrics_eval.sh`-образные скрипты) и
  анализ H1/H2 — отдельная сессия по результатам обучения.
- Бит-в-бит приёмка `static=None` против дерева до правки существующим
  harness-приёмом (как в deep refactor 2026-07) — вручную перед запуском волны.
