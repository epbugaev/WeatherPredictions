"""Static input channels (exp 24): loader, mixin, model integration."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5netcdf
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics_residual import PhysicsResidualMixin  # noqa: E402
from utils.static_input import (  # noqa: E402
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
        np.testing.assert_array_equal(raw["orography"], self.fields["orography"][2:10, 4:20])
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
        torch.testing.assert_close(static[1], torch.from_numpy(self.fields["lsm"]), rtol=0, atol=0)

    def test_field_order_follows_request(self) -> None:
        ab = load_static_input_fields(self.nc_path, ["orography", "lsm"], FULL_CUT, GRID_H, GRID_W)
        ba = load_static_input_fields(self.nc_path, ["lsm", "orography"], FULL_CUT, GRID_H, GRID_W)
        torch.testing.assert_close(ab[0], ba[1], rtol=0, atol=0)

    def test_rejects_unknown_field(self) -> None:
        self.assertEqual(STATIC_INPUT_FIELDS, ("orography", "lsm"))
        with self.assertRaises(ValueError):
            load_static_input_fields(self.nc_path, ["orography", "slt"], FULL_CUT, GRID_H, GRID_W)

    def test_rejects_empty_fields(self) -> None:
        with self.assertRaises(ValueError):
            load_static_input_fields(self.nc_path, [], FULL_CUT, GRID_H, GRID_W)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            load_static_input_fields(self.nc_path, ["orography"], [0, 8, 0, 16], GRID_H, GRID_W)


class _ToyModel(StaticInputMixin, nn.Module):
    """Минимальный носитель миксина для юнит-тестов."""

    def __init__(self, nc_path: str | None, fields: list[str] | None) -> None:
        super().__init__()
        self.num_static = self.init_static_input(fields, nc_path, FULL_CUT, GRID_H, GRID_W)


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


class TestLoadStaticGeoUnchanged(StaticInputFileMixin):
    """Регресс-гард рефакторинга: _load_static_geo бит-в-бит как старая формула."""

    def test_matches_legacy_formula(self) -> None:
        geo = PhysicsResidualMixin._load_static_geo(self.nc_path, FULL_CUT, GRID_H, GRID_W)
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
        torch.testing.assert_close(geo, torch.from_numpy(expected).float(), rtol=0, atol=0)

    def test_requires_path(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsResidualMixin._load_static_geo(None, FULL_CUT, GRID_H, GRID_W)


if __name__ == "__main__":
    unittest.main()
