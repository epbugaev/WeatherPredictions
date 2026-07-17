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
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.IAM4VP import IAM4VP  # noqa: E402
from Models.PredRNN import PI_PredRNNv2_Model, PredRNNv2_Model  # noqa: E402
from Models.SimVP import PI_SimVP_Model, SimVP_Model  # noqa: E402
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
            in_shape=SIMVP_SHAPE,
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=2,
            **self._static_kwargs(),
        )
        model.eval()
        x = torch.randn(2, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (2, 2, 69, GRID_H, GRID_W))

    def test_disabled_matches_param_absence_bitexact(self) -> None:
        """None-ветка консистентна (SimVP); кросс-версия — голдены exp16 + harness-приёмка."""
        torch.manual_seed(0)
        plain = SimVP_Model(in_shape=SIMVP_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2)
        torch.manual_seed(0)
        disabled = SimVP_Model(
            in_shape=SIMVP_SHAPE,
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=2,
            static_input_fields=None,
        )
        self.assertEqual(list(plain.state_dict().keys()), list(disabled.state_dict().keys()))
        plain.eval()
        disabled.eval()
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            torch.testing.assert_close(plain(x), disabled(x), rtol=0, atol=0)

    def test_pi_model_with_static_and_physics(self) -> None:
        torch.manual_seed(0)
        model = PI_SimVP_Model(
            in_shape=SIMVP_SHAPE,
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=2,
            **self._static_kwargs(),
            **PHYSICS_NOPHYS,
        )
        model.eval()
        model.set_physics_normalization(torch.zeros(69), torch.ones(69))
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (1, 2, 69, GRID_H, GRID_W))
        self.assertIn("static_input", model.state_dict())


class TestIAM4VPStaticInput(StaticInputFileMixin):
    def _build(self, **kwargs) -> IAM4VP:
        torch.manual_seed(0)
        # hid_S=64 (not 8): Models/IAM4VP_utils.py ConvNeXt_block/_bottle hardcode
        # nn.Linear(64, dim) for the time embedding, so Time_MLP's output width
        # must be 64 regardless of hid_S — a pre-existing constraint, unrelated
        # to static-input channels.
        model = IAM4VP(
            T_data=2,
            C_data=69,
            H_data=GRID_H,
            W_data=GRID_W,
            hid_S=64,
            N_S=4,
            N_T=2,
            use_physics=False,
            **kwargs,
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
        """None-ветка консистентна (IAM4VP); кросс-версия — голдены exp16 + harness-приёмка."""
        plain = self._build()
        disabled = self._build(static_input_fields=None)
        self.assertEqual(list(plain.state_dict().keys()), list(disabled.state_dict().keys()))
        x = torch.randn(1, 2, 69, GRID_H, GRID_W)
        t = torch.full((1,), 100.0)
        with torch.no_grad():
            torch.testing.assert_close(plain(x, None, t), disabled(x, None, t), rtol=0, atol=0)


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
        """None-ветка консистентна (PredRNNv2); кросс-версия — голдены exp16 + harness-приёмка."""
        torch.manual_seed(0)
        plain = PredRNNv2_Model(2, (8, 8), _predrnn_configs(1))
        torch.manual_seed(0)
        disabled = PredRNNv2_Model(2, (8, 8), _predrnn_configs(1), static_input_fields=None)
        self.assertEqual(list(plain.state_dict().keys()), list(disabled.state_dict().keys()))
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


sys.path.insert(0, str(REPO_ROOT / "docs" / "experiments" / "24_static_inputs"))
import make_configs  # noqa: E402


class TestMakeConfigsExp24(unittest.TestCase):
    def test_staticize_adds_static_block_and_renames(self) -> None:
        base = yaml.safe_load(
            (REPO_ROOT / "configs/abl16_long/abl16_r0_no_physics_t12.yaml").read_text()
        )
        cfg = make_configs.staticize(base, "iam4vp", "nophys", ["orography", "lsm"], "static", 0)
        params = cfg["model"]["params"]
        self.assertEqual(params["static_input_fields"], ["orography", "lsm"])
        self.assertEqual(params["static_cut"], [75, 107, 164, 228])
        self.assertTrue(params["static_constants_path"].endswith("constants_1.40625deg.nc"))
        self.assertEqual(cfg["experiment"]["name"], "exp24-iam4vp-nophys-static-usa-s0")
        self.assertEqual(cfg["training"]["seed"], 0)

    def test_staticize_seed_and_orog_tag(self) -> None:
        base = yaml.safe_load(
            (REPO_ROOT / "configs/abl16_long/abl16_r3_a2_exp13_t12.yaml").read_text()
        )
        cfg = make_configs.staticize(base, "iam4vp", "a2", ["orography"], "orog", 1)
        self.assertEqual(cfg["experiment"]["name"], "exp24-iam4vp-a2-orog-usa-s1")
        self.assertEqual(cfg["training"]["seed"], 1)
        self.assertEqual(cfg["model"]["params"]["static_input_fields"], ["orography"])
        self.assertEqual(
            make_configs.config_stem("iam4vp", "a2", "orog", 1), "exp24_iam4vp_a2_orog_usa_s1"
        )
        self.assertEqual(
            make_configs.config_stem("iam4vp", "a2", "orog", 0), "exp24_iam4vp_a2_orog_usa"
        )

    def test_staticize_keeps_budget_and_physics(self) -> None:
        base_path = REPO_ROOT / "configs/exp21_long/exp21_s3_a2_exp13_t12_s0.yaml"
        base = yaml.safe_load(base_path.read_text())
        expected_epoch = base["training"]["max_epoch"]
        expected_mode = base["model"]["params"]["physics_feature_mode"]
        cfg = make_configs.staticize(base, "simvpv2", "a2", ["orography", "lsm"], "static", 0)
        self.assertEqual(cfg["training"]["max_epoch"], expected_epoch)
        self.assertEqual(cfg["model"]["params"]["physics_feature_mode"], expected_mode)
        self.assertEqual(cfg["data"]["cut"], [[75, 107], [164, 228]])

    def test_staticize_rejects_non_usa_parent(self) -> None:
        base = yaml.safe_load((REPO_ROOT / "configs/exp22/exp22_iam4vp_a2_npac.yaml").read_text())
        with self.assertRaises(ValueError):
            make_configs.staticize(base, "iam4vp", "a2", ["orography", "lsm"], "static", 0)

    def test_generated_configs_match_generator(self) -> None:
        """Все закоммиченные YAML exp24 — ровно то, что выдаёт генератор (все JOBS)."""
        for family, arm, tag, seed in make_configs.JOBS:
            base_rel = make_configs.BASE_CONFIGS[(family, arm)]
            base = yaml.safe_load((REPO_ROOT / base_rel).read_text())
            fields = make_configs.STATIC_SETS[tag]
            expected = make_configs.staticize(base, family, arm, fields, tag, seed)
            stem = make_configs.config_stem(family, arm, tag, seed)
            out_path = REPO_ROOT / "configs/exp24" / f"{stem}.yaml"
            self.assertTrue(out_path.exists(), f"нет {out_path} — прогони make_configs")
            self.assertEqual(yaml.safe_load(out_path.read_text()), expected)


if __name__ == "__main__":
    unittest.main()
