"""Гард генератора конфигов exp26 (кросс-региональные статические входы).

Проверяет, что закоммиченные YAML в ``configs/exp26/`` — ровно выход
``make_configs.py``, и что регионализация трогает только геометрию (cut,
static_cut, diabatic_cut, physics_lat_start_deg, имя), а бюджет/физрежим —
наследуются от USA-родителя exp24.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "docs" / "experiments" / "26_cross_region_static"))

import make_configs  # noqa: E402


class TestMakeConfigsExp26(unittest.TestCase):
    def test_generated_configs_match_generator(self) -> None:
        for base_rel in make_configs.BASE_CONFIGS:
            base_stem = Path(base_rel).stem
            for region_key, region in make_configs.REGIONS.items():
                base = yaml.safe_load((REPO_ROOT / base_rel).read_text())
                expected = make_configs.regionalize(base, base_stem, region_key, region)
                stem = make_configs.out_stem(base_stem, region_key)
                out_path = REPO_ROOT / "configs/exp26" / f"{stem}.yaml"
                self.assertTrue(out_path.exists(), f"нет {out_path} — прогони make_configs")
                self.assertEqual(yaml.safe_load(out_path.read_text()), expected)

    def test_regionalize_sets_geometry_and_keeps_budget(self) -> None:
        base_rel = "configs/exp24/exp24_iam4vp_a2_static_usa.yaml"
        base = yaml.safe_load((REPO_ROOT / base_rel).read_text())
        expected_epoch = base["training"]["max_epoch"]
        expected_mode = base["model"]["params"]["physics_feature_mode"]
        cfg = make_configs.regionalize(
            base, Path(base_rel).stem, "france", make_configs.REGIONS["france"]
        )
        self.assertEqual(cfg["experiment"]["name"], "exp26-iam4vp-a2-static-france-s0")
        self.assertEqual(cfg["data"]["cut"], [[81, 113], [0, 64]])
        self.assertEqual(cfg["model"]["params"]["static_cut"], [81, 113, 0, 64])
        self.assertEqual(cfg["model"]["params"]["diabatic_cut"], [81, 113, 0, 64])
        self.assertEqual(cfg["model"]["params"]["physics_lat_start_deg"], 26.71875)
        # Бюджет и физрежим наследуются от USA-родителя — регион их не трогает.
        self.assertEqual(cfg["training"]["max_epoch"], expected_epoch)
        self.assertEqual(cfg["model"]["params"]["physics_feature_mode"], expected_mode)

    def test_nophys_arm_has_no_diabatic_cut(self) -> None:
        base_rel = "configs/exp24/exp24_iam4vp_nophys_orog_usa.yaml"
        base = yaml.safe_load((REPO_ROOT / base_rel).read_text())
        cfg = make_configs.regionalize(
            base, Path(base_rel).stem, "npac", make_configs.REGIONS["npac"]
        )
        self.assertNotIn("diabatic_cut", cfg["model"]["params"])
        self.assertEqual(cfg["model"]["params"]["static_cut"], [74, 106, 96, 160])


if __name__ == "__main__":
    unittest.main()
