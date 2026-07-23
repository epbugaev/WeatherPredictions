"""Генератор региональных конфигов exp22 из USA-конфигов.

exp22 = 3 семейства × 5 армов (no_physics / A2 exp13 / legacy / A2-noQ / +exp14)
× 2 региона (France/Europe, North Pacific). Изначальная волна несла только 3
арма (no_physics/A2/legacy); A2-noQ и +exp14 добавлены расширением, чтобы
покрыть остальную физ-лестницу exp16/20/21 (кроме +exp15 — тот требует
region-specific климатологии `eq15_clim_summary_<region>_2000.npz`, которой нет
ни для France, ни для NPac; это отдельная задача подготовки данных, не входит
в это расширение). Регион меняет РОВНО несколько полей, поэтому конфиги не
пишутся руками, а генерируются из существующих USA-арм-конфигов подстановкой
региональных значений (DRY, спек §6):

  * ``data.cut`` — окно кропа региона (оба 32×64, спек §3);
  * ``model.params.physics_lat_start_deg`` — широта первого латентного ряда
    (только если поле есть: у legacy-арма его намеренно нет — нормализованный
    режим геометрию не использует);
  * ``model.params.diabatic_cut`` — тот же cut плоским списком (только у A2-арма
    с диабатикой; глобальные константы режутся этим окном);
  * ``experiment.name`` → ``exp22-<fam>-<arm>-<region>-s0``;
  * общий бюджет: ``training.max_epoch = 150``, early-stopping снят (гарантия
    ОБЩЕЙ эпохи для честного ранжирования, exp16 §11.3), ``training.seed = 0``.

Геометрия НЕ трогается там, где её нет, и ``physics_dlat_deg/dlon`` (5.625)
регион-независимы. Мемап подаётся через ``ORIG_MEMMAP`` env при запуске, поэтому
``memmap_path`` остаётся ``null``.

Запуск (локально):
    python docs/experiments/22_cross_region_physics/make_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "configs" / "exp22"

# Регионы: cut (вложенный, для data.cut), тот же плоским (для diabatic_cut) и
# широта первого латентного ряда physics_lat_start_deg = -90 + (lat0+2)*1.40625.
REGIONS: dict[str, dict] = {
    "france": {
        "cut": [[81, 113], [0, 64]],
        "cut_flat": [81, 113, 0, 64],
        "physics_lat_start_deg": 26.71875,
    },
    "npac": {
        "cut": [[74, 106], [96, 160]],
        "cut_flat": [74, 106, 96, 160],
        "physics_lat_start_deg": 16.875,
    },
}

# (семейство, арм) -> базовый USA-конфиг. Арм-ключи едины для всех семейств.
BASE_CONFIGS: dict[tuple[str, str], str] = {
    ("iam4vp", "nophys"): "configs/pi_iam4vp_residual_no_physics_usa_v4.yaml",
    ("iam4vp", "a2"): "configs/pi_iam4vp_residual_diabatic_usa_v4.yaml",
    ("iam4vp", "legacy"): "configs/pi_iam4vp_residual_legacy_hybrid_usa_v4.yaml",
    ("iam4vp", "a2noq"): "configs/pi_iam4vp_residual_no_diabatic_usa_v4.yaml",
    ("iam4vp", "exp14"): "configs/pi_iam4vp_residual_exp14_usa_v4.yaml",
    ("predrnnv2", "nophys"): "configs/exp20/exp20_p0_no_physics_s0.yaml",
    ("predrnnv2", "a2"): "configs/exp20/exp20_p3_a2_exp13_s0.yaml",
    ("predrnnv2", "legacy"): "configs/exp20/exp20_p1_legacy_hybrid_s0.yaml",
    ("predrnnv2", "a2noq"): "configs/exp20/exp20_p3a_no_diabatic_s0.yaml",
    ("predrnnv2", "exp14"): "configs/exp20/exp20_p4_exp14_s0.yaml",
    ("simvpv2", "nophys"): "configs/exp21/exp21_s0_no_physics_s0.yaml",
    ("simvpv2", "a2"): "configs/exp21/exp21_s3_a2_exp13_s0.yaml",
    ("simvpv2", "legacy"): "configs/exp21/exp21_s1_legacy_hybrid_s0.yaml",
    ("simvpv2", "a2noq"): "configs/exp21/exp21_s3a_no_diabatic_s0.yaml",
    ("simvpv2", "exp14"): "configs/exp21/exp21_s4_exp14_s0.yaml",
}

# Новые (family, arm) добавленные этим расширением — используются лончером,
# чтобы отличить "досчитать существующие" от "запустить с нуля".
EXTENSION_ARMS = {"a2noq", "exp14"}

# Бюджет эпох — пер-семейный. PredRNNv2 на 150 эпох не влезает в дедлайн аллокации
# кластера (~27 ч, ресюма нет; ~16 мин/эпоху у физ-арма → 150 эп ≈ 40 ч), поэтому у
# него общая эпоха 40 (как exp20). Внутрисемейное сравнение честно (все 3 арма на 40);
# кросс-семейное — по относительной Δ к no_physics, разные эпохи между семействами ОК.
MAX_EPOCH_BY_FAMILY = {"iam4vp": 150, "simvpv2": 150, "predrnnv2": 40}
EARLY_STOP_DISABLED = 100_000  # никогда не срабатывает -> все армы доходят до бюджета


def regionalize(config: dict, family: str, arm: str, region_key: str, region: dict) -> dict:
    """Подставить региональные поля в загруженный конфиг арма (мутирует и возвращает).

    Args:
        config: разобранный YAML базового USA-конфига.
        family: ``iam4vp`` / ``predrnnv2`` / ``simvpv2``.
        arm: ``nophys`` / ``a2`` / ``legacy``.
        region_key: ``france`` / ``npac``.
        region: запись `REGIONS` (cut, cut_flat, physics_lat_start_deg).

    Returns:
        Тот же ``config`` с региональными значениями и общим бюджетом эпох.
    """
    config["experiment"]["name"] = f"exp22-{family}-{arm}-{region_key}-s0"
    config["data"]["cut"] = [list(pair) for pair in region["cut"]]

    params = config["model"]["params"]
    if "physics_lat_start_deg" in params:
        params["physics_lat_start_deg"] = region["physics_lat_start_deg"]
    if "diabatic_cut" in params:
        params["diabatic_cut"] = list(region["cut_flat"])

    training = config["training"]
    training["max_epoch"] = MAX_EPOCH_BY_FAMILY[family]
    training["seed"] = 0
    if "early_stopping_patience" in training:
        training["early_stopping_patience"] = EARLY_STOP_DISABLED
    return config


def main() -> None:
    """Сгенерировать 18 конфигов в ``configs/exp22/``. Side effect: пишет YAML на диск."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (family, arm), base_rel in BASE_CONFIGS.items():
        base_path = REPO_ROOT / base_rel
        assert base_path.exists(), f"нет базового конфига {base_path}"
        for region_key, region in REGIONS.items():
            # Свежая загрузка на каждый регион — regionalize мутирует словарь.
            config = regionalize(
                yaml.safe_load(base_path.read_text()), family, arm, region_key, region
            )
            out_path = OUT_DIR / f"exp22_{family}_{arm}_{region_key}.yaml"
            header = (
                f"# GENERATED by docs/experiments/22_cross_region_physics/make_configs.py "
                f"from {base_rel}.\n# Do not edit by hand — regenerate. Region={region_key}, "
                f"arm={arm}, family={family}.\n"
            )
            body = yaml.safe_dump(
                config, sort_keys=False, allow_unicode=True, default_flow_style=False
            )
            out_path.write_text(header + body)
            written += 1
    print(f"[make_configs] {written} конфигов -> {OUT_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
