"""Опирается ли обученный корректор на физический признак? Абляция δ_phys на инференсе.

Косинус между поправкой и δ_phys равен −0.002 (см. ``closure_probe.py``), но
ортогональность **не доказывает**, что признак не используется: сеть может читать
δ_phys и выдавать нечто ему перпендикулярное. Прямой тест — отобрать признак у
**уже обученной** модели и посмотреть, развалится ли она.

Три режима на одном чекпоинте R3, одни и те же сэмплы:

* ``none``    — как есть (базис);
* ``zero``    — δ_phys ≡ 0: признака нет вовсе. Если RMSE не шелохнулся, корректор на
  него не опирается;
* ``shuffle`` — δ_phys от **чужого сэмпла** батча (штатный режим модели
  ``physics_residual_shuffle='batch'``). Это не «признака нет», а **дезинформация**:
  тенденция от другого дня. Деградация здесь означает лишь, что на признак обращают
  внимание, а не что он полезен.

Различать эти два надо жёстко. δ_phys = physics(prev) − prev — **детерминированная
функция от prev**, а prev корректор и так получает; ядро в ``stable_physical_v2`` не
имеет обучаемых параметров. Значит, признак не несёт **новой информации** и может быть
только индуктивным смещением. Отсюда «shuffle хуже базиса» и «zero равен базису»
совместимы и говорят о разном.

Запуск (кластер, CPU):
    REPO_ROOT=~/wt_fix_v2 python dphys_ablation.py \
        --checkpoint ~/abl16_long_ckpt/abl16L-r3-a2-exp13-t12-s0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --out ~/abl16_closure/dphys_ablation.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader  # noqa: E402

import Data  # noqa: E402, F401  (регистрирует датасеты)
import Models  # noqa: E402, F401  (регистрирует модели)
from utils.normalize import WeatherNormalize  # noqa: E402
from utils.registry import get_model  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    """Подключить соседний скрипт как модуль (харнесс лежит вне пакета)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml = _load("exp16_metrics_lib", HERE.parent / "metrics" / "metrics_lib.py")
me = _load("exp16_metrics_eval", HERE.parent / "metrics" / "metrics_eval.py")

MODES = ("none", "zero", "shuffle")
UPPER_AIR = slice(4, 69)  # 65 верхних каналов — те, на которые ложится поправка


def apply_mode(model, mode: str) -> None:
    """Настроить модель на режим абляции, не трогая веса.

    ``zero`` подменяет генератор приора тождеством: ``y_phys = prev`` ⇒ δ_phys ≡ 0.
    ``shuffle`` включает штатный флаг модели — тот самый контроль, что заложен в код.
    """
    model.physics_residual_shuffle = "none"
    model._physics_prior_from_state = model._prior_original
    if mode == "zero":
        model._physics_prior_from_state = lambda prev_state: prev_state
    elif mode == "shuffle":
        model.physics_residual_shuffle = "batch"


def evaluate(model, loader, normalize, weights, rollout: int, mode: str, max_samples: int):
    """Широтно-взвешенный RMSE по 65 верхним каналам в заданном режиме абляции."""
    apply_mode(model, mode)
    totals: list[torch.Tensor] = []
    seen = 0
    with torch.inference_mode():
        for x_raw, y_raw in loader:
            frames = normalize(x_raw)
            pred = normalize.denormalize(me.predict_window(model, frames, rollout))
            obs = y_raw[:, :rollout]
            totals.append(ml.weighted_rmse(pred[:, :, UPPER_AIR], obs[:, :, UPPER_AIR], weights))
            seen += x_raw.shape[0]
            if seen >= max_samples:
                break
    return torch.cat(totals).numpy(), seen


def main() -> None:
    """CLI: чекпоинт + memmap → RMSE в трёх режимах и относительные дельты к базису."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memmap", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    args: Namespace = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["config"]
    print(f"[ablation] арм={cfg['experiment']['name']} эпоха={payload.get('epoch', -1) + 1}")  # noqa: T201

    model = get_model(cfg["model"]["type"])(**cfg["model"]["params"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model._prior_original = model._physics_prior_from_state
    assert model.physics_feature_mode == "tendency", "у арма нет физического признака"

    n_channels = len(me.channel_names())
    normalize = WeatherNormalize(mean=torch.zeros(n_channels), std=torch.ones(n_channels))
    normalize.load_state_dict(payload["normalize"])
    model.set_physics_normalization(normalize.mean.view(-1), normalize.std.view(-1))

    rollout = int(cfg["training"]["extra_kwargs"]["time_prediction"])
    dataset = me.build_val_dataset(cfg["data"], args.memmap, rollout)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    weights = ml.lat_weights(32)

    scores: dict[str, np.ndarray] = {}
    for mode in MODES:
        scores[mode], seen = evaluate(
            model, loader, normalize, weights, rollout, mode, args.max_samples
        )
        print(f"[ablation] {mode}: RMSE={np.nanmean(scores[mode]):.5f} (сэмплов {seen})")  # noqa: T201

    base = np.nanmean(scores["none"])
    summary = {
        "arm": cfg["experiment"]["name"],
        "epoch": int(payload.get("epoch", -1)) + 1,
        "n_samples": int(seen),
        "rmse": {mode: float(np.nanmean(values)) for mode, values in scores.items()},
        "delta_percent_vs_intact": {
            mode: float(100.0 * (np.nanmean(values) / base - 1.0))
            for mode, values in scores.items()
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("")  # noqa: T201
    print("=" * 66)  # noqa: T201
    print("АБЛЯЦИЯ δ_phys НА ИНФЕРЕНСЕ (обученный R3, веса не тронуты)")  # noqa: T201
    print("=" * 66)  # noqa: T201
    for mode in MODES:
        delta = summary["delta_percent_vs_intact"][mode]
        print(f"  {mode:<8} RMSE {summary['rmse'][mode]:.5f}   Δ к базису {delta:+.2f} %")  # noqa: T201
    print("=" * 66)  # noqa: T201
    print(f"[ablation] записано {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
