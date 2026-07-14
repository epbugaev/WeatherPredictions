"""Предпроверка «замыкание или свободный источник» на ОБУЧЕННОМ чекпоинте R3.

Обучать ничего не надо: арм R3 уже сошёлся, и мы просто смотрим, **с чем скоррелирован
выход Q_θ**. Две несовместимые версии, обе согласуются со всем, что измерено до сих пор:

* **замыкание** — Q_θ чинит структурную дырку уравнений (влажностный бюджет ядра
  односторонний: конденсация строго отрицательна, испарения и подсеточной конвекции
  нет вовсе). Тогда его выход должен быть привязан к внутренним полям ядра — к режиму
  насыщенного подъёма и к ω — и **без уравнений чинить будет нечего**, значит арм X1
  обязан оказаться заметно слабее −2.98 %;
* **свободный источник** — Q_θ выучил испарение и конвекцию сам, из состояния и
  географии. Тогда его выход объясняется орографией, сушей и широтой, к ядру он
  безразличен, и **X1 ≈ −3 %**, а уравнения — декорация.

Предсказания зарегистрированы ЗАРАНЕЕ, до запуска: docs/architecture.md §7.6.

Запуск (кластер, CPU):
    REPO_ROOT=~/wt_fix_v2 python closure_probe.py \
        --checkpoint ~/abl16_long_ckpt/abl16L-r3-a2-exp13-t12-s0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --out ~/abl16_closure/closure_r3.npz
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
import torch.nn.functional as F

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


cl = _load("exp16_closure_lib", HERE / "closure_lib.py")
me = _load("exp16_metrics_eval", HERE.parent / "metrics" / "metrics_eval.py")

# Раскладка 65 верхних каналов, на которые ложится поправка: z, t, r, u, v по 13 уровней.
T_BLOCK = slice(13, 26)
Q_BLOCK = slice(26, 39)
COSINE_KEY = "physics_residual_correction_to_tendency_cosine"


class KernelTap:
    """Съём внутренних полей PDE-ядра, которые оно наружу не отдаёт: ``cond`` и ``w``.

    Дублировать в зонде подготовку физического состояния (денормировка, r→q, санитайз,
    даунсэмпл на 8×16) — значит завести вторую копию логики, которая молча разъедется с
    моделью, и тогда диагностика будет врать, не падая. Поэтому оборачиваем методы
    **экземпляра** ядра: снятое не может отстать от реально посчитанного.

    За один forward модели ядро зовётся 9 раз (3 вызова блока × глубина 3). Нужен
    **первый**: только он считается на настоящем ``prev_state``, остальные — на уже
    проэволюционировавших промежуточных состояниях.
    """

    def __init__(self, kernels: list[torch.nn.Module]) -> None:
        self.cond: list[torch.Tensor] = []
        self.w: list[torch.Tensor] = []
        for kernel in kernels:
            self._wrap(kernel)

    def _wrap(self, kernel: torch.nn.Module) -> None:
        original_cond = kernel._condensation_source
        original_w = kernel.get_w

        def cond_tap(t, q, w, _original=original_cond):
            out = _original(t, q, w)
            self.cond.append(out.detach())
            return out

        def w_tap(u, v, _original=original_w):
            out = _original(u, v)
            self.w.append(out.detach())
            return out

        kernel._condensation_source = cond_tap
        kernel.get_w = w_tap

    def reset(self) -> None:
        self.cond.clear()
        self.w.clear()

    def first(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cond, w)`` первого вызова ядра — того, что считался на настоящем prev_state."""
        assert self.cond and self.w, "ядро не звалось: у арма выключена физика?"
        return self.cond[0], self.w[0]


class HeadTap:
    """Выход Q_θ через forward-hook: модель его не публикует, только RMS.

    Хук снимает выход головы ДО канальной маски, но на срезах t и q маска равна единице
    (``t_and_q``), поэтому на интересующих нас каналах сырой выход совпадает с
    применённым.
    """

    def __init__(self, head: torch.nn.Module) -> None:
        self.output: torch.Tensor | None = None
        head.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        self.output = output.detach()


def upsample_to(field: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Латентная сетка 8×16 → сетка данных 32×64 — тем же билинейным путём, что и модель."""
    return F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)


def probe(args: Namespace) -> dict[str, np.ndarray]:
    """Прогнать вал и снять связь выхода Q_θ с полями ядра и со статической географией."""
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["config"]
    arm = cfg.get("experiment", {}).get("name", Path(args.checkpoint).parent.name)
    epoch = int(payload.get("epoch", -1)) + 1
    print(f"[closure] арм={arm} эпоха={epoch}", flush=True)  # noqa: T201

    model = get_model(cfg["model"]["type"])(**cfg["model"].get("params", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    assert model.diabatic_head is not None, "у арма нет Q_θ — предпроверка бессмысленна"

    n_channels = len(me.channel_names())
    normalize = WeatherNormalize(mean=torch.zeros(n_channels), std=torch.ones(n_channels))
    normalize.load_state_dict(payload["normalize"])
    model.set_physics_normalization(normalize.mean.view(-1), normalize.std.view(-1))

    kernel_tap = KernelTap(list(model.hybrid_block.pde_block.PDE_kernels))
    head_tap = HeadTap(model.diabatic_head)
    geo = model.diabatic_geo  # (1, 3, H, W)

    rollout = int(cfg["training"]["extra_kwargs"]["time_prediction"])
    dataset = me.build_val_dataset(cfg["data"], args.memmap, rollout)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    keys = (
        "r_q_cond",
        "r_q_omega",
        "r_t_cond",
        "r_t_omega",
        "ev_geo_q",
        "ev_kernel_q",
        "ev_geo_t",
        "ev_kernel_t",
        "q_mean",
        "t_mean",
    )
    records: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    contrasts: list[dict[str, float]] = []
    cosines: list[float] = []
    seen = 0

    with torch.inference_mode():
        for x_raw, _ in loader:
            frames = normalize(x_raw)
            predictions: list[torch.Tensor] = []
            for idx_time in range(rollout):
                kernel_tap.reset()
                step = torch.full((frames.shape[0],), (idx_time + 1) * 100.0)
                predictions.append(model(frames, predictions, step).detach())
                _accumulate(model, kernel_tap, head_tap, geo, records, contrasts, cosines)
            seen += frames.shape[0]
            print(f"[closure] сэмплов: {seen}", flush=True)  # noqa: T201
            if seen >= args.max_samples:
                break

    out = {key: np.concatenate(values) for key, values in records.items()}
    out["contrast"] = np.array([c["contrast"] for c in contrasts])
    out["contrast_inside"] = np.array([c["inside"] for c in contrasts])
    out["contrast_outside"] = np.array([c["outside"] for c in contrasts])
    out["mask_fraction"] = np.array([c["mask_fraction"] for c in contrasts])
    out["cosine_correction_vs_dphys"] = np.array(cosines)
    out["n_samples"] = np.array([seen])
    out["epoch"] = np.array([epoch])
    return out


def _accumulate(model, kernel_tap, head_tap, geo, records, contrasts, cosines) -> None:
    """Один шаг роллаута → корреляции, R² и контраст режима, разложенные по уровням."""
    diagnostics = model._last_residual_diagnostics
    if COSINE_KEY in diagnostics:
        cosines.append(float(diagnostics[COSINE_KEY]))

    q_theta = head_tap.output  # (B, 65, H, W)
    cond_coarse, w_coarse = kernel_tap.first()
    height, width = q_theta.shape[-2:]
    cond = upsample_to(cond_coarse, height, width).numpy().astype(np.float64)
    omega = (-100.0 * upsample_to(w_coarse, height, width)).numpy().astype(np.float64)

    q_out = q_theta[:, Q_BLOCK].numpy().astype(np.float64)
    t_out = q_theta[:, T_BLOCK].numpy().astype(np.float64)

    records["r_q_cond"].append(cl.spatial_pearson(q_out, cond).ravel())
    records["r_q_omega"].append(cl.spatial_pearson(q_out, omega).ravel())
    records["r_t_cond"].append(cl.spatial_pearson(t_out, cond).ravel())
    records["r_t_omega"].append(cl.spatial_pearson(t_out, omega).ravel())
    records["q_mean"].append(q_out.mean(axis=(-2, -1)).ravel())
    records["t_mean"].append(t_out.mean(axis=(-2, -1)).ravel())

    # R²: география против полей ядра — обе гипотезы на одной сетке, по уровням.
    rows = q_out.shape[0] * q_out.shape[1]
    flat_q = q_out.reshape(rows, height, width)
    flat_t = t_out.reshape(rows, height, width)
    geo_fields = np.broadcast_to(
        geo.numpy().astype(np.float64), (rows, geo.shape[1], height, width)
    )
    kernel_fields = np.stack(
        [cond.reshape(rows, height, width), omega.reshape(rows, height, width)], axis=1
    )
    records["ev_geo_q"].append(cl.explained_variance(flat_q, geo_fields))
    records["ev_kernel_q"].append(cl.explained_variance(flat_q, kernel_fields))
    records["ev_geo_t"].append(cl.explained_variance(flat_t, geo_fields))
    records["ev_kernel_t"].append(cl.explained_variance(flat_t, kernel_fields))

    contrasts.append(cl.regime_contrast(q_out, cond < 0))


def report(out: dict[str, np.ndarray]) -> str:
    """Сводка + прочтение в терминах предсказания для арма X1."""
    lines = [
        "",
        "=" * 74,
        f"ПРЕДПРОВЕРКА ЗАМЫКАНИЯ — чекпоинт R3, эпоха {int(out['epoch'][0])}, "
        f"сэмплов {int(out['n_samples'][0])}",
        "=" * 74,
        "Корреляция выхода Q_θ с полями ядра (по сэмпл × шаг × уровень):",
    ]
    for key in ("r_q_cond", "r_q_omega", "r_t_cond", "r_t_omega"):
        s = cl.summarize(out[key])
        lines.append(
            f"  {key:<11} r = {s['mean']:+.3f}  медиана {s['median']:+.3f}  "
            f"доля |r|>0.3: +{s['frac_pos']:.0%} / −{s['frac_neg']:.0%}  "
            f"валидных {s['frac_valid']:.0%}"
        )
    lines.append("")
    lines.append("Доля дисперсии Q_θ, объяснённая — ГЕОГРАФИЕЙ против ПОЛЕЙ ЯДРА:")
    for key in ("ev_geo_q", "ev_kernel_q", "ev_geo_t", "ev_kernel_t"):
        s = cl.summarize(out[key])
        lines.append(f"  {key:<11} R² = {s['mean']:.3f}  медиана {s['median']:.3f}")
    lines.append("")
    lines.append(
        f"Контраст Q_θ(q) в режиме конденсации: {np.nanmean(out['contrast']):+.4g} "
        f"(внутри {np.nanmean(out['contrast_inside']):+.4g}, "
        f"вне {np.nanmean(out['contrast_outside']):+.4g}); "
        f"режим = {np.nanmean(out['mask_fraction']):.1%} точек"
    )
    lines.append(
        f"Среднее Q_θ(q) = {np.nanmean(out['q_mean']):+.4g}   "
        f"Среднее Q_θ(t) = {np.nanmean(out['t_mean']):+.4g}   «нетто-источник?»"
    )
    if out["cosine_correction_vs_dphys"].size:
        lines.append(
            f"Косинус(поправка корректора, δ_phys) = "
            f"{np.nanmean(out['cosine_correction_vs_dphys']):+.3f}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)


def main() -> None:
    """CLI: чекпоинт + memmap → npz/txt/json со статистикой связи Q_θ с ядром и гео."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memmap", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args()

    out = probe(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    text = report(out)
    print(text)  # noqa: T201
    out_path.with_suffix(".txt").write_text(text)
    summary = {
        key: cl.summarize(out[key])
        for key in ("r_q_cond", "r_q_omega", "ev_geo_q", "ev_kernel_q", "ev_geo_t", "ev_kernel_t")
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[closure] записано {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
