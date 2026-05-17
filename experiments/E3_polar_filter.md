# E3 — Полярный Фурье-фильтр (идея 1): полюсная CFL

## Гипотеза / корень

Регулярная lat-lon: верхняя широта 32-точечной сетки ≈ 84.8°,
`cos φ ≈ 0.09` → `Δx` сжимается ~10×, локальный CFL ~10× хуже экватора.
Любой явный шаг, настроенный на экватор, взрывается у полюса первым.
Классический докоспектральный cure: zonal FFT по долготе, обрезать
зональные гармоники m > m_cut(φ) (эффективный Δx делается
широто-независимым). Гипотеза: поверх E1+E2 это снимает полюсный взрыв и
реально доводит pure-physics до h=48.

## Изменение кода

- `utils/physics.py`: `import torch.fft`; `polar_filter: bool=False`,
  `polar_filter_lat_deg: float=60.0`; helper `_apply_polar_filter(field)`:
  rFFT по долготе на каждой широте, обнулить m > m_cut(φ) =
  `max(1, round((W/2)·cosφ/cosφ₀))`, irFFT; применить к u,v,t,q,z перед
  return `step()` (обе ветки).
- `tools/check_physics_new_kernel.py`: `--polar-filter`,
  `--polar-filter-lat-deg`; method_name/tags.

## Sanity (`Models/dev/sanity_polar_filter.py`, PASS-gate)

- m=0 (зональное среднее) сохраняется точно; площадно-взвешенный интеграл
  поля сохраняется в tol.
- высокая m (m=W/4) у полюса (φ=85°) убрана (амплитуда → ~0), та же m у
  экватора (φ=0°) сохранена; без NaN.

## Команды

```bash
.venv/bin/python Models/dev/sanity_polar_filter.py        # PASS
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme ssp_rk3 --coriolis spherical \
    --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --polar-filter-lat-deg 60
# кластер: ablation runner label E3 (накопленно: ssp_rk3 + hyperdiff + polar)
```

## Comet

- method_name: `fd4_ssprk3_spherical+hyperdiff+polar`
- url / key / tags: _(заполнить)_

## Результаты

### sanity ✅ (2026-05-15)

`Models/dev/sanity_polar_filter.py` PASS:

```text
zonal_mean Δ=7.15e-7  global_mean Δ=0.000e+00      # m=0 нетронут (масса)
m=16: amp0=0.707  eq(φ=-2.7°)=0.707  pole(φ=-84.5°)=4.2e-7
```

m=0 (зональное среднее) сохраняется точно — фильтр консервативен. Высокая
мода m=W/4 у полюса убрана (0.707→4e-7), у экватора цела (0.707) — корректная
широто-зависимая отсечка, снимающая полюсную CFL.

### smoke 6h ✅ pipeline (2026-05-15)

`smoke_ablation.py … --polar-filter --polar-filter-lat-deg 60 --abl-label E3`:
отдельный Comet-эксперимент `fd4_ssp_rk3_spherical_E3+hyperdiff+polar`;
`_finalize`-хук корректно оборачивает все 3 ветки (euler/rk4/ssp_rk3) без
регрессий. На бело-шумной синтетике всё ещё взрыв@h=1 (worst-case данные,
не физ-вердикт — см. E1/E2). Реальный сигнал — кластер.

### кластер 48h ✅ (job 3998911, real ERA5, 12 IC)

`frac_ic_blown_up`: h0=0.0 → **h1=1.0** → h48=1.0; prognostic = **NaN с h1**.
**Тождественно E0/E2** — polar filter не сдвинул взрыв. Гипотеза «E3
убирает полюсный взрыв → дожить до h=48» **опровергнута**: взрыв не
полюсной природы (мгновенный, не растущий часами). Comet:
`fd4_ssp_rk3_spherical_E3+hyperdiff+polar`.

| метрика | h=1 | h=6 | h=48 | vs E2 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | 1.0 | 1.0 | 1.0 | ≡ | хуже (fixC=0.0) |
| weighted_rmse/z/500hPa | NaN | NaN | NaN | ≡ | fixC 1.1e5→6.9e10 |
| acc/z/500hPa | NaN | NaN | NaN | ≡ | fixC 0.44→0.24 |

## Вердикт

Фильтр корректен и консервативен (sanity острый), но **не стабилизирует** —
взрыв@h1 ≡ E0. Ключевой вывод: корень НЕ полюсная CFL (иначе E3 помог бы);
это мгновенная стиффность старта. Честный негатив (README §Главный вывод).

## Решение / next

polar остаётся в стеке. → E4 (балансировка IC).
