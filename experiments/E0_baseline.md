# E0 — Baseline (контроль): pure-physics без стабилизаторов

## Гипотеза / корень

Контрольный прогон. Гипотеза: с корректной физикой (post-БАГ-1) FD-4 +
Forward Euler + spherical Coriolis на real ERA5 при dt=300 s **взрывается
к h≈1** (frac_ic_blown_up→1), потому что нет ни одного диссипативного члена,
Euler безусловно неустойчив для адвекции/гравволн, а полюсная сходимость
сетки рвёт CFL. Это baseline, относительно которого мерится Δ от E1..E5.

## Изменение кода

Нет. Используется текущий `tools/check_physics_new_kernel.py` без новых флагов.

```text
--stencil fd4 --time-scheme euler --coriolis spherical \
--boundary-x periodic --boundary-y replicate --boundary-z replicate
```

## Sanity

n/a (контроль; sanity-инварианты вводятся с E1).

## Команды

```bash
# локальный synthetic-smoke 6h
.venv/bin/python Models/dev/make_synthetic_era5.py /tmp/syn_era5.dat
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme euler --coriolis spherical

# кластер 48h, 12 IC (через ablation runner, label E0)
bash sh_files/check_physics_ablation.sh
```

## Comet

- method_name: `fd4_euler_spherical`
- url / key: _(заполнить после кластерного прогона)_
- tags: `abl_E0`, `new_kernel`, `stencil_fd4`, `time_euler`, `coriolis_spherical`

Историческая ссылка: кластерный job 3998511 (2026-05-15) уже показал post-fix
взрыв всех pure-physics методов на h=1 — см. память
`project_physics_frozen_z_artefact.md`. E0 переподтверждает это как именованный
контроль абляции.

## Результаты

### Локальный synthetic-smoke 6h ✅ (2026-05-15)

`Models/dev/smoke_ablation.py --stencil fd4 --time-scheme euler --coriolis
spherical --abl-label E0 --horizon-hours 6` (synthetic, 2 IC):

```text
[init] method_name=fd4_euler_spherical_E0
  h=  1  |u|max=nan  wrmse(u@500)=nan
  h=  6  |u|max=nan  wrmse(u@500)=nan
nan_count/{t,u,v,z} : (0.0, 26624.0)   # 0 NaN @h0 → 13·32·64=26624 (полный NaN) @h≥1
```

Итог: взрыв на h=1, полный NaN всех уровней (обе IC). Контроль подтверждён —
pure-physics FD-4 + Forward Euler нестабилен с первого часа. На синтетике
(нет hydrostatic balance) взрыв ещё резче, чем на real ERA5 — согласуется с
кластерным job 3998511. Offline-артефакт:
`logs/comet_offline/fd4_euler_spherical_E0/*.zip`.

### Кластер 48h (12 IC, real ERA5)

| метрика | h=6 | h=24 | h=48 |
| --- | --- | --- | --- |
| frac_ic_blown_up | | | |
| weighted_rmse/z/500hPa | | | |
| weighted_rmse/t/850hPa | | | |
| weighted_rmse/u/500hPa | | | |
| weighted_rmse/v/500hPa | | | |
| weighted_rmse/q/700hPa | | | |
| acc/z/500hPa | | | |

Якоря (один раз, переиспользуются во всех Ek):

| якорь | frac@48 | wrmse/z500@48 | acc/z500@48 |
| --- | --- | --- | --- |
| persistence | | | |
| fixC | | | |

## Вердикт

_(заполнить: подтверждён ли взрыв@h≈1; на каком h frac>0.5)_

## Решение / next

→ E1 (SSP-RK3): атакуем Euler-неустойчивость первой.
