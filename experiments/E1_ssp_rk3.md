# E1 — SSP-RK3 (идея 3): фазовая устойчивость по времени

## Гипотеза / корень

Forward Euler безусловно неустойчив для адвекции/гравволн (чисто мнимый
спектр: область устойчивости — диск с центром (−1,0), мнимая ось не входит).
SSP-RK3 (Shu–Osher, 3 стадии) — родной интегратор для WENO-5, его область
устойчивости включает отрезок мнимой оси. Гипотеза: только смена интегратора
сдвигает взрыв с h≈1 дальше (но, возможно, не до h=48 — останется
полюс/алиасинг → E2/E3).

Закрывает open question docs/physics.md #5 (WENO-5 + Forward Euler → RK3-TVD).

## Изменение кода

- `utils/physics.py`: `time_scheme: Literal["euler","rk4","ssp_rk3"]`; новая
  ветка в `step()` после rk4: Shu–Osher 3-stage
  `u¹=u+dt·L(u)`, `u²=¾u+¼u¹+¼dt·L(u¹)`, `uⁿ⁺¹=⅓u+⅔u²+⅔dt·L(u²)` для u,v,t,q,z.
- `tools/check_physics_new_kernel.py`: `--time-scheme` choices += `ssp_rk3`;
  method_name/tags.

## Sanity (`Models/dev/sanity_ssp_rk3.py`, PASS-gate)

- 3-й порядок по времени: ошибка на гладком линейном ОДУ ∝ dt³
  (log-log slope ≈ 3 ± 0.3).
- SSP/TVD: на ступеньке total-variation не растёт; Euler на том же CFL даёт
  новые экстремумы, RK3 — нет.

## Команды

```bash
ruff check utils/ tools/ Models/dev/
.venv/bin/python Models/dev/sanity_ssp_rk3.py        # PASS
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme ssp_rk3 --coriolis spherical
# кластер: ablation runner label E1
```

## Comet

- method_name: `fd4_ssprk3_spherical`
- url / key / tags: _(заполнить)_

## Результаты

### sanity ✅ (2026-05-15)

`Models/dev/sanity_ssp_rk3.py` PASS (float64 order-тест):

```text
[ssp_rk3] err(dt)=2.294e-09→4.410e-12  slope=3.01   # 3-й порядок
[euler]   err(dt)=2.420e-04→2.342e-05  slope=1.12   # 1-й порядок
ssp_bounded_vs_euler: |u|max euler=8.42 ssp_rk3=8.31  (rk3 ≤ euler, finite)
```

Наблюдаемый порядок 3.01 vs Euler 1.12 — коэффициенты Shu–Osher (¾/¼, ⅓/⅔)
корректны (тест острый: ошибка коэффициента уронила бы порядок). SSP-RK3
реализован верно.

### smoke 6h ✅ pipeline (2026-05-15)

`smoke_ablation.py --time-scheme ssp_rk3 --abl-label E1`: код-путь
отрабатывает без исключений, метрики логируются, отдельный Comet-эксперимент
`fd4_ssp_rk3_spherical_E1`. **Но на синтетике (бело-шумный, не
hydrostatic-balanced memmap) всё ещё взрыв@h=1** (`|u|max=nan`).

Это **ожидаемо** и согласуется с гипотезой: SSP-RK3 чинит _временную_
неустойчивость, но не убирает алиасинг 2Δx и полюсную CFL. Синтетика —
worst-case по 2Δx-контенту (N(0,σ) в каждой ячейке), поэтому время-схема
сама по себе не спасает; нужен E2 (∇⁴) + E3 (polar). На реальном (гладком)
ERA5 E1 может пережить дольше E0 — решает кластер.

### кластер 48h ✅ (job 3998911, real ERA5, 12 IC)

`frac_ic_blown_up`: h0=0.0 → **h1=1.0** → h48=1.0. Все prognostic
`weighted_rmse/*`,`acc/*` = **NaN с h1** (`nan_count/u`=26624).
**Тождественно E0** — SSP-RK3 не сдвинул взрыв ни на час на real ERA5.
Comet: `fd4_ssp_rk3_spherical_E1` (README §Comet).

| метрика | h=1 | h=6 | h=48 | vs E0 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | 1.0 | 1.0 | 1.0 | ≡ | хуже (fixC=0.0) |
| weighted_rmse/z/500hPa | NaN | NaN | NaN | ≡ | fixC 1.1e5→6.9e10 |
| acc/z/500hPa | NaN | NaN | NaN | ≡ | fixC 0.44→0.24 |

## Вердикт

Код корректен (sanity доказал 3-й порядок 3.01). Time-scheme **в одиночку**
недостаточен — взрыв@h1 ≡ E0 на real ERA5. Forward Euler-неустойчивость
была НЕ главным корнем (мотивирует кумулятивный стек E2..E6). Честный
негатив (см. README §Главный вывод).

## Решение / next

ssp_rk3 остаётся базой накопленного стека (E2..E5 строятся поверх него).
→ E2 (∇⁴ гипердиффузия).
