# E2 — ∇⁴ гипердиффузия (идея 2): анти-алиасинг

## Гипотеза / корень

Нелинейная (Phillips) неустойчивость: энстрофия каскадирует на 2Δx,
алиасит обратно, взрыв. У всех реальных GCM (IFS/GFS/WRF) есть ∇⁴/∇⁶ или
спектральная отсечка. `scale_diff` — неселективный по масштабу клиппинг
амплитуды; ∇⁴ — принципиальная scale-selective замена. Гипотеза: поверх E1
гипердиффузия снимает 2Δx-шум и продлевает устойчивость, мало трогая
крупный масштаб (скилл).

Закрывает open question docs/physics.md #4 (`scale_diff` без обоснования).

## Изменение кода

- `utils/physics.py`: `hyperdiffusion: bool=False`,
  `hyperdiff_tau_hours: float=6.0`; K4 выводится из `grid.pixel_x`
  (e-folding моды 2Δx за τ); член `−K4·lap(lap(X))` в `rhs()` для u,v,t,q,z,
  где `lap = d_x∘d_x + d_y∘d_y` через существующий `self.diff` (DRY).
- `tools/check_physics_new_kernel.py`: `--hyperdiffusion`,
  `--hyperdiff-tau-hours`; method_name/tags.

## Sanity (`Models/dev/sanity_hyperdiffusion.py`, PASS-gate)

- scale-selective: 2Δx-шахматка гаснет за ~τ в ≈1/e; крупная мода (k=1)
  почти цела (>0.9 амплитуды за τ).
- `∇⁴(const)=0` (сохранение домен-среднего, дрейф < 1e-6).

## Команды

```bash
.venv/bin/python Models/dev/sanity_hyperdiffusion.py        # PASS
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme ssp_rk3 --coriolis spherical \
    --hyperdiffusion --hyperdiff-tau-hours 6
# кластер: ablation runner label E2 (накопленно: ssp_rk3 + hyperdiff)
```

## Comet

- method_name: `fd4_ssprk3_spherical+hyperdiff`
- url / key / tags: _(заполнить)_

## Результаты

### sanity ✅ (2026-05-15)

`Models/dev/sanity_hyperdiffusion.py` PASS:

```text
max|∇⁴(const)|=0.000e+00                       # консервативен (масса)
K4=4.408e+17  t_efold(2Δx)=2.171e4 s (τ=21600 s)   # ratio 1.005 ≈ exact
t_efold(k=1) ≫ 50·τ                            # scale-selective
```

Поймана и исправлена реальная ошибка: ∇² нельзя строить как `d_x∘d_x`
(FD-4 первая производная имеет **нулевой** отклик на 2Δx-моде Найквиста →
гипердиффузия не трогала бы именно ту моду). Реализована 3-точечная вторая
разность (periodic-x roll, replicate-y pad), K4 калиброван по дискретному
собственному значению (16/Δx⁴), а не континуальному π⁴. e-folding 2Δx =
2.17e4 s ≈ τ — калибровка точная; крупный масштаб не тронут.

### smoke 6h ✅ pipeline (2026-05-15)

`smoke_ablation.py … --hyperdiffusion --hyperdiff-tau-hours 6 --abl-label E2`:
код-путь отрабатывает, отдельный Comet-эксперимент
`fd4_ssp_rk3_spherical_E2+hyperdiff`. На синтетике (бело-шумный memmap)
всё ещё взрыв@h=1 — синтетика worst-case по 2Δx + не balanced + dt=300s
> адвективный CFL; диффузия (timescale 6h) не успевает. Это **пайплайн-гейт**,
не физ-вердикт: реальный сигнал (гладкий ERA5, где алиасинг — настоящий
механизм отказа) — на кластере.

### кластер 48h ✅ (job 3998911, real ERA5, 12 IC)

`frac_ic_blown_up`: h0=0.0 → **h1=1.0** → h48=1.0. Все `weighted_rmse/*`,
`acc/*` (prognostic) = **NaN с h1** (`nan_count/u`=26624). **Тождественно
E0/E1** — ∇⁴ не сдвинул взрыв ни на час. Comet:
`fd4_ssp_rk3_spherical_E2+hyperdiff` (см. README §Comet).

| метрика | h=1 | h=6 | h=48 | vs E1 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | 1.0 | 1.0 | 1.0 | ≡ (оба 1.0) | хуже (fixC=0.0) |
| weighted_rmse/z/500hPa | NaN | NaN | NaN | ≡ | fixC=1.1e5→6.9e10 |
| acc/z/500hPa | NaN | NaN | NaN | ≡ | fixC 0.44→0.24 |

## Вердикт

Оператор и калибровка корректны (sanity острый, поймал баг d_x∘d_x), но
∇⁴ (τ=6ч) **не стабилизирует** на real ERA5 — взрыв@h1 ≡ E0. Диффузия
работает на медленном таймскейле, взрыв же мгновенный (стиффность старта).
Честный негатив (см. README §Главный вывод).

## Решение / next

hyperdiff остаётся в накопленном стеке (E3..E5 поверх). → E3 (polar filter).
