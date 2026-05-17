# E5 — Консервативная форма (идея 5): структурное сохранение

## Гипотеза / корень

Адвективная форма `u·∇X` на коллокированной сетке с центральным FD
неконсервативна и структурно поддерживает алиасинг; диагностический
`w = ∫−(u_x+v_y)` копит вертикальный дрейф дивергенции. Flux-форма +
mass-consistent w делают оператор структурно устойчивым (сохранение
1-го/2-го моментов) — единственная идея, чинящая корень, который E2
(диффузия) лишь маскирует. Гипотеза: поверх полного стека даёт лучшую
долгосрочную (h=48) консервацию энергии/энстрофии и скилл.

## Изменение кода

- `utils/physics.py`: `advection_form: Literal["advective","flux"]
  ="advective"` (flux: `∂(uX)/∂x+∂(vX)/∂y − X·(u_x+v_y)`);
  `w_diagnostic: Literal["plain","mass_consistent"]="plain"` (вычесть
  колоночное среднее дивергенции → ∫div dp≈0). Arakawa-якобиан —
  stretch-подопция, не блокер.
- `tools/check_physics_new_kernel.py`: `--advection-form`,
  `--w-diagnostic`; method_name/tags.

## Sanity (`Models/dev/sanity_conservation.py`, PASS-gate)

- при нулевом форсинге и периодичном x: flux-форма держит ∑X и ∑X²
  (относительный дрейф < 1e-3 за N=100 шагов) — много лучше advective
  (которая дрейфует > 1e-1).
- mass-consistent w: `∫(u_x+v_y) dp ≈ 0` на каждый столб (< 1e-6 от
  типичной |div|).

## Команды

```bash
.venv/bin/python Models/dev/sanity_conservation.py        # PASS
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme ssp_rk3 --coriolis spherical \
    --hyperdiffusion --polar-filter --balance-ic dfi \
    --advection-form flux --w-diagnostic mass_consistent
# кластер: ablation runner label E5 (полный стек)
```

## Comet

- method_name: `fd4_ssp_rk3_spherical_E5+hyperdiff+polar+dfi+flux+masscons`
- url / key / tags: *(заполнить после кластерного sweep)*

## Результаты

### sanity ✅ (2026-05-15)

`Models/dev/sanity_conservation.py` PASS:

```text
|∑t|/∑|t|:  advective=1.214e-02   flux=1.244e-09     # flux телескопирует к 0
plain ∫div·Δp(rel)=3.05e-01   mass_consistent=2.58e-08
```

- **flux**: ∑ tendency = −∑[∂ₓ(uX)+∂_y(vX)] ≡ 0 до round-off (1.2e-9) при
  любом X,u,v → ∑X (масса) сохраняется КАЖДЫЙ шаг; advective не
  телескопирует (1.2e-2) и дрейфует при ∇·V≠0.
- **mass_consistent w**: колоночный ∫div·Δp обнулён (0.30→2.6e-8).

Поймана концептуальная ошибка плана: исходная формула
`∂ₓ(uX)+∂_y(vX)−X·div` алгебраически **тождественна** advective (нет
выигрыша в сохранении). Реализована истинная дивергентная форма
`−∇·(VX)` (стандартная flux-форма динамического ядра) — она и сохраняет
∑X точно.

### smoke 6h ✅ pipeline (2026-05-15)

`smoke_ablation.py` полный стек → отдельный Comet-эксперимент
`fd4_ssp_rk3_spherical_E5+hyperdiff+polar+dfi+flux+masscons`; все 5
стабилизаторов в одном пути, без исключений. Синтетика-взрыв@h=1 (как E0–E4,
worst-case данные — не физ-вердикт). Регрессия: все 5 sanity-гейтов PASS.

### кластер 48h ✅ (job 3998911, real ERA5, 12 IC)

Полный стек (ssp_rk3+∇⁴+polar+dfi+flux+masscons): `frac_ic_blown_up`
h0=0.0 → **h1=1.0** → h48=1.0; prognostic = **NaN с h1**. **Тождественно
E0** — даже полный кумулятивный стек не сдвинул взрыв ни на час. Comet:
`fd4_ssp_rk3_spherical_E5+hyperdiff+polar+dfi+flux+masscons`.

| метрика | h=1 | h=6 | h=48 | vs E4 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | 1.0 | 1.0 | 1.0 | ≡ | хуже (fixC=0.0) |
| weighted_rmse/z/500hPa | NaN | NaN | NaN | ≡ | fixC 1.1e5→6.9e10 |
| acc/z/500hPa | NaN | NaN | NaN | ≡ | fixC 0.44→0.24 |

## Вердикт

flux/mass-consistent корректны (sanity острый, поймал ошибку формулы
плана). Полный стек E5 ≡ E0 — взрыв@h1. **Фаза B (leave-one-out) не нужна**:
вклад каждого Ek в стабильность = 0 (E5≡E0), атрибутировать нечего.
Честный негатив (README §Главный вывод).

## Решение / next

Полный стек собран. → Шаг 6: кластерный sweep E0..E5+fixC, заполнение
таблиц из Comet, синтез, docs/physics.md + CHANGELOG.
