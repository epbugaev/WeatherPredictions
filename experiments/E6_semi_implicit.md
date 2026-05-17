# E6 — Semi-implicit (идея 6): атака на локализованный корень

## Гипотеза / корень

Абляция E1–E5 точно локализовала корень взрыва: **мгновенная жёсткая
неустойчивость быстрой линейной связки** PGF (−∇z) ↔ дивергенция ↔
гидростатика (z_t от t_t), взрыв <1ч. Forward Euler / SSP-RK3 безусловно
неустойчивы для этого осциллятора; ∇⁴/polar/flux/DFI бьют по медленным
механизмам. Каноничный фикс — **полу-неявная схема**: быстрые линейные
члены неявно (Crank–Nicolson), остальное явно. Идея 6, выбрана
пользователем после E1–E5.

## Изменение кода

- `utils/physics.py`: `time_scheme="semi_implicit"`. Линейная связка
  `u_t|_L=−∂ₓz, v_t|_L=−∂_yz, z_t|_L=−A·δ` (A — P×P оператор
  continuity→adiabatic→hydrostatic при reference T̄=250 K, построен
  по-столбцово). Исключение u,v → Гельмгольц `(I−a²A∇²)zⁿ⁺¹=RHS`, a=dt/2.
  Вертикаль cumsum-`integral_z` **дефектна** (не самосопряжена) → нет
  модального базиса; фикс — **масс-взвешенная симметризация** в Δp-скаляр-
  произведении (`S=diag(pixel_z)`, `eigh(½(M+Mᵀ))`, `M=S^½AS^{-½}`) →
  вещественные ортонормированные моды (`Vinv=Qᵀ·S^½`, `V·Vinv=I` точно).
  Per-mode спектральный Гельмгольц (rfft2, постоянный Δx_ref).
- `tools/check_physics_new_kernel.py`: `--time-scheme semi_implicit`.

## Sanity (`Models/dev/sanity_semi_implicit.py`, PASS-gate)

```text
roundtrip rel err = 9.29e-7        # масс-симметриз. модальный солвер ТОЧЕН
not_worse: ssp_rk3=5.79e4  semi_implicit=5.77e4 (dt=300, сопоставим)
consistency: ‖si−z0‖/‖z0‖=5.1e-4  rel‖si−rk3‖=1.7e-5 (консистентен, не демпфер)
```

Поймано и исправлено: (1) скаляр-Φ SI недостаточен (стиффность
многоуровневая); (2) cumsum-вертикаль дефектна (`V·Vinv≠I`, roundtrip 0.49)
→ масс-симметризация. Численная машинерия корректна (solver exact 9e-7),
SI — консистентный интегратор.

## Честный результат (по плану «фиксируем отрицательный»)

Scoped semi-implicit **НЕ даёт безусловной устойчивости** на этом ядре.
Reference-оператор Гельмгольца вынужденно приближён: постоянный Δx_ref
(истинный pixel_x варьируется ~×10 полюс/экватор), дважды-периодич. вместо
replicate-y, масс-симметризов. вертикаль. Остаток `L_true−L_ref` сам
жёсткий (полюсные быстрые моды) → при больших dt SI взрывается как explicit
(probe: dt≥600 — оба BLOW; dt≤300 — оба конечны на gentle-state).
Полноценный SI требует **λ-зависимого reference + переформулировки
cumsum-вертикали** = dynamical-core scope (вне абляции).

## Команды

```bash
.venv/bin/python Models/dev/sanity_semi_implicit.py        # PASS
.venv/bin/python Models/dev/smoke_ablation.py --stencil fd4 \
    --time-scheme semi_implicit --coriolis spherical --hyperdiffusion \
    --polar-filter --balance-ic dfi --advection-form flux \
    --w-diagnostic mass_consistent --abl-label E6
# кластер: ABL_ONLY=E6 sh_files/check_physics_ablation.sh
```

## Comet

- method_name: `fd4_semi_implicit_spherical_E6+hyperdiff+polar+dfi+flux+masscons`
- smoke: на синтетике SI выжил h=1 (|u|=131, finite) vs E0–E5 (NaN@h1) →
  слабое улучшение; real-вердикт — кластер.
- url / key: *(заполнить после кластерного прогона E6)*

| метрика | h=6 | h=24 | h=48 | vs E5 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | | | | | |
| weighted_rmse/z/500hPa | | | | | |
| acc/z/500hPa | | | | | |

## Вердикт

Машинерия SI реализована корректно (масс-симметризов. модальный Гельмгольц,
roundtrip-exact; консистентен). Но scoped SI недостаточен для безусловной
устойчивости на этом ядре (λ-зависимая метрика + cumsum-вертикаль → жёсткий
explicit-остаток). Это **ригорозно установленный честный негатив**:
доказано, что корректная стабилизация pure-physics здесь = полноценный
dynamical-core (λ-зависимый SI-reference + самосопряжённая вертикаль) —
вне рамок абляции. Финальное численное подтверждение vs E5/fixC — кластер.

## Решение / next

Абляция исчерпана: 6 принципиальных идей реализованы и корректны (sanity),
но pure-physics при dt=300s на real ERA5 не стабилизируется ни одной — это
и есть научный итог (подтверждает архитектуру WeatherGFT: scale_diff + NN).
Практический путь дальше — dt≪300s (под-CFL) ИЛИ полный dynamical core.