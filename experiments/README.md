# Абляция: стабилизация pure-physics ядра

Forward-абляция `utils.physics.PurePDEKernel`: от baseline (взрыв на h=1)
**постепенно** добавляем по одному принципиальному стабилизатору. Каждый Ek —
отдельный флаг, отдельный Comet-эксперимент, отдельный sanity-тест,
кумулятивный набор флагов E1..Ek.

Контекст и корни взрыва — см. план `~/.claude/plans/radiant-booping-charm.md`
и память `project_physics_frozen_z_artefact.md`. Кратко: после фикса БАГ-1
(z_t был ×100 заморожен) ни один pure-physics интегратор не доживает дальше
h=1 на real ERA5 при dt=300 s (полюсная CFL + нет диссипации + Euler-
неустойчивость + IC-shock + неконсервативность). `scale_diff` маскирует это
клиппингом. Цель — честная замена `scale_diff`.

## Матрица абляции

| Эксп | Идея | Корень | Флаг (новый) | Атакует |
| --- | --- | --- | --- | --- |
| **E0** | — | контроль | (нет) | воспроизвести post-fix взрыв |
| **E1** | 3 | время | `--time-scheme ssp_rk3` | Euler-неустойчивость |
| **E2** | 2 | алиасинг | `--hyperdiffusion --hyperdiff-tau-hours` | Phillips / 2Δx-каскад |
| **E3** | 1 | полюс | `--polar-filter --polar-filter-lat-deg` | CFL у полюса |
| **E4** | 4 | shock | `--balance-ic {none,dfi,geostrophic}` | imbalance IC |
| **E5** | 5 | сохранение | `--advection-form flux --w-diagnostic mass_consistent` | неконсервативность |

E_k = накопленный набор E1..E_k. После E5 — опциональная фаза B
(leave-one-out из полного стека для атрибуции вклада каждого).

## Якоря сравнения (фиксируются один раз)

- **persistence** — тривиальный пол (незаэволюционировавший IC).
- **fixC** — production `scale_diff`+`.detach()`
  (`tools/check_physics_fix.py --fix-mode C`); единственный, кто сейчас
  доживает до h=48 (acc/z 0.97→0.24). Верхняя планка «текущий хак».

Оценка каждого Ek: max устойчивый горизонт (первый h с
`frac_ic_blown_up`>0.5), `weighted_rmse` и `acc` для
z500/t850/u500/v500/q700 на h∈{6,24,48}, `frac_ic_blown_up`@{6,24,48} —
vs E(k−1), persistence, fixC.

## Как запускать

```bash
# 1. lint
ruff check utils/ tools/ Models/dev/ && ruff format --check utils/ tools/ Models/dev/

# 2. ЖЁСТКИЙ GATE: sanity текущей идеи (CPU)
.venv/bin/python Models/dev/sanity_<idea>.py        # PASS, иначе СТОП

# 3. локальный synthetic-smoke 6h
.venv/bin/python Models/dev/make_synthetic_era5.py /tmp/syn_era5.dat
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --horizon-hours 6 <флаги Ek>

# 4. кластер 48h, 12 IC, online Comet
bash sh_files/check_physics_ablation.sh        # или remote_submit.sh
```

## ⚠ КОРРЕКЦИЯ: прежний «all-fail@h1» был ЧАСТЬЮ BUG-АРТЕФАКТОМ

Аудит ablation-кода (после вопроса пользователя про «логические баги»)
нашёл **3 blocking-бага в самих стабилизаторах** — т.е. идентичный
взрыв@h1 у E0–E5 в job 3998911 был НЕ чистой физикой, а во многом
следствием того, что стабилизаторы были сломаны (ровно урок
[[project_physics_frozen_z_artefact]]: не выдавать артефакт за результат):

- **B1** (∇⁴-гипердиффузия): единый K4 на самой крупной ячейке + ЯВНЫЙ
  −K4·∇⁴. Т.к. собств.знач. ∇⁴ ∝ 1/Δx⁴, на полюсе явный множитель ≈ **−172**
  → гипердиффузия **усиливала** 2Δx ×172/шаг (E2–E5 были АКТИВНО ХУЖЕ E0).
- **B2** (semi-implicit): `lam.clamp_min(0)` зануляло БЫСТРЕЙШИЕ
  гравитационные моды → `_si_solve` ≈ identity (E6 SI был **no-op**).
- **B3** (DFI): span=6ч → 145 forward-шагов на нестабильном ядре → всегда
  NaN → silent fallback → **E4 ≡ E3** байт-в-байт.
- **B4** (CN ∇²-несогласованность 52%); + проверено КОРРЕКТНЫМ: БАГ-1 фикс,
  единицы, проводка, знаки flux/polar/geo, GeometryCPU↔Grid.

**Исправлено:** B1→безусловно-устойчивый НЕЯВНЫЙ зональный спектр.
фильтр; B2→`|λ|` (все моды); B4→зонально-неявный λ-зависимый Δx, один
∂²ₓ по обе стороны; B3→span 1ч. Все 6 sanity-гейтов переписаны/усилены
(теперь ловят B1/B2) и PASS.

## Статус

| Ek | sanity (после фикса) | job 3998911 (BUG-masked) | локальный re-run (fixed, гладкий IC) |
| --- | --- | --- | --- |
| E0 euler | n/a | h1=1.0 | взрыв @substep 26 (2.2ч) |
| E1 ssp_rk3 | ✅ order 3.01 | h1=1.0 | @17 (1.4ч) |
| E2 +hyperdiff | ✅ не усиливает (≤1), e-fold≈τ | h1=1.0 (B1!) | @18 |
| E3 +polar | ✅ m=0 точно | h1=1.0 | @24 (\|u\|@surv 3e2 ≪ E0 3e6) |
| E4 +dfi | ✅ geo точен | h1=1.0 (B3: ≡E3) | @3 (DFI вредит) |
| E5 +flux+mc | ✅ flux→1e-9 | h1=1.0 | @23 |
| E6 semi_impl | ✅ implicit O(1)=2.4, stab. лин.grav@dt300 | n/a | @13 |
| fixC (якорь) | — | h48=0.0 (не NaN, но acc/z 0.44→0.24) | — |

(sanity = код-корректность доказана и B1/B2 пойманы; локальный re-run —
гладкий сбалансированный synthetic, `Models/dev/rerun_ablation_local.py`;
кластерный re-run исправленного кода — фоновый watcher ждёт VPN.)

## ГЛАВНЫЙ ВЫВОД (уточнённый, без bug-маскировки)

1. **Прежний «6 идей тождественно проваливаются@h1» — некорректен:** он
   был во многом артефактом B1–B3. После фикса методы **РАЗЛИЧАЮТСЯ**
   (локально blow@substep 3..26, не идентично), стабилизаторы дают разный
   эффект (E3-polar держит \|u\| на 4 порядка меньше до взрыва; E6-SI
   доказанно стабилизирует ЛИНЕЙНУЮ грав-волну при dt=300, где ssp_rk3→inf).
2. **Но pure-physics при dt=300s всё равно НЕ доживает** (даже на гладком
   сбалансированном IC все взрываются за 0.25–2.2ч; на real ERA5 ожидается
   аналогично — кластер-подтверждение pending VPN). Этот вывод теперь
   **строгий, не bug-masked**.
3. Корень — мгновенная жёсткая стиффность связки PGF↔div↔гидростатика
   (≤1–2ч), не медленный алиасинг/полюс. Зонально-неявный SI лечит
   ЗОНАЛЬНУЮ часть, но МЕРИДИОНАЛЬНАЯ гравиволна остаётся явной
   (forward-Euler-нестабильна) → полная устойчивость требует **2D-неявного
   Гельмгольца (блок-трёхдиаг.) = полноценное dynamical-core ядро**
   (вне рамок абляции — установленная архитектурная граница, не баг).
4. `fixC` (scale_diff+detach) — единственный численно-конечный, но
   физически расходится → подтверждает: WeatherGFT держится на
   scale_diff + обучаемой NN-коррекции, не на чистой физике.

**Прочее:**

- Sanity-гейты по ходу поймали РЕАЛЬНЫЕ ошибки кода: ∇²=`d_x∘d_x`
  (нулевой Найквист), flux-формула плана (≡ advective), cumsum-вертикаль
  дефектна (масс-симметризация), и финальный аудит — B1–B4. Гейты
  переписаны так, чтобы ловить B1 (контракция ≤1 при любых τ/dt) и B2
  (implicit-strength O(1), не identity).
- Фаза B (leave-one-out) теперь осмысленна (методы различаются) — провести
  ПОСЛЕ кластерного re-run исправленного кода.
- surface-метрики `weighted_rmse/surface/*` → `persistence/surface/*`
  (физика их не прогнозирует → тождественны у всех; не «баг-дубликат»).

**Что нужно для безусловной устойчивости (dynamical-core scope, не баг):**
2D-неявный Гельмгольц (rfft-x + блок-трёхдиаг. по широте, чтобы и
МЕРИДИОНАЛЬНАЯ гравиволна была неявной) + самосопряжённая (не cumsum)
вертикаль; и/или dt ≪ 300s (под-CFL гравволн). Зонально-неявный E6 —
корректный, но частичный шаг к этому.

## Comet-эксперименты

| method | Comet experiment id |
| --- | --- |
| fd4_euler_spherical_E0 | 3908eb3886c94d618338de7826125429 |
| fd4_ssp_rk3_spherical_E1 | …/23ca9c9807ee473e86687329a00449b4 |
| fd4_ssp_rk3_spherical_E2+hyperdiff | …/9f9f4eb7bc4740ea9a4961976c62480d |
| fd4_ssp_rk3_spherical_E3+hyperdiff+polar | …/780e1ec3282744bca5a5d7b26465b1c8 |
| fd4_ssp_rk3_spherical_E4+hyperdiff+polar+dfi | …/0a3835ec145241a4a2b143b4aa1fd9f0 |
| fd4_ssp_rk3_spherical_E5+…+flux+masscons | …/661fe147cdc24f18b77408527683fda8 |
| fd4_semi_implicit_spherical_E6+… | ⏳ кластерный прогон ждёт VPN (ABL_ONLY=E6) |
| fd4_euler_spherical_fixC_scaleDiffDetach (якорь) | …/63b8407ddb89442091c706ea0a69f63d |

Свежие данные: `.venv/bin/python Models/dev/fetch_ablation_summary.py`.
