# Эксперимент 22 — перенос эффекта физики на новые регионы × три архитектуры

**Статус:** дизайн (2026-07-15), до запуска. Ветка: `fix_inline_v2`.

## 1. Цель и гипотезы

exp16/20/21 измерили эффект физического приора (исправленной A2 и легаси) на **одном**
регионе (USA-кроп) для трёх носителей. exp22 проверяет **перенос** этих выводов на два
новых региона с иным динамическим режимом:

- **H1 (перенос знака).** Корректная физика (A2 exp13) на новых регионах не хуже
  «без физики», легаси-физика вредит — как на USA.
- **H2 (регион-зависимость).** Величина эффекта зависит от режима: во влажном/
  конвективном или штормовом регионе диабатический Q_θ несёт больше, чем на USA.
- **H3 (архитектура × регион).** Порядок носителей по чувствительности к физике
  (IAM4VP/PredRNNv2 авторегрессивные vs SimVPv2 MIMO — см. exp21 §8.5) сохраняется
  на новых регионах.

Сравнение — строго на **общей эпохе** (exp16 §11.3): все армы обучаются фиксированно
150 эпох, ранжируются по `last.pt`, без early-stopping-разброса эпох.

## 2. Матрица прогонов (18)

3 семейства × 3 арма × 2 региона. Сид 0, горизонт t=12, train 2000–2003 / val 2004.

**Бюджет эпох — пер-семейный** (правка 2026-07-15): SimVPv2 и IAM4VP — **150**;
PredRNNv2 — **40** (как exp20). Причина: аллокация кластера proj_1715 имеет
фиксированный дедлайн (~27 ч), трейнер не поддерживает ресюм, а PredRNNv2 при ~16
мин/эпоху физ-арма даёт ≈40 ч на 150 эпох — не влезает. Внутрисемейное сравнение
честно (3 арма семейства на общей эпохе); кросс-семейное — по **относительной** Δ к
`no_physics`, поэтому разные общие эпохи между семействами допустимы.

**Гейт оценки:** метрики считать ТОЛЬКО когда все 3 арма лестницы дошли до своей
общей эпохи (150 или 40). Ранний прогон на недоученных физ-армах даёт мусор
(доученный контроль против стартовавшей физики) — проверено на SimVPv2 npac.

| семейство | арм `no_physics` | арм `A2 exp13` | арм `legacy` |
|---|---|---|---|
| PI-IAM4VP | ✓×2 региона | ✓×2 | ✓×2 |
| PI-PredRNNv2 | ✓×2 | ✓×2 | ✓×2 |
| PI-SimVPv2 | ✓×2 | ✓×2 | ✓×2 |

Имя рана: `exp22-<fam>-<arm>-<region>-s0`, напр. `exp22-simvpv2-a2-npac-s0`.

## 3. Регионы

Оба кропа **обязаны быть 32×64** — physics-residual захардкожен на латент 8×16
(`utils/physics_residual.py:826-831`); иначе физ-армы падают fail-fast. Сетка
WeatherBench 128×256, 1.40625°, ряды **юг→север** (индекс 0 = −90N), долгота 0–360°E.

| регион | ключ | `data.cut` | охват | `physics_lat_start_deg` | `diabatic_cut` |
|---|---|---|---|---|---|
| Европа/Франция | `france` | `[[81,113],[0,64]]` | 23.9–68.9N, 0–90E | `26.71875` | `[81,113,0,64]` |
| Сев. Тихий океан | `npac` | `[[74,106],[96,160]]` | 14.1–59.1N, 135E–135W | `16.875` | `[74,106,96,160]` |

`physics_dlat_deg = physics_dlon_deg = 5.625` (латент = данные ÷4). Формула широты
первого латентного ряда: `physics_lat_start_deg = −90 + (lat0 + 2)·1.40625`.

**Оговорка по Франции.** 90°-широкое окно у нулевого меридиана без долготного
переноса ставит Францию (2E) у западного края (к востоку — Европа, Зап. Азия).
Атлантический апстрим Франции остаётся за окном — методологически как у USA-кропа
(регион, не полный домен). Долготный roll отклонён: он несовместим с `diabatic_cut`,
который режет **глобальные** константы простым срезом (свёрнутые данные разъехались
бы с несвёрнутыми константами).

## 4. Армы — точные поля физики

Все три арма делят один `PhysicsResidualMixin`; отличаются только `physics_*`.
Регион-независимые поля — ниже; регион-зависимые (`data.cut`, `physics_lat_start_deg`,
`diabatic_cut`) берутся из §3.

**`no_physics`** (контроль): `physics_feature_mode: no_physics` — та же ёмкость
головы, нулевая физфича. Остальные `physics_*` не влияют.

**`A2 exp13`** (исправленная физика + диабатика):
```yaml
use_physics_residual_corrector: true
physics_feature_mode: tendency
physics_residual_hybrid_mode: stable_physical_v2
physics_residual_input_space: physical
physics_residual_humidity_mode: relative_to_specific
physics_residual_tendency_clip: 8.0
physics_residual_hybrid_steps: 3
physics_w_diagnostic: mass_consistent
use_diabatic_term: true
diabatic_hidden_channels: 64
diabatic_lambda_l1: 0.0001
diabatic_apply_to: t_and_q
diabatic_constants_path: /home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc
```

**`legacy`** (оригинал WeatherGFT, с багами):
```yaml
use_physics_residual_corrector: true
physics_feature_mode: tendency
physics_residual_hybrid_mode: legacy_normalized
physics_residual_input_space: normalized
physics_residual_humidity_mode: as_is
physics_residual_tendency_clip: 0
use_diabatic_term: false
```

Диабатические константы (`constants_1.40625deg.nc`) — глобальные, world-readable,
переиспользуются всеми регионами через `diabatic_cut`. Отдельную климатологию Q
считать НЕ нужно.

## 5. Подготовка данных

Источник: `/home/fratnikov/weather_bench/1.40625deg/<var>/<var>_<year>_1.40625deg.nc`
(14 переменных, 2000–2004, world-readable).

1. **Репак мемапов** (`tools/repack_era5.py`, CPU, `sh_files/repack_era5.sh`), на регион:
   ```
   REPACK_CUT="81 113 0 64"   PREFIX=predformer_france_2000_2004  → предформ. .dat (~25 ГБ)
   REPACK_CUT="74 106 96 160" PREFIX=predformer_npac_2000_2004    → .dat (~25 ГБ)
   ```
   Годы 2000–2004; train читает 2000–2003, val 2004 (сплит в датасете). Кроп мемапа
   валидируется против `data.cut` конфига fail-fast (`weatherbench_128_v3.py:413-432`).
2. **Климатология ACC** (`docs/experiments/16_model_ablation_ladder/metrics/climatology.py`):
   `climatology_{france,npac}_2000_2003.npz`.
3. **Пороги CSI/FSS** (`.../metrics/thresholds.py`): `thresholds_{france,npac}_2004.npz`
   (квантили ИСТИНЫ на валидации региона).

## 6. Генерация конфигов

18 конфигов генерируются скриптом `docs/experiments/22_cross_region_physics/make_configs.py`
из трёх арм-шаблонов на семейство (существующие USA-конфиги) подстановкой РОВНО
региональных полей (`data.cut`, `physics_lat_start_deg`, `diabatic_cut`, `experiment.name`,
`max_epoch: 150`, отключённый early-stopping). Никакого ручного копипаста — DRY.
Раскладка: `configs/exp22/<fam>_<arm>_<region>.yaml`.

Общий бюджет: `max_epoch: 150`, `early_stopping_patience` снят (гарантия общей эпохи).

## 7. Запуск — поэтапно (одобрен подход A)

Общий лончер `sh_files/train_v4_memmap.sh <config>` + `ORIG_MEMMAP=<региональный .dat>`,
`-A proj_1715`, `--constraint=type_e|type_f|type_h`, волнами.

- **Фаза 1 — валидация пайплайна на SimVPv2** (быстрейший, MIMO): 6 прогонов
  (3 арма × 2 региона). Прогнать данные→train→метрики целиком, убедиться, что
  новорегиональный путь цел, ДО тяжёлого компьюта.
- **Фаза 2 — тяжёлые носители:** PI-IAM4VP + PI-PredRNNv2 (12 прогонов) волнами.

## 8. Оценка

Переиспользуется кросс-архитектурный харнесс (метрики + rollout), собранный в
exp20/21: `tools/metrics_ladder.py`, `tools/rollout_ladder.py`, ядро метрик exp16.
На регион: форест (Δ к `no_physics`), levels×шаг, PSD, rollout ratio/levels —
форматы как exp21. Плюс кросс-регион-сводка: переносится ли знак и растёт ли Q_θ.

## 9. Бюджет компьюта и риски

- **18 × 150 эпох** — недели GPU. PredRNNv2 тяжелейший (batch32=OOM; long-волна не
  влезает в 2-суточный лимит одной джобой). Заложить **автоматический ресюм с
  чекпоинта** (как exp20) для тяжёлых семейств.
- Поэтапность (§7) дерискует: баг нового региона всплывёт на дешёвом SimVPv2.
- GPU-квота proj_1715 делится с другими джобами — реалистично считать волнами.

## 10. Артефакты

`configs/exp22/` (18) · `sh_files/exp22_*.sh` (репак, train-волны, метрики, rollout) ·
`make_configs.py` · региональные мемапы/климатологии/пороги (на кластере) ·
`docs/experiments/22_cross_region_physics/` (этот README + метрики/rollout по регионам,
кросс-регион-сводка) · запись в `CHANGELOG.md`.
