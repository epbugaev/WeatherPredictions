# exp24 — метрики: хитмапы относительно `no_physics + orography`

Хитмапы «уровень × шаг» вклада каждого арма IAM4VP над контролем
**`no_physics + orography`** (без физики, но с каналом рельефа). Формат и код —
общий харнесс `tools/metrics_ladder.py` (как в exp16/20/21/22): парный бутстрап,
ограниченные метрики (ACC/CSI/FSS) — абсолютной разницей, PSD — по модулю.

## Лестница (baseline = `no_physics+orog`)

| арм | что изолирует над контролем |
| --- | --- |
| `a2-orog` | вклад физики A2-exp13 поверх орографии (тест H2 спека) |
| `nophys-static` | вклад канала суша/море (lsm) поверх орографии |
| `a2-static` | физика A2 + lsm вместе |

## ⚠️ Оговорка: армы на РАЗНЫХ эпохах (last.pt)

Обучение оборвано концом `proj_1715`, поэтому армы сняты на разных эпохах last.pt,
общей эпохи нет:

| арм | эпоха last.pt |
| --- | --- |
| `no_physics+orog` (baseline) | 369 |
| `a2+orog` | 249 |
| `no_physics+static` | 429 |
| `a2+static` | 279 |

По уроку проекта (`val_plateau_not_skill_plateau`) разница в скилле на разбеге
эпох 249↔429 (~10 %+) **больше эффекта статики/физики (2–7 %)**. Поэтому хитмапы
**смешивают** вклад орографии/lsm/физики с недо-обучением: положительный «выигрыш»
недо-обученного арма может быть занижен, отстающего — искажён. Читать как
предварительную картину, не как чистый вклад признака. Чистый замер требует
дообучения всех армов до общей эпохи (500) — не сделано (кончился аккаунт).
Эпоха каждого арма вынесена в его подпись на фигурах. Форест (`fig_ci_forest`)
намеренно не строится — его подпись утверждает «общую эпоху», которой здесь нет.

## Как пересобрать

```bash
# 1) eval на кластере (4 арма last.pt, USA) + парные дельты к nophys-orog
sbatch -A proj_1717 sh_files/exp24_metrics_eval.sh

# 2) скачать npz в results/iam4vp_usa/, переименовать paired_deltas
mkdir -p results/iam4vp_usa
scp 'cluster:~/exp24_metrics/metrics_exp24-iam4vp-*-usa-s0.npz' results/iam4vp_usa/
scp cluster:~/exp24_metrics/paired_deltas_iam4vp_usa.npz results/iam4vp_usa/paired_deltas.npz

# 3) фигуры (форест НЕ строится, см. выше)
python docs/experiments/24_static_inputs/metrics/make_figures.py
```

Выход: `results/iam4vp_usa/fig_levels_<метрика>.png` (сводные хитмапы, строка =
арм) + `results/iam4vp_usa/heatmaps/<метрика>/<арм>.png` (пер-армовые) +
`fig_psd.png` + `metrics_table.md`.
