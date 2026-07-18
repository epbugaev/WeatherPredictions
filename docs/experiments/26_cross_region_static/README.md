# Эксперимент 26 — статические входы вне USA (France, North Pacific)

Кросс-региональная реплика [эксперимента 24](../24_static_inputs/): те же
статические входы (орография / орография+lsm), те же армы (no_physics, A2-exp13),
те же три семейства (PI-IAM4VP, PI-SimVPv2, PI-PredRNNv2) — но на двух новых
регионах. Отношение к exp24 такое же, как у exp22 к exp16/20/21.

## Вопрос

Держатся ли выводы exp24 вне USA?

- **H1.** Орография — достаточный статический сигнал и на France/NPac (маска
  суша/море сверху не добавляет значимо).
- **H2.** Физика A2 даёт устойчивый выигрыш RMSE поверх статики и в других
  регионах (не региональный артефакт USA).

## Матрица (16 ранов, seed 0)

8 конфигов exp24-USA × 2 региона. IAM4VP несёт полный A/B (orog / static);
SimVPv2 и PredRNNv2 — только static (как в exp24).

| семейство | армы | регионы |
| --- | --- | --- |
| PI-IAM4VP | nophys/a2 × orog/static | france, npac |
| PI-SimVPv2 | nophys/a2 × static | france, npac |
| PI-PredRNNv2 | nophys/a2 × static | france, npac |

Регионы (32×64, как в exp22):

| регion | data.cut | physics_lat_start_deg |
| --- | --- | --- |
| france | `[[81, 113], [0, 64]]` | 26.71875 |
| npac | `[[74, 106], [96, 160]]` | 16.875 |

Конфиги `configs/exp26/` генерируются `make_configs.py` из USA-родителей exp24
(руками не править). Регион меняет РОВНО геометрию (`data.cut`, `static_cut`,
`diabatic_cut` у A2, `physics_lat_start_deg`, имя); бюджет эпох, батч, lr, набор
статических полей и физрежим наследуются от exp24 бит-в-бит — сравнение внутри
региона честно на общей эпохе.

## Запуск

```bash
# generate
python docs/experiments/26_cross_region_static/make_configs.py

# train (регион берётся из имени конфига; IAM4VP/PredRNNv2 — NGPU=2)
ACCOUNT=proj_1717 NGPU=2 EXP26_JOBS="exp26_iam4vp_nophys_orog_france \
  exp26_iam4vp_a2_orog_france exp26_iam4vp_nophys_static_france \
  exp26_iam4vp_a2_static_france" bash sh_files/exp26_train.sh
```

Кластерный worktree `~/wt_exp26` требует минимальный `.env` (3 COMET_-строки) в
корне. Мемапы регионов (`predformer_{france,npac}_2000_2004.dat`), климатология и
пороги уже репакнуты (использовались в exp22).

## Протокол сравнения

Как в exp24: общая эпоха, парный бутстрап (`tools/metrics_ladder.py` /
`sh_files/exp24_metrics_eval.sh`-образный харнесс с регионом), ограниченные
метрики — абсолютной разницей. Результаты — сюда.
