# Эксперимент 24 — статические входы (орография + маска суша/море)

Spec: [docs/superpowers/specs/2026-07-16-exp24-static-inputs-design.md](../../superpowers/specs/2026-07-16-exp24-static-inputs-design.md)

## Вопрос

Помогают ли модели статические поля (орография, lsm), поданные как
дополнительные входные каналы бэкбона? И меняется ли при этом выигрыш
физического приора A2-exp13?

- **H1.** Статика улучшает обе армы (no_physics и A2) по RMSE/ACC на общей эпохе.
- **H2.** Выигрыш A2 поверх статики сжимается: диабатический блок уже видит
  орографию/lsm/|lat| через `_load_static_geo`, часть его вклада может быть
  чисто географической.

## Матрица (6 ранов, USA, seed 0)

| семейство | nophys+static | a2+static | база без статики (существующие раны) |
| --- | --- | --- | --- |
| PI-IAM4VP | exp24-iam4vp-nophys-static-usa-s0 | exp24-iam4vp-a2-static-usa-s0 | abl16L r0/r3 t12 (500 эпох) |
| PI-PredRNNv2 | exp24-predrnnv2-nophys-static-usa-s0 | exp24-predrnnv2-a2-static-usa-s0 | exp20 p0/p3 (40 эпох) |
| PI-SimVPv2 | exp24-simvpv2-nophys-static-usa-s0 | exp24-simvpv2-a2-static-usa-s0 | exp21L s0/s3 t12 (500 эпох) |

Конфиги: `configs/exp24/` (генерятся `make_configs.py`, руками не править).
Бюджет/батч/lr — бит-в-бит от родителя: пара «static vs база» сравнивается в
одинаковом харнессе, армы ранжируются только на общей эпохе (уроки exp16/18).

## Механизм

`utils/static_input.py`: орография (z-score по кропу) + lsm (0..1) из
`constants_1.40625deg.nc`, буфер `(1, S, H, W)` в модели, конкат к кадрам на
входе первого слоя. Выход моделей — прежние 69 каналов; физпуть не тронут.
У PredRNNv2 статика приклеивается после scheduled-sampling-смешивания;
при patch_size>1 буфер патчится тем же патчем.

Буфер статики персистентен в чекпоинте, но конструирование static-модели всегда
читает constants-файл — при офлайн-eval вне кластера подмени `static_constants_path`.

## Запуск

```bash
EXP24_JOBS="exp24_simvpv2_nophys_static_usa exp24_simvpv2_a2_static_usa" \
  bash sh_files/exp24_train.sh
# IAM4VP / PredRNNv2 — тяжёлые: NGPU=2
```

Кластерный worktree: не забыть минимальный `.env` (3 COMET_-строки) в корень.

## Протокол сравнения

Общая эпоха; парный бутстрап (`tools/metrics_ladder.py`); ограниченные метрики
(ACC/CSI/FSS) — абсолютной разницей; PSD только по её модулю. Результаты — сюда.
