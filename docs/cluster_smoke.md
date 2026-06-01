# Cluster Smoke Checklist

Запусти команды по порядку. Каждая — отдельная проверка ядра миграции на
суперкомпе. Если шаг падает — остальные пропускать, разбираемся.

Все команды предполагают, что ты в корне репозитория и активировал нужный
conda-env с обновлённым `requirements.txt`.

## 0. Окружение

```bash
uv pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "
import comet_ml, yaml, h5netcdf, xarray, einops, timm
print('comet_ml:', comet_ml.__version__)
print('yaml:', yaml.__version__)
print('h5netcdf:', h5netcdf.__version__)
print('xarray:', xarray.__version__)
print('einops:', einops.__version__)
print('timm:', timm.__version__)
"
```

Ожидаем: `torch.cuda.is_available()` True; `comet_ml` 3.57+.

## 1. Импорты, реестры, модели

```bash
python -c "
import Models, Data, training_strategies
from utils.registry import MODELS, DATASETS, STRATEGIES
print('MODELS:    ', sorted(MODELS))
print('DATASETS:  ', sorted(DATASETS))
print('STRATEGIES:', sorted(STRATEGIES))
m = MODELS['SimVP'](in_shape=[12, 69, 32, 64])
print('SimVP params:', sum(p.numel() for p in m.parameters()))
m = MODELS['PI-IAM4VP']()
print('IAM4VP params:', sum(p.numel() for p in m.parameters()))
"
```

Ожидаем 9 моделей, 4 датасета, 6 ключей стратегий:
`SimVP`, `WeatherGFT`, `WeatherGFTSingle`, `PredFormer`, `PredFormerGFT`,
`PredFormerGFT_HybridBlock`, `PI-IAM4VP`, `PredRNN`, `PredRNNv2`;
`v1`, `v3`, `v3_memmap`, `v4`.

## 2. Lint

```bash
ruff check trainer.py train.py utils/registry.py utils/distributed.py \
  utils/experiment.py utils/checkpointing.py utils/early_stopping.py \
  training_strategies/ train/__init__.py train/_common.py \
  train/train_*.py train/dev/__init__.py train/dev/train.py \
  train/dev/train_single_imvp.py \
  Models/__init__.py Data/__init__.py
```

Ожидаем: `All checks passed!`.

## 3. Single-GPU smoke (1 эпоха)

Используй самый дешёвый v4-конфиг, если memmap доступен. Для настоящего
smoke лучше временно скопировать YAML, поставить `training.max_epoch: 1`
и, при необходимости, укоротить `data.train.end_time` / `data.val.end_time`.
Не коммить временный конфиг.

```bash
# Через launcher (torchrun под капотом):
NGPUS=1 NNODES=1 bash sh_files/launch_train.sh simvp_usa
```

Ожидаем: процесс не падает, в `${checkpoint_base}/${experiment.name}/...`
появляются `best.pt`, `last.pt`, `epoch=XX-val_loss=YYYY.pt`. В Comet
прилетают `train_loss`, `lr`, `val_loss`, `RMSE_*`.

## 4. Multi-GPU smoke на одном узле (DDP, 2 GPU)

```bash
NGPUS=2 NNODES=1 bash sh_files/launch_train.sh simvp_usa
```

Что проверяем:

- обе видяшки в `nvidia-smi` нагружены;
- лог `[distributed] world_size=2 rank=0` появляется ровно из одного процесса;
- Comet получает метрики только с rank 0 (без дублей).

## 5. Multi-node smoke (если есть кластерное время)

`sbatch sh_files/train_usa_2gpu.sh simvp_usa` (или твой эквивалент) с
`--nodes=N --gres=gpu:K`. `torchrun` подхватит `SLURM_*` через
`launch_train.sh`. Обрати внимание на `MASTER_ADDR` — для multi-node
нужно подставлять адрес head-ноды (например, из `SLURM_NODELIST`).

## 6. Конвертация старого Lightning checkpoint

```bash
python -m utils.checkpointing convert \
  "${WEATHERPRED_CHECKPOINT_BASE:-./checkpoints}/SimVP-USA/<run-id>/epoch=NN-val_loss=YYYY.ckpt" \
  /tmp/converted.pt

# Загрузить в чистую модель и убедиться что веса совпадают:
python -c "
import torch
from Models.SimVP import SimVP_Model
from utils.checkpointing import load_checkpoint
m = SimVP_Model(in_shape=(12, 69, 32, 64))
meta = load_checkpoint('/tmp/converted.pt', m, strict=True)
print('OK, meta:', meta)
"
```

## 7. Numerical parity (опционально, дорогое)

Это самый дорогой шаг плана. Делать, если хочется убедиться что в новом
коде нет регрессов в обучении.

```bash
# 1) Закоммить текущее состояние new branch.
git status

# 2) Запустить 1 эпоху новым кодом с фиксированным seed:
#    добавь во временный YAML training.seed: 42
PYTHONHASHSEED=42 NGPUS=1 NNODES=1 bash sh_files/launch_train.sh simvp_usa

# 3) Переключиться на main (Lightning), запустить ту же конфигурацию:
git checkout main
PYTHONHASHSEED=42 NGPUS=1 NNODES=1 bash sh_files/launch_train.sh simvp_usa

# 4) Сравнить логи Comet (train_loss за каждый batch, val_loss за эпоху).
#    Толерантность 1e-4 — нормально; больше — копаем.
```

Типичные источники расхождения (по убыванию вероятности):

1. Scheduler step semantics — проверь что в `trainer.py:_train_one_epoch`
   `scheduler.step()` вызывается после каждого `optimizer.step()` (как
   и было в старой конфигурации). Сделано.
2. `DistributedSampler.set_epoch` — в `trainer.fit` есть вызов на DDP-сэмплере.
3. `optimizer.zero_grad(set_to_none=True)` — передаётся явно.
4. DataLoader workers' RNG — мы пока не ставим `worker_init_fn`. Если parity
   не сходится из-за этого — добавить, формулу можно подсмотреть в любой
   распределённой обёртке (например, в исходниках старого фреймворка).

## Если что-то падает

- `ModuleNotFoundError: Models` или `Data` при запуске из `train/train_*.py` —
  проверить, что `sys.path.append` указывает в корень проекта.
- `init_process_group` падает — проверить, что `torchrun` (не `python`)
  запускает процесс; `MASTER_ADDR`/`MASTER_PORT` доступны.
