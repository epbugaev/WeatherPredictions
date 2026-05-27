# Запуск тренировок на cHARISMa из локальной машины

Документ описывает фактический workflow: правки на ноутбуке → коммит → одна команда submit-ит job на HSE cHARISMa.

## 1. Предусловия

### Локально

- SSH-алиас `cluster` настроен в `~/.ssh/config` и ходит на cHARISMa по ключу (без пароля). Если ещё нет — настройка с нуля:

  1. Сгенерировать ключ (если нет):

     ```bash
     ssh-keygen -t ed25519 -f ~/.ssh/charisma_hse_key -C "fa.buzaev@hse.ru"
     ```

  2. Положить публичную часть на cHARISMa (через web-портал HSE HPC, либо через админа кластера):

     ```bash
     cat ~/.ssh/charisma_hse_key.pub
     ```

  3. Добавить запись в `~/.ssh/config` (создать файл, если нет; `chmod 600 ~/.ssh/config`):

     ```sshconfig
     Host cluster cluster.hpc.hse.ru
         HostName cluster.hpc.hse.ru
         User fa.buzaev
         Port 2222
         IdentityFile ~/.ssh/charisma_hse_key
         ServerAliveInterval 60
         ForwardAgent no
     ```

     Под другого пользователя: поменять `User`, `IdentityFile` и (по желанию) имя алиаса. `Port 2222` и `HostName cluster.hpc.hse.ru` для cHARISMa фиксированы.

  4. Проверка:

     ```bash
     ssh -o BatchMode=yes cluster hostname
     ```

     Должно ответить именем head-ноды (например, `sms`) без запроса пароля.

- Git-remote `origin` совпадает с тем, что прописан в клоне на кластере. Проверка:

  ```bash
  git remote -v
  ssh cluster "cd /home/fa.buzaev/WeatherPredictions && git remote -v"
  ```

  Обе команды должны вернуть один и тот же URL.

- Push-доступ к этому remote есть.

### На кластере (одноразово)

- Клон лежит в [`/home/fa.buzaev/WeatherPredictions`](../).
- Создано conda-окружение `weatherpred-gft-fix` (см. [`environment.yml`](../environment.yml) + [`requirements.txt`](../requirements.txt)).
- Активируется одним из двух способов (автоматически в [`sh_files/_shell_contract.sh`](../sh_files/_shell_contract.sh)):
  1. `CONDA_ENV_BIN=/home/fa.buzaev/.conda/envs/weatherpred-gft-fix/bin` — prepend в `PATH`. Самый быстрый путь; можно прописать в `~/.bashrc` на кластере или в `${REPO_ROOT}/.env`.
  2. Fallback: `module load Python/Miniconda_v25` → `conda activate weatherpred-gft-fix`. Работает только под Slurm.

## 2. Локальный dev-loop

```bash
# 1. Правишь код в IDE.
# 2. Коммитишь (--allow-dirty в submit-скрипте не пропустит несохранённое — это сознательно).
git add -A
git commit -m "feat: tweak predformer lr schedule"

# 3. Один submit-скрипт делает всё остальное:
bash sh_files/remote_submit.sh sh_files/train_simvp_usa_v4_2gpu.sh
```

Что произойдёт под капотом ([`sh_files/remote_submit.sh`](../sh_files/remote_submit.sh)):

1. Проверка чистоты worktree (`git diff --quiet HEAD` + untracked files).
2. `git push origin HEAD:<current-branch>`.
3. По SSH к `cluster`:
   - `cd /home/fa.buzaev/WeatherPredictions`
   - `git fetch --prune origin`
   - `git checkout <branch>` (или `-b` если не существует локально на кластере)
   - `git pull --ff-only origin <branch>` — упадёт, если истории разошлись (вместо тихого `reset --hard`).
   - `sbatch sh_files/train_simvp_usa_v4_2gpu.sh`
4. Парсит `JobID` из вывода `sbatch`, выводит пути к stdout/stderr.

### Tail-логи сразу после submit

```bash
bash sh_files/remote_submit.sh -f sh_files/train_simvp_usa_v4_2gpu.sh
```

Скрипт дождётся появления файлов в `logs/` и запустит `tail -F` обоих стримов. Ctrl-C **только** отрывает tail — job продолжает крутиться.

### Submit без коммита

```bash
bash sh_files/remote_submit.sh --allow-dirty sh_files/train_simvp_usa_v4_2gpu.sh
```

Внимание: незакоммиченные изменения на кластер **не уезжают**. Запустится последний коммит ветки. Флаг нужен только чтобы не падать на проверке worktree.

### Переопределение хоста и пути

```bash
REMOTE_HOST=alt-cluster REMOTE_REPO=/path/to/repo \
  bash sh_files/remote_submit.sh sh_files/train_simvp_usa_v4_2gpu.sh
```

## 3. Структура запуска на кластере

Цепочка вызовов после `sbatch sh_files/train_simvp_usa_v4_2gpu.sh`:

```text
sbatch
  └── sh_files/train_simvp_usa_v4_2gpu.sh     # #SBATCH-директивы (gres, time, constraint)
        └── sh_files/launch_train.sh simvp_usa_v4
              └── sh_files/_shell_contract.sh   # REPO_ROOT, conda env, .env, PYTHONPATH
              └── exec python -m torch.distributed.run \
                    --nnodes ${NNODES} \
                    --nproc_per_node ${NGPUS} \
                    --rdzv_backend c10d \
                    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
                    train.py --config configs/simvp_usa_v4.yaml
```

### Параметры SLURM (используемые)

Все `#SBATCH`-директивы зашиты в `sh_files/train_*.sh`. Типовые значения:

| Директива | Значение | Что означает |
| --- | --- | --- |
| `--gres=gpu:1` | 1 GPU | один GPU per job |
| `--cpus-per-task=8` | 8 CPU | под DataLoader workers |
| `--time=2-18:00:00` | 2 суток 18 часов | hard limit |
| `--constraint="type_e\|type_f"` | A100 \| H100 | константы cHARISMa |
| `--output=logs/slurm-%x-%j.out` | по JobName+JobID | stdout |
| `--error=logs/slurm-%x-%j.err` | то же | stderr |

GPU-типы (через `--constraint`):

- `type_a/b/c/d` — V100
- `type_e` — A100
- `type_f` — H100
- `type_h` — H200 (экспериментальный)

## 4. Мониторинг и управление

```bash
# Очередь — только свои job-ы:
ssh cluster squeue -u fa.buzaev

# Конкретный job:
ssh cluster squeue -j <JOB_ID>

# История завершённых:
ssh cluster sacct -u fa.buzaev --starttime=today --format=JobID,JobName,State,Elapsed,MaxRSS

# Tail логов уже запущенного job-а:
ssh cluster "cd /home/fa.buzaev/WeatherPredictions && tail -F logs/slurm-train-simvp-usa-v4-2gpu-<JOB_ID>.{out,err}"

# Отмена:
ssh cluster scancel <JOB_ID>
```

## 5. Добавление нового эксперимента

1. Положить YAML в [`configs/`](../configs/), например `configs/myexp.yaml`.
2. Скопировать любой `sh_files/train_*.sh` под новое имя, в нём:
   - поменять `#SBATCH --job-name=` на осмысленный (он же попадёт в `logs/slurm-<job-name>-%j.{out,err}`);
   - подкрутить `--time`, `--constraint` под нужный GPU;
   - в последней строке поменять стем конфига: `exec bash "${weatherpred__launcher}" myexp "$@"`.
3. Закоммитить и submit-нуть через `bash sh_files/remote_submit.sh sh_files/train_myexp.sh`.

## 6. Локальный запуск (без cHARISMa)

Иногда нужно прогнать smoke на ноутбуке — например, проверить, что dataset открывается:

```bash
conda activate weatherpred-gft-fix
bash sh_files/launch_train.sh simvp_usa_v4
```

`launch_train.sh` обнаружит отсутствие `SLURM_JOB_ID` и не будет звать `module load`. Использует уже активированное окружение. `NNODES`/`NGPUS` по умолчанию `1`/`1` — на маке без CUDA `torchrun` поднимется в CPU-режиме (или упадёт на `nccl`), поэтому на маке smoke имеет смысл прерывать на этапе `build_dataset`.

Для v4-конфигов можно временно переопределить пути без изменения YAML:

```bash
MEMMAP_PATH_OVERRIDE=/path/to/predformer_usa_2000_2004.dat \
MEMMAP_META_PATH_OVERRIDE=/path/to/predformer_usa_2000_2004.meta.json \
CHECKPOINT_BASE_OVERRIDE=/tmp/weatherpred-checkpoints \
  bash sh_files/launch_train.sh simvp_usa_v4
```

## 7. Troubleshooting

| Симптом | Причина | Что делать |
| --- | --- | --- |
| `Permission denied (publickey)` при ssh | ключ не загружен в агент | `ssh-add ~/.ssh/charisma_hse_key` или поставить в `~/.ssh/config` `IdentityFile` |
| `[remote_submit] Есть незакоммиченные изменения` | dirty worktree | `git commit` или `--allow-dirty` (правки не уедут) |
| `fatal: Not possible to fast-forward` на pull | история разошлась (force-push? коммиты на кластере?) | вручную: `ssh cluster "cd .../WeatherPredictions && git status && git log --oneline -5"`. Не делать `reset --hard` без понимания, что теряется. |
| `sbatch: error: invalid partition specified` | partition не подходит для constraint | убрать `--partition=...` либо подобрать совместимый |
| `srun: error: Unable to allocate resources` | в очереди нет узлов с таким `--constraint` | ослабить (`type_e\|type_f\|type_d`) или увеличить `--time` ожидания |
| `module: command not found` | не Slurm-сессия или нестандартный shell | задать `CONDA_ENV_BIN` явно в `.env` репозитория |
| `train.py: ModuleNotFoundError: comet_ml` | окружение не активировано | проверить `which python` в логах джоба — должен быть `.../envs/weatherpred-gft-fix/bin/python` |
| `Address already in use` в `torch.distributed` | два job-а используют один `MASTER_PORT` | новые `train_*.sh` задают порт от `SLURM_JOB_ID`; если запускаешь вручную, выставь `MASTER_PORT=29xxx` |
| `Memmap cut ... does not match config cut ...` | YAML и `.meta.json` описывают разные окна | используй правильный memmap или переопредели `MEMMAP_PATH_OVERRIDE`/`MEMMAP_META_PATH_OVERRIDE` |

## 8. Файлы, на которые опирается этот workflow

- [`sh_files/_shell_contract.sh`](../sh_files/_shell_contract.sh) — bootstrap окружения на кластере
- [`sh_files/launch_train.sh`](../sh_files/launch_train.sh) — generic launcher → `train.py`
- [`sh_files/train_*.sh`](../sh_files/) — конкретные SLURM job-скрипты
- [`sh_files/remote_submit.sh`](../sh_files/remote_submit.sh) — локальный submit-helper
- [`train.py`](../train.py) — единая точка входа обучения
- [`configs/`](../configs/) — YAML-конфиги экспериментов
