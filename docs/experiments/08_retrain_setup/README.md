# Переобучение на исправленных уравнениях (A100)

Готовая инфраструктура для переобучения `PI-IAM4VP-ResidualStablePhysics` на исправленной физике (ветка `fix_inline_equations`). **Запуск не выполнялся** — по решению оставлен на потом (открыт вопрос region-aware геометрии + AAAI27-джобы держат GPU-квоту).

## Файлы (остаются в рабочих папках, чтобы быть запускаемыми)

- **`sh_files/train_pi_iam4vp_residual_fixedeq_a100.sh`** — sbatch, 1× A100 (`rocky`, `type_e|type_f|type_h`). Пинит `REPO_ROOT` на исправленную копию, branch-guard на пропатченный Кориолис, `CHECKPOINT_BASE_OVERRIDE=/home/fa.buzaev/checkpoints`, стейджинг memmap в node-local `/tmp`, запуск через `launch_train.sh`.
- **`configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml`** — копия residual-конфига с именем эксперимента `...-FIXEDEQ` для чистого A/B-сравнения со старым (багованным) раном.

## Запуск

```bash
cd /home/fa.buzaev/WeatherPredictions
sbatch sh_files/train_pi_iam4vp_residual_fixedeq_a100.sh
```

Эксперимент уйдёт в Comet как `PI-IAM4VP-ResidualStablePhysics-USA-v4-FIXEDEQ`. `set_physics_normalization` вызывается автоматически из статистик датасета.

## Проверки перед запуском (выполнены)

- Синтаксис sbatch (`bash -n`) OK
- Branch-guard: пропатченный `f_field = 2·7.2921e-5·sinφ` присутствует в `Models/WeatherGFT.py`
- Конфиг резолвится: exp_name `...-FIXEDEQ`, checkpoint_base `/home/fa.buzaev/checkpoints`
- Construction-валидация: WeatherGFT 470547485 параметров, PI-IAM4VP 12868559, forward без NaN/Inf, число параметров не изменилось после фикса (см. `02_old_vs_new_block/construction_validation.json`)
