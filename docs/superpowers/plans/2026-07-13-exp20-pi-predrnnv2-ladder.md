# Эксперимент 20 — PI-PredRNNv2 и перенос лестницы exp16 на вторую архитектуру

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Починить PredRNNv2 (его декаплинг-лосс сейчас не доходит до оптимизатора), подключить к нему то же физядро, что и к PI-IAM4VP, и прогнать лестницу уравнений — чтобы узнать, переносится ли вывод exp16 «верная физика даёт ≈−2 %, сломанная вредит» на вторую, принципиально другую архитектуру.

**Architecture:** Физический residual-путь вынимается из `Models/IAM4VP.py` в общий миксин `utils/physics_residual.py` (он там уже архитектурно-нейтрален: работает в пространстве состояния `(B,69,H,W)`, не касается внутренностей модели). `IAM4VP` начинает наследовать миксин — побитово без изменения поведения. Новый `PI_PredRNNv2_Model` наследует тот же миксин и врезает коррекцию внутрь своего авторегрессивного цикла. Контракт `forward` моделей семейства PredRNN меняется с `(next_frames, loss)` на `(next_frames, aux_losses: dict)`, и стратегия впервые начинает эти аукс-лоссы прибавлять к задаче.

**Tech Stack:** Python 3.10+, PyTorch 2.6, DDP через `torch.distributed.run`, conda-env `pi-iamvp` на cHARISMa, Comet для метрик, ruff для линта.

## Global Constraints

- **Кластер:** sbatch всегда с `-A proj_1715`; env `WEATHERPRED_CONDA_ENV_NAME=pi-iamvp`; memmap `~/era5_memmap/predformer_usa_2000_2004.dat`; ноды пинить через `--constraint=type_e|type_f|type_h` (v100 в 6–9× медленнее и сожжёт walltime).
- **Не трогать обучающий каркас:** `train.py`, `trainer.py`, `sh_files/launch_train.sh`, `sh_files/train_*.sh`. Всё, что нужно — прокидывается конфигом и env.
- **Чужие джобы не трогать:** на кластере крутится волна exp16-long (`abl16*_t12`, 8 джоб, лимит 2 суток). Ничего из неё не отменять.
- **CLAUDE.md:** никаких `try/except` (LBYL), никаких локальных импортов, никакого мёртвого кода, docstring в Google-стиле с shape тензоров, type hints на публичных сигнатурах. После правок — `ruff check .` и `ruff format .`.
- **Новый worktree на кластере требует ручной копии `.env`** (он gitignored и не наследуется).
- **Ранжировать армы только на общей эпохе.** Разброс эпох `best.pt` стоит ~10 % скилла — больше, чем весь эффект физики (2–7 %). Это вывод §11.3 exp16, нарушение которого уже стоило отзыва exp18.

---

## Лестница exp20

Армы P2a/P2 из exp16 (состояния кода до аудита 13) **невоспроизводимы** и в лестницу не входят: они живут в worktree `wt_prefix13`, где PI-PredRNNv2 никогда не существовало, а бэкпорт нового кода в старое дерево лишил бы их смысла (они и есть «старое дерево»). Это фиксируется в отчёте как ограничение.

| # | Арм | Отличие от предыдущей ступени | Конфиг |
|---|-----|-------------------------------|--------|
| P0 | `no_physics` | физики нет; residual-голова той же ёмкости (контроль архитектуры) | `exp20_p0_no_physics_*.yaml` |
| P1 | `legacy_hybrid` | легаси-уравнения в нормализованном пространстве (`legacy_normalized`) | `exp20_p1_legacy_hybrid_*.yaml` |
| P3a | `A2-noQ` | `stable_physical_v2` passthrough + знаковые фиксы, **без** диабатика | `exp20_p3a_no_diabatic_*.yaml` |
| P3 | `A2+13` | + обучаемый диабатик Q_θ (`diabatic_apply_to: t_and_q`) | `exp20_p3_a2_exp13_*.yaml` |
| P4 | `+exp14` | + адвективная форма, метрика сферы, трение, lagrange3, mass-consistent ω | `exp20_p4_exp14_*.yaml` |
| P5 | `+exp15` | + `omega_free`, скрытое тепло, клим. Q₁/Q₂ | `exp20_p5_exp15_*.yaml` |

Нумерация арм намеренно повторяет exp16 (P↔R), чтобы таблицы сравнивались построчно.

---

## Файловая структура

| Файл | Ответственность |
|---|---|
| `utils/physics_residual.py` (создать) | `PhysicsResidualMixin` — вся физическая обвязка residual-пути: нормализация, санитайзеры, гард градиентов, физприор, коррекция, аукс-лосс, диагностика. Ничего не знает об архитектуре-носителе. |
| `Models/IAM4VP.py` (изменить) | Наследует миксин; удаляются перенесённые методы. Поведение — побитово прежнее. |
| `Models/PredRNN.py` (изменить) | Контракт `forward -> (next_frames, aux_losses)`; новый `PI_PredRNNv2_Model`. |
| `Models/__init__.py` (изменить) | Регистрация `PI-PredRNNv2`. |
| `training_strategies/predrnn.py` (изменить) | Прибавляет аукс-лоссы к задаче; логирует физдиагностику. |
| `configs/exp20/*.yaml` (создать) | 6 арм короткой волны + 6 арм long. |
| `tests/test_pi_predrnnv2.py` (создать) | Контракт PI-PredRNNv2. |
| `tests/test_physics_residual_mixin.py` (создать) | Побитовая эквивалентность IAM4VP до/после рефакторинга. |
| `docs/experiments/20_pi_predrnnv2_ladder/` (создать) | Дизайн, результаты, `collect_metrics.py`, `make_figures.py`. |

---

### Task 1: Вынести физический residual-путь в общий миксин

Цель — устранить дублирование до его появления: `PI-PredRNNv2` не должен копипастить 650 строк из `IAM4VP`. Риск задачи — тихо изменить поведение модели, на которой стоят все результаты exp16. Поэтому задача начинается с золотого теста, фиксирующего текущий выход побитово.

**Files:**
- Create: `utils/physics_residual.py`
- Create: `tests/test_physics_residual_mixin.py`
- Modify: `Models/IAM4VP.py` (удалить перенесённые методы, добавить наследование)

**Interfaces:**
- Produces: `PhysicsResidualMixin` со следующим публичным контрактом, на который опираются Task 3 и `training_strategies/`:
  - `init_physics_residual(*, C_data: int, H_data: int, W_data: int, downscaling_factor_all: int, **physics_kwargs) -> None` — вызывается из `__init__` модели-носителя после `super().__init__()`; строит буферы, `HybridBlock`, `PhysicsTendencyResidualCorrector`, диабатическую голову; валидирует все режимы fail-fast.
  - `set_physics_normalization(mean: torch.Tensor, std: torch.Tensor) -> None`
  - `_apply_physics_residual(y_nn: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor` — обе `(B, C_data, H, W)` нормализованные; возвращает скорректированное предсказание.
  - `physics_residual_aux_loss() -> torch.Tensor | None`
  - `physics_residual_diagnostics() -> dict[str, torch.Tensor]`
  - `set_residual_warmup(active: bool) -> None`
  - атрибуты `freeze_iam4vp_for_residual_warmup: bool`, `residual_warmup_epochs: int` (имена сохраняются — по ним duck-typing в `iterative_manual.py:72-101`).

- [ ] **Step 1: Написать золотой тест побитовой эквивалентности (падает — модуля ещё нет)**

Тест строит `IAM4VP` на фиксированном сиде, гоняет forward, и сравнивает с эталоном, снятым с текущего `main`-поведения. Эталон снимается в этом же тесте до рефакторинга и кладётся в `tests/goldens/` — так тест ловит регрессию, а не собственную реализацию.

```python
# tests/test_physics_residual_mixin.py
"""Побитовая эквивалентность IAM4VP до и после выноса физики в миксин."""

from __future__ import annotations

import unittest

import torch

from Models.IAM4VP import IAM4VP
from utils.physics_residual import PhysicsResidualMixin

PHYSICS_ARM_R5 = {
    "use_physics": False,
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 32,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_feature_mode": "tendency",
    "physics_residual_hybrid_steps": 1,
    "physics_residual_hybrid_mode": "stable_physical_v2",
    "physics_w_diagnostic": "mass_consistent",
    "physics_advection_form": "advective",
    "physics_metric_terms": True,
    "physics_spherical_divergence": True,
    "physics_rayleigh_friction": True,
    "physics_vertical_scheme": "lagrange3",
    "physics_omega_free": ["t", "q", "u", "v"],
    "physics_latent_heating_coupling": True,
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
    "T_data": 12,
}


def _build_iam4vp(**overrides):
    torch.manual_seed(0)
    params = dict(PHYSICS_ARM_R5)
    params.update(overrides)
    model = IAM4VP(**params)
    model.eval()
    mean = torch.zeros(69)
    std = torch.ones(69)
    model.set_physics_normalization(mean, std)
    return model


class TestIAM4VPInheritsMixin(unittest.TestCase):
    def test_iam4vp_is_a_physics_residual_mixin(self) -> None:
        self.assertTrue(issubclass(IAM4VP, PhysicsResidualMixin))

    def test_apply_physics_residual_is_not_redefined_on_iam4vp(self) -> None:
        # Метод должен приходить из миксина, а не остаться копией в IAM4VP.
        self.assertNotIn("_apply_physics_residual", IAM4VP.__dict__)
        self.assertNotIn("_physics_prior_from_state", IAM4VP.__dict__)
        self.assertNotIn("set_physics_normalization", IAM4VP.__dict__)


class TestPhysicsResidualGolden(unittest.TestCase):
    """Выход физического пути не изменился при выносе в миксин."""

    def test_apply_physics_residual_matches_golden(self) -> None:
        model = _build_iam4vp()
        torch.manual_seed(1)
        prev_state = torch.randn(2, 69, 32, 64)
        y_nn = torch.randn(2, 69, 32, 64)

        with torch.no_grad():
            y_hat = model._apply_physics_residual(y_nn, prev_state)

        golden = torch.load("tests/goldens/iam4vp_physics_residual_r5.pt")
        torch.testing.assert_close(y_hat, golden["y_hat"], rtol=0, atol=0)
        torch.testing.assert_close(
            model.physics_residual_aux_loss(), golden["aux_loss"], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Снять эталон с ТЕКУЩЕГО (до-рефакторингового) кода**

Это делается ДО правки `Models/IAM4VP.py` — иначе эталон зафиксирует уже сломанное поведение. Скрипт одноразовый, в репозиторий не коммитится (пишем во временный файл, запускаем, удаляем).

```bash
cat > /tmp/make_golden.py <<'PY'
import torch
from Models.IAM4VP import IAM4VP

PARAMS = { ... }  # ровно PHYSICS_ARM_R5 из теста, скопировать дословно

torch.manual_seed(0)
model = IAM4VP(**PARAMS)
model.eval()
model.set_physics_normalization(torch.zeros(69), torch.ones(69))
torch.manual_seed(1)
prev_state = torch.randn(2, 69, 32, 64)
y_nn = torch.randn(2, 69, 32, 64)
with torch.no_grad():
    y_hat = model._apply_physics_residual(y_nn, prev_state)
torch.save(
    {"y_hat": y_hat, "aux_loss": model.physics_residual_aux_loss()},
    "tests/goldens/iam4vp_physics_residual_r5.pt",
)
print("golden saved", y_hat.shape, y_hat.float().abs().mean().item())
PY
python /tmp/make_golden.py && rm /tmp/make_golden.py
```

Ожидаемо: печатает `golden saved torch.Size([2, 69, 32, 64]) <число>`. Файл `tests/goldens/iam4vp_physics_residual_r5.pt` создан и коммитится.

- [ ] **Step 3: Прогнать тест — должен упасть на импорте**

Run: `python -m pytest tests/test_physics_residual_mixin.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.physics_residual'`.

- [ ] **Step 4: Создать `utils/physics_residual.py` — перенести код механически**

Перенести из `Models/IAM4VP.py` **дословно, без изменений логики**:
- из `__init__` (строки ~330-639): всё присваивание `self.physics_*` / `self.diabatic_*`, регистрацию буферов `physics_data_mean`, `physics_data_std`, `physics_pressure_pa`, всю fail-fast валидацию режимов, построение `self.hybrid_block`, регистрацию градиентных хуков, построение `self.physics_residual_corrector` и `self.diabatic_head`, все `warnings.warn` — в метод `init_physics_residual(...)`;
- методы целиком: `set_physics_normalization`, `_require_physics_normalization`, `_denormalize_state`, `_normalize_state`, `_nonfinite_ratio`, `_finite_or_fallback`, `_finite_clamp`, `_sanitize_hybrid_param_grad`, `_sanitize_physical_parts`, `_sanitize_hybrid_latent_physical`, `_clip_normalized_tendency`, `_hybrid_block_forward`, `_hybrid_bn_gamma_drift`, `x_to_zquvtw`, `_rms`, `_load_static_geo`, `_physics_prior_from_state`, `_build_diabatic_mask`, `_residual_slice`, `_apply_physics_residual`, `physics_residual_aux_loss`, `physics_residual_diagnostics`, `set_residual_warmup`.

Обязательные правки при переносе (иначе миксин останется IAM4VP-специфичным):
1. Ссылки на `self.C_data`, `self.surface_channels`, `self.downscaling_factor_all` остаются — но теперь это **контракт миксина**: их выставляет `init_physics_residual`, а не `__init__` носителя. Соответственно `init_physics_residual` принимает `C_data`, `H_data`, `W_data`, `downscaling_factor_all` и присваивает их сам.
2. Сообщение об ошибке `"PI-IAM4VP HybridBlock currently has hardcoded derivative geometry..."` → `"Physics HybridBlock has hardcoded derivative geometry for an 8x16 latent grid..."` (без имени архитектуры).
3. Аналогично во всех `warnings.warn`: `"PI-IAM4VP residual-corrector mode..."` → `"Physics residual-corrector mode..."`.
4. Класс объявляется как `class PhysicsResidualMixin:` — без наследования `nn.Module` (носитель уже `nn.Module`; миксин полагается на `register_buffer`/`parameters()` носителя). В docstring это зафиксировать явно.

- [ ] **Step 5: Переключить `Models/IAM4VP.py` на миксин**

```python
# было
class IAM4VP(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
        # ~310 строк физической инициализации

# стало
from utils.physics_residual import PhysicsResidualMixin

class IAM4VP(PhysicsResidualMixin, nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
        self.init_physics_residual(
            C_data=C_data,
            H_data=H_data,
            W_data=W_data,
            downscaling_factor_all=self.downscaling_factor_all,
            use_physics_residual_corrector=use_physics_residual_corrector,
            physics_feature_mode=physics_feature_mode,
            # ... все physics_* / diabatic_* / freeze_* аргументы конструктора
        )
```

Удалить из `IAM4VP` все перенесённые методы. Легаси-путь `use_physics=True` (физика в латенте, строки 1313-1349) **остаётся в IAM4VP** — он архитектурно-специфичен (`self.lp_phys`, mask_token) и в миксин не идёт.

- [ ] **Step 6: Прогнать золотой тест + весь существующий набор**

Run: `python -m pytest tests/ -v`
Expected: PASS, все 31+ тестов, включая `test_physics_residual_mixin.py`, `test_physics_hybrid_exp16_port.py`, `test_physics_sign_conventions.py`, `test_exp16_rollout.py`.
Если золотой тест падает — рефакторинг изменил поведение; чинить, а не переснимать эталон.

- [ ] **Step 7: Линт и коммит**

```bash
ruff check . && ruff format --check .
git add utils/physics_residual.py Models/IAM4VP.py tests/test_physics_residual_mixin.py tests/goldens/iam4vp_physics_residual_r5.pt
git commit -m "refactor(physics): extract PhysicsResidualMixin from IAM4VP

Физический residual-путь не зависел от архитектуры-носителя, но жил
внутри IAM4VP. Вынесен в utils/physics_residual.py, чтобы PI-PredRNNv2
(exp20) переиспользовал его, а не копировал. Поведение IAM4VP
зафиксировано побитовым золотым тестом."
```

---

### Task 2: Оживить декаплинг-лосс PredRNNv2 (аукс-контракт)

Сейчас `PredRNNv2_Model.forward` считает декаплинг-штраф и возвращает его в связке с внутренним MSE — а стратегия этот второй элемент выбрасывает (`y_hat_perm, _ = model(inp, mask)`). В результате `decouple_beta` ни на что не влияет, `adapter` не получает градиента, и v2 обучается как v1.

Чинить прямым «прибавить возвращённый loss» нельзя: внутренний MSE модели конфликтует с MAE-лоссом стратегии (`loss_type: MAE`) — мы бы молча подмешали вторую задачу. Поэтому модель начинает возвращать **только аукс-члены**, отдельным словарём, а задачу считает стратегия.

**Files:**
- Modify: `Models/PredRNN.py` (обе модели: контракт `forward`)
- Modify: `training_strategies/predrnn.py`
- Create: `tests/test_predrnn_aux_loss.py`

**Interfaces:**
- Consumes: ничего из Task 1.
- Produces: контракт `forward(frames_tensor, mask_true) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`, где второй элемент — **уже взвешенные** аукс-лоссы, готовые к прибавлению. Ключи: `"decouple"` (только v2), `"physics_aux"` (только PI-варианты, Task 3). Task 3 опирается на этот контракт.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_predrnn_aux_loss.py
"""Декаплинг-лосс PredRNNv2 доходит до оптимизатора."""

from __future__ import annotations

import unittest

import torch

from Models.PredRNN import PredRNN_Model, PredRNNv2_Model

CONFIGS = {
    "in_shape": (12, 69, 8, 16),
    "patch_size": 1,
    "filter_size": 3,
    "stride": 1,
    "layer_norm": True,
    "pre_seq_length": 4,
    "aft_seq_length": 4,
    "reverse_scheduled_sampling": 0,
    "decouple_beta": 0.1,
}


def _inputs(total_length: int = 8):
    torch.manual_seed(0)
    frames = torch.randn(2, total_length, 8, 16, 69)
    mask = torch.zeros(1, 4, 1, 1, 1)
    return frames, mask


class TestAuxLossContract(unittest.TestCase):
    def test_v1_returns_empty_aux_dict(self) -> None:
        model = PredRNN_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (2, 7, 8, 16, 69))
        self.assertEqual(aux, {})

    def test_v2_returns_weighted_decouple_loss(self) -> None:
        model = PredRNNv2_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        _, aux = model(frames, mask)
        self.assertEqual(set(aux), {"decouple"})
        self.assertEqual(aux["decouple"].ndim, 0)
        self.assertGreater(aux["decouple"].item(), 0.0)

    def test_decouple_beta_scales_the_term(self) -> None:
        frames, mask = _inputs()
        torch.manual_seed(0)
        model_a = PredRNNv2_Model(2, (16, 16), {**CONFIGS, "decouple_beta": 0.1})
        torch.manual_seed(0)
        model_b = PredRNNv2_Model(2, (16, 16), {**CONFIGS, "decouple_beta": 0.2})
        _, aux_a = model_a(frames, mask)
        _, aux_b = model_b(frames, mask)
        torch.testing.assert_close(aux_b["decouple"], 2.0 * aux_a["decouple"])

    def test_adapter_receives_gradient(self) -> None:
        """Регрессия: раньше adapter был мёртв (лосс выбрасывался стратегией)."""
        model = PredRNNv2_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        loss = next_frames.abs().mean() + aux["decouple"]
        loss.backward()
        self.assertIsNotNone(model.adapter.weight.grad)
        self.assertGreater(model.adapter.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Прогнать — должен упасть**

Run: `python -m pytest tests/test_predrnn_aux_loss.py -v`
Expected: FAIL — `aux` сейчас скаляр-тензор, а не dict; `self.assertEqual(aux, {})` падает.

- [ ] **Step 3: Поменять контракт в `Models/PredRNN.py`**

В `PredRNN_Model.forward` — убрать мёртвый MSE (он никем не читается; CLAUDE.md запрещает мёртвый код):

```python
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)
        return next_frames, {}
```

В `PredRNNv2_Model.forward` — вернуть взвешенный декаплинг, тоже без внутреннего MSE:

```python
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)
        return next_frames, {"decouple": self.decouple_beta * decouple_loss}
```

Удалить `self.mse_criterion = nn.MSELoss()` из обоих `__init__` (больше не используется). Обновить docstring `forward` у обеих моделей: возвращается `tuple[torch.Tensor, dict[str, torch.Tensor]]`, второй элемент — уже взвешенные аукс-лоссы; задачу считает стратегия.

- [ ] **Step 4: Прибавить аукс-лоссы в стратегии**

`training_strategies/predrnn.py`:

```python
def _predrnn_forward(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run a full ``(x, y)`` concat-permute-forward-permute pipeline.

    Returns:
        Tuple ``(inp, y_hat, aux_losses)``; ``inp``/``y_hat`` are
        ``(B, T, C, H, W)``, ``aux_losses`` maps a name to an already-weighted
        scalar penalty the caller must add to the task loss.
    """
    inp = torch.cat([x, y], dim=1)
    inp = inp.permute(0, 1, 3, 4, 2).contiguous()
    mask = torch.zeros((1, y.shape[1], 1, 1, 1), device=x.device, dtype=x.dtype)
    y_hat_perm, aux_losses = model(inp, mask)
    y_hat = y_hat_perm.permute(0, 1, 4, 2, 3)
    inp = inp.permute(0, 1, 4, 2, 3)
    return inp, y_hat, aux_losses
```

`train_step` — прибавляет и логирует каждый член отдельно (иначе в Comet не видно, сколько весит физика против задачи):

```python
    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        inp, y_hat, aux_losses = _predrnn_forward(model, x, y)
        task_loss = self.loss(inp[:, 1:, ...], y_hat)
        loss = task_loss
        for aux_loss in aux_losses.values():
            loss = loss + aux_loss
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=loss.device)
        metrics = {"loss": loss, "task_loss": task_loss.detach(), "lr": lr}
        for name, aux_loss in aux_losses.items():
            metrics[f"aux_{name}"] = aux_loss.detach()
        return metrics
```

`val_step` — распаковать три значения: `_, y_hat_full, _ = _predrnn_forward(model, x, y)`.

- [ ] **Step 5: Прогнать тесты**

Run: `python -m pytest tests/test_predrnn_aux_loss.py tests/test_model_registry.py -v`
Expected: PASS (4 теста аукс-контракта + реестр).

- [ ] **Step 6: Линт и коммит**

```bash
ruff check . && ruff format --check .
git add Models/PredRNN.py training_strategies/predrnn.py tests/test_predrnn_aux_loss.py
git commit -m "fix(PredRNNv2): wire the memory-decoupling loss into the optimizer

PredRNNStep выбрасывал второй элемент forward, поэтому decouple_beta ни
на что не влиял, adapter не получал градиента, а v2 де-факто обучался как
v1. Модели семейства теперь возвращают dict уже взвешенных аукс-лоссов,
стратегия их прибавляет. Внутренний MSE моделей удалён: он конфликтовал
с MAE-лоссом стратегии и всё равно не читался."
```

---

### Task 3: `PI_PredRNNv2_Model` — физика внутри авторегрессивного цикла

**Files:**
- Modify: `Models/PredRNN.py` (новый класс)
- Modify: `Models/__init__.py` (регистрация)
- Create: `tests/test_pi_predrnnv2.py`

**Interfaces:**
- Consumes: `PhysicsResidualMixin` (Task 1) — `init_physics_residual`, `_apply_physics_residual`, `physics_residual_aux_loss`; аукс-контракт `forward -> (next_frames, dict)` (Task 2).
- Produces: класс `PI_PredRNNv2_Model`, зарегистрированный в реестре под ключом `"PI-PredRNNv2"`; принимает те же `physics_*`/`diabatic_*` параметры, что и `IAM4VP`, плюс `num_layers`, `num_hidden`, `configs`.

**Проектные решения (обосновать в docstring класса):**
1. **Точка врезки — внутри цикла, после `x_gen = self.conv_last(...)`.** Там `net` — вход шага, `x_gen` — выход шага, оба в пространстве состояния. Это ровно семантика `_apply_physics_residual(y_nn, prev_state)`.
2. **`patch_size` обязан быть 1.** Физядро требует `(B, 69, 32, 64)`; при `patch_size > 1` каналы свёрнуты с пространством, и `net` перестаёт быть состоянием. Проверять в `__init__` и падать (LBYL).
3. **`prev_state` детачится.** Паритет с PI-IAM4VP (`iterative_manual.py:145` кладёт в историю `prediction.detach()`): физприор не пробрасывает градиент сквозь предыдущие шаги. Без этого 23 шага физики попадают в один граф.
4. **Аукс-лосс усредняется по шагам.** `_apply_physics_residual` перезаписывает `_last_residual_aux_loss` на каждом вызове; копим список и усредняем — иначе в лосс попадёт штраф только последнего шага.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_pi_predrnnv2.py
"""Контракт PI-PredRNNv2: физика подключена, отключаемая, и не ломает v2."""

from __future__ import annotations

import unittest

import torch

from Models import get_model
from Models.PredRNN import PI_PredRNNv2_Model, PredRNNv2_Model
from utils.physics_residual import PhysicsResidualMixin

BASE_CONFIGS = {
    "in_shape": (12, 69, 32, 64),
    "patch_size": 1,
    "filter_size": 3,
    "stride": 1,
    "layer_norm": True,
    "pre_seq_length": 2,
    "aft_seq_length": 2,
    "reverse_scheduled_sampling": 0,
    "decouple_beta": 0.1,
}

PHYSICS_P3 = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_feature_mode": "tendency",
    "physics_residual_hybrid_steps": 1,
    "physics_residual_hybrid_mode": "stable_physical_v2",
    "physics_w_diagnostic": "mass_consistent",
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
}

PHYSICS_P0 = {**PHYSICS_P3, "physics_feature_mode": "no_physics"}


def _build(physics: dict, **overrides):
    torch.manual_seed(0)
    configs = {**BASE_CONFIGS, **overrides.pop("configs", {})}
    model = PI_PredRNNv2_Model(
        num_layers=2, num_hidden=(16, 16), configs=configs, **physics, **overrides
    )
    model.eval()
    model.set_physics_normalization(torch.zeros(69), torch.ones(69))
    return model


def _inputs(total_length: int = 4):
    torch.manual_seed(1)
    frames = torch.randn(1, total_length, 32, 64, 69)
    mask = torch.zeros(1, 2, 1, 1, 1)
    return frames, mask


class TestRegistry(unittest.TestCase):
    def test_registered_under_pi_predrnnv2(self) -> None:
        self.assertIs(get_model("PI-PredRNNv2"), PI_PredRNNv2_Model)


class TestContract(unittest.TestCase):
    def test_inherits_mixin_and_v2(self) -> None:
        self.assertTrue(issubclass(PI_PredRNNv2_Model, PhysicsResidualMixin))
        self.assertTrue(issubclass(PI_PredRNNv2_Model, PredRNNv2_Model))

    def test_patch_size_other_than_one_is_rejected(self) -> None:
        """Физядро ест состояние (B,69,32,64); патчинг смешал бы каналы с пространством."""
        with self.assertRaises(ValueError) as ctx:
            _build(PHYSICS_P3, configs={"patch_size": 2})
        self.assertIn("patch_size", str(ctx.exception))

    def test_forward_shape_and_aux_keys(self) -> None:
        model = _build(PHYSICS_P3)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (1, 3, 32, 64, 69))
        self.assertEqual(set(aux), {"decouple", "physics_aux"})

    def test_zero_init_corrector_is_identity_at_step_zero(self) -> None:
        """physics_residual_zero_init=True => коррекция стартует нулевой."""
        model = _build(PHYSICS_P3)
        torch.manual_seed(0)
        baseline = PredRNNv2_Model(2, (16, 16), BASE_CONFIGS)
        baseline.eval()
        frames, mask = _inputs()
        with torch.no_grad():
            pi_out, _ = model(frames, mask)
            base_out, _ = baseline(frames, mask)
        torch.testing.assert_close(pi_out, base_out, rtol=1e-5, atol=1e-6)

    def test_no_physics_arm_yields_zero_delta_phys(self) -> None:
        """P0: голова той же ёмкости, но физпризнак тождественно нулевой."""
        model = _build(PHYSICS_P0)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (1, 3, 32, 64, 69))
        self.assertIn("physics_aux", aux)

    def test_physics_changes_output_once_corrector_is_trained(self) -> None:
        """Не-нулевая голова => физика реально двигает предсказание."""
        model = _build(PHYSICS_P3, physics_residual_zero_init=False)
        torch.manual_seed(0)
        baseline = PredRNNv2_Model(2, (16, 16), BASE_CONFIGS)
        baseline.eval()
        frames, mask = _inputs()
        with torch.no_grad():
            pi_out, _ = model(frames, mask)
            base_out, _ = baseline(frames, mask)
        self.assertGreater((pi_out - base_out).abs().max().item(), 0.0)

    def test_physics_aux_loss_is_finite_and_backpropagates(self) -> None:
        model = _build(PHYSICS_P3, physics_residual_zero_init=False)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        loss = next_frames.abs().mean() + aux["decouple"] + aux["physics_aux"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [
            p.grad.abs().sum().item()
            for p in model.physics_residual_corrector.parameters()
            if p.grad is not None
        ]
        self.assertTrue(grads and sum(grads) > 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Прогнать — упадёт на импорте `PI_PredRNNv2_Model`**

Run: `python -m pytest tests/test_pi_predrnnv2.py -v`
Expected: FAIL — `ImportError: cannot import name 'PI_PredRNNv2_Model'`.

- [ ] **Step 3: Реализовать класс в `Models/PredRNN.py`**

```python
class PI_PredRNNv2_Model(PhysicsResidualMixin, PredRNNv2_Model):
    """PredRNN-V2 с физическим residual-корректором (exp20).

    Тот же физический путь, что у PI-IAM4VP: на каждом шаге авторегрессии
    выход ячейки ``x_gen`` корректируется головой, которой на вход подаётся
    физическая невязка ``delta_phys = physics(prev) - prev``. Коррекция
    врезается ВНУТРЬ цикла (у PI-IAM4VP цикл живёт в стратегии, здесь — в
    ``forward``), поэтому ``prev_state`` детачится вручную: без этого 23 шага
    физядра попали бы в один граф.

    Требует ``patch_size == 1``: физядро работает в пространстве состояния
    ``(B, 69, 32, 64)``, а патчинг свернул бы пространство в каналы.

    Args:
        num_layers: число слоёв :class:`SpatioTemporalLSTMCellv2`.
        num_hidden: скрытые каналы по слоям, длина ``num_layers``.
        configs: геометрия PredRNN (см. :func:`_parse_configs`).
        **physics_kwargs: параметры физического пути, см.
            :meth:`utils.physics_residual.PhysicsResidualMixin.init_physics_residual`.
    """

    def __init__(self, num_layers: int, num_hidden, configs: dict, **physics_kwargs) -> None:
        super().__init__(num_layers, num_hidden, configs)
        if self.patch_size != 1:
            raise ValueError(
                "PI-PredRNNv2 requires patch_size=1: the physics kernel consumes the "
                f"raw state (B, 69, H, W), got patch_size={self.patch_size} which folds "
                "space into channels."
            )
        _, img_channel, img_height, img_width = tuple(configs["in_shape"])
        self.init_physics_residual(
            C_data=img_channel,
            H_data=img_height,
            W_data=img_width,
            downscaling_factor_all=4,
            **physics_kwargs,
        )

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Roll out predictions, correcting every step with the physics head.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip.
            mask_true: scheduled-sampling mask ``(1, T_pred, 1, 1, 1)``.

        Returns:
            Tuple ``(next_frames, aux_losses)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``aux_losses`` carries the weighted
            ``"decouple"`` penalty and the ``"physics_aux"`` L1 penalty of the
            residual correction, averaged over rollout steps.
        """
        patches = _reshape_patch(frames_tensor, self.patch_size)
        frames = patches.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        device = frames.device
        batch, total_length = frames.shape[0], frames.shape[1]
        height, width = frames.shape[3], frames.shape[4]

        next_frames = []
        h_t, c_t, delta_c_list, delta_m_list = [], [], [], []
        decouple_loss = []
        physics_aux = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)

        x_gen = frames[:, 0]
        for t in range(total_length - 1):
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_length:
                    net = frames[:, t]
                else:
                    net = (
                        mask_true[:, t - self.input_length] * frames[:, t]
                        + (1 - mask_true[:, t - self.input_length]) * x_gen
                    )

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
            )
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
                )

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            x_gen = self._apply_physics_residual(x_gen, net.detach())
            step_aux = self.physics_residual_aux_loss()
            if step_aux is not None:
                physics_aux.append(step_aux)
            next_frames.append(x_gen)
            for i in range(self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))
                    )
                )

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)

        aux_losses = {"decouple": self.decouple_beta * decouple_loss}
        if physics_aux:
            aux_losses["physics_aux"] = torch.mean(torch.stack(physics_aux, dim=0))
        return next_frames, aux_losses
```

Импорты в шапке `Models/PredRNN.py` дополнить: `from utils.physics_residual import PhysicsResidualMixin`.

- [ ] **Step 4: Зарегистрировать в `Models/__init__.py`**

Рядом с `_build_predrnn_v2` добавить билдер, приводящий `in_shape`/`num_hidden` к кортежам (как соседние), и зарегистрировать под `"PI-PredRNNv2"`. Обновить ожидаемый список в `tests/test_model_registry.py:39-44`:

```python
        self.assertEqual(
            registered_model_names(),
            ["IAM4VP", "PI-IAM4VP", "PI-PredRNNv2", "PredRNN", "PredRNNv2", "SimVP"],
        )
```

- [ ] **Step 5: Прогнать весь набор**

Run: `python -m pytest tests/ -v`
Expected: PASS — все тесты, включая 8 новых в `test_pi_predrnnv2.py` и обновлённый реестр.

- [ ] **Step 6: Линт и коммит**

```bash
ruff check . && ruff format --check .
git add Models/PredRNN.py Models/__init__.py tests/test_pi_predrnnv2.py tests/test_model_registry.py
git commit -m "feat(exp20): PI-PredRNNv2 — physics residual corrector inside the rollout"
```

---

### Task 4: Конфиги лестницы exp20

**Files:**
- Create: `configs/exp20/exp20_p{0,1,3a,3,4,5}_*_s0.yaml` (короткая волна)
- Create: `configs/exp20_long/exp20_p{0,1,3a,3,4,5}_*_t12.yaml` (long-волна)

**Interfaces:**
- Consumes: ключ модели `PI-PredRNNv2` (Task 3); стратегия `mutiout_predrnn` (Task 2).

Базой берётся `configs/predrnnv2_usa_v4.yaml` (данные/кроп/горизонты), физический блок — построчно из соответствующего арма `configs/abl16_long/`. Так межархитектурное сравнение остаётся честным: физика идентична, различается только носитель.

- [ ] **Step 1: Собрать конфиг P0 (контроль) короткой волны**

`configs/exp20/exp20_p0_no_physics_s0.yaml`. Отличия от `predrnnv2_usa_v4.yaml`:

```yaml
experiment:
  name: exp20-p0-no-physics-s0
model:
  type: PI-PredRNNv2
  params:
    num_layers: 4
    num_hidden: [128, 128, 128, 128]
    configs:
      in_shape: [12, 69, 32, 64]
      patch_size: 1
      filter_size: 3
      stride: 1
      layer_norm: true
      pre_seq_length: 12
      aft_seq_length: 12
      reverse_scheduled_sampling: 0
      decouple_beta: 0.1
    use_physics_residual_corrector: true
    physics_residual_hidden_channels: 128
    physics_residual_apply_to: upper_air_only
    physics_residual_zero_init: true
    physics_residual_lambda_l1: 0.0001
    physics_feature_mode: no_physics
    physics_residual_shuffle: none
    physics_lat_start_deg: 18.28125
    physics_dlat_deg: 5.625
    physics_dlon_deg: 5.625
training:
  seed: 0
  litmodel: mutiout_predrnn
  lr: 0.0001
  max_epoch: 40
  loss_type: MAE
  val_every_n_epochs: 2
trainer:
  static_graph: false
  find_unused_parameters: true
```

(Остальные секции — `data`, `logging`, warmup, patience — копируются из `predrnnv2_usa_v4.yaml` дословно; `data.sample_stride: 12` и `batch_size` подбираются в Task 5 по итогам смоука.)

- [ ] **Step 2: Собрать остальные 5 арм короткой волны**

Физические блоки копируются построчно из `configs/abl16_long/`:
- **P1** ← `abl16_r1_legacy_hybrid_t12.yaml`: `physics_feature_mode: tendency`, `physics_residual_hybrid_mode: legacy_normalized`, легаси-геометрия (`physics_lat_start_deg: -70.0`, `physics_dlat_deg: 20.0`, `physics_dlon_deg: 22.5`), `physics_t_t_formulation: legacy_paper`, `physics_use_universal_R: true`, `physics_coriolis_formulation: beta_plane`, `physics_tendency_limiter: scale_diff`, `physics_tendency_on_latent: false`, `use_diabatic_term: false`.
- **P3a** ← `abl16_r3a_no_diabatic_t12.yaml`: `stable_physical_v2`, `physics_w_diagnostic: mass_consistent`, `use_diabatic_term: false`.
- **P3** ← `abl16_r3_a2_exp13_t12.yaml`: то же + `use_diabatic_term: true`, `diabatic_apply_to: t_and_q`, `diabatic_hidden_channels: 64`, `diabatic_lambda_l1: 0.0001`, `diabatic_constants_path`, `diabatic_cut: [75, 107, 164, 228]`.
- **P4** ← `abl16_r4_exp14_t12.yaml`: P3 + `physics_advection_form: advective`, `physics_metric_terms: true`, `physics_spherical_divergence: true`, `physics_rayleigh_friction: true`, `physics_vertical_scheme: lagrange3`.
- **P5** ← `abl16_r5_exp15_t12.yaml`: P4 + `physics_omega_free: [t, q, u, v]`, `physics_latent_heating_coupling: true`, `physics_clim_sources_path`.

Проверка на глаз недостаточна — сверить diff'ом:

```bash
for arm in p1:r1_legacy_hybrid p3a:r3a_no_diabatic p3:r3_a2_exp13 p4:r4_exp14 p5:r5_exp15; do
  p="${arm%%:*}"; r="${arm##*:}"
  echo "=== $p vs $r (должны совпасть все physics_*/diabatic_* строки) ==="
  diff <(grep -E '^\s+(physics_|diabatic_|use_diabatic)' configs/exp20/exp20_${p}_*_s0.yaml | sed 's/^\s*//') \
       <(grep -E '^\s+(physics_|diabatic_|use_diabatic)' configs/abl16_long/abl16_${r}_t12.yaml | sed 's/^\s*//')
done
```
Ожидаемо: единственные различия — отсутствие в exp20 ключей, которых у PredRNNv2 нет по определению (`T_data`, `hid_S` и т.п.), и `physics_clim_sources_path`, если путь на кластере иной.

- [ ] **Step 3: Прогнать конфиги через реальный билдер модели**

Конфиг может быть синтаксически валиден и при этом не собираться. Дешёвая проверка до кластера:

```bash
python - <<'PY'
import yaml
from Models import get_model

for arm in ["p0_no_physics", "p1_legacy_hybrid", "p3a_no_diabatic",
            "p3_a2_exp13", "p4_exp14", "p5_exp15"]:
    import glob
    path = glob.glob(f"configs/exp20/exp20_{arm}_s0.yaml")[0]
    cfg = yaml.safe_load(open(path))
    params = cfg["model"]["params"]
    # diabatic_constants_path указывает на кластерный путь — на ноуте его нет
    params.pop("diabatic_constants_path", None)
    params.pop("physics_clim_sources_path", None)
    if params.pop("use_diabatic_term", False):
        print(f"{arm}: diabatic head skipped locally (needs cluster constants)")
    model = get_model(cfg["model"]["type"])(**params)
    n = sum(p.numel() for p in model.parameters())
    print(f"{arm}: OK, {n/1e6:.1f}M params")
PY
```
Ожидаемо: 6 строк `OK`, число параметров у всех арм близкое (различие только на голову/диабатик).

- [ ] **Step 4: Коммит**

```bash
git add configs/exp20/
git commit -m "feat(exp20): ladder configs for the PI-PredRNNv2 arms"
```

---

### Task 5: Смоук на кластере, калибровка бюджета, запуск короткой волны

Задача существует, потому что бюджет считать по exp16 нельзя: PredRNNv2 — рекуррентная модель, её шаг стоит иначе, чем у IAM4VP, а физика в ней зовётся 23 раза за forward (против 12 у PI-IAM4VP, где цикл живёт в стратегии). Цену надо **измерить**, а не оценить.

**Files:**
- Create: `sh_files/train_pi_predrnnv2.sh` (по образцу `train_pi_iam4vp_residual_usa_v4_2gpu.sh`: конфиг первым аргументом)

- [ ] **Step 1: Скрипт запуска, принимающий конфиг аргументом**

Скопировать `sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh`, поменять `--job-name` и хвост на:
```bash
weatherpred__config="${1:-exp20/exp20_p0_no_physics_s0}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "${weatherpred__launcher}" "${weatherpred__config}" "$@"
```
Остальное (стейджинг memmap на node-local, trap, экспорт `MEMMAP_PATH_OVERRIDE`) не трогать.

- [ ] **Step 2: Завести worktree exp20 на кластере и положить в него `.env`**

```bash
ssh cluster '
  git -C ~/WeatherPredictions fetch --prune origin
  git -C ~/WeatherPredictions worktree add --detach ~/wt_exp20 origin/exp20_pi_predrnnv2 || \
    git -C ~/wt_exp20 fetch origin && git -C ~/wt_exp20 checkout --detach origin/exp20_pi_predrnnv2
  cp ~/wt_fix_v2/.env ~/wt_exp20/.env
  ls -l ~/wt_exp20/.env && git -C ~/wt_exp20 log --oneline -1'
```
Ожидаемо: `.env` (97 B) на месте, HEAD = последний коммит ветки. **Без `.env` ран падает на Comet-ключе.**

- [ ] **Step 3: Смоук — 2 арма (P0 и P5), по 30 минут, с ограничением шагов**

P0 — самый дешёвый, P5 — самый дорогой; они ограничивают вилку.

```bash
ssh cluster '
MM=$HOME/era5_memmap/predformer_usa_2000_2004.dat
EXPORT=ALL,WEATHERPRED_USA_MEMMAP=$MM,WEATHERPRED_CONDA_ENV_NAME=pi-iamvp,WEATHERPRED_CHECKPOINT_BASE=$HOME/exp20_smoke_ckpt,BENCH_MAX_STEPS=20
cd ~/wt_exp20
for c in p0_no_physics p5_exp15; do
  sbatch --parsable -A proj_1715 -J exp20-smoke-$c \
    --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 \
    --constraint=type_e\|type_f\|type_h --export=$EXPORT \
    sh_files/train_pi_predrnnv2.sh configs/exp20/exp20_${c}_s0.yaml
done
squeue -u fa.buzaev -o "%.10i %.28j %.9T %R"'
```

- [ ] **Step 4: Прочитать смоук — три числа решают всё**

```bash
ssh cluster 'grep -E "steps/s|it/s|epoch|CUDA out of memory|Error|Traceback" ~/wt_exp20/logs/slurm-exp20-smoke-*.out | tail -40'
```
Из логов снять: **(а)** шагов/с у P0 и у P5 (отсюда цена эпохи и налог физики), **(б)** пиковую память (влезает ли `batch_size`), **(в)** отсутствие NaN в `physics_residual_nonfinite_ratio`.

Решающее правило:
- если P5 медленнее P0 более чем в 4× — уменьшить `physics_residual_hybrid_steps` с 3 до 1 во **всех** физ-армах (иначе long-волна не влезет в 2 суток), и записать это в README как отклонение от протокола exp16;
- если OOM — снизить `batch_size` до 4 во всех армах одинаково (не только в упавшем: сравнение требует идентичного протокола);
- если `nonfinite_ratio > 0` — не запускать волну, идти в systematic-debugging.

- [ ] **Step 5: Запустить короткую волну — 6 арм**

`max_epoch` и `--time` проставить по замеру шага 4: walltime = (цена эпохи P5) × max_epoch × 1.3.

```bash
ssh cluster '
MM=$HOME/era5_memmap/predformer_usa_2000_2004.dat
EXPORT=ALL,WEATHERPRED_USA_MEMMAP=$MM,WEATHERPRED_CONDA_ENV_NAME=pi-iamvp,WEATHERPRED_CHECKPOINT_BASE=$HOME/exp20_ckpt
cd ~/wt_exp20
for c in p0_no_physics p1_legacy_hybrid p3a_no_diabatic p3_a2_exp13 p4_exp14 p5_exp15; do
  sbatch --parsable -A proj_1715 -J exp20-$c \
    --gres=gpu:1 --cpus-per-task=8 --time=<T_из_шага_4> \
    --constraint=type_e\|type_f\|type_h --export=$EXPORT \
    sh_files/train_pi_predrnnv2.sh configs/exp20/exp20_${c}_s0.yaml
done
squeue -u fa.buzaev -o "%.10i %.28j %.9T %R"'
```

- [ ] **Step 6: Дождаться и проверить, что все 6 дошли до общей эпохи**

Полить логи; зафиксировать максимальную эпоху, достигнутую **всеми** армами — сравнение пойдёт по ней (§11.3 exp16: ранжировать на разных эпохах нельзя).

---

### Task 6: Сбор результатов и отчёт

**Files:**
- Create: `docs/experiments/20_pi_predrnnv2_ladder/README.md`
- Create: `docs/experiments/20_pi_predrnnv2_ladder/collect_metrics.py`
- Create: `docs/experiments/20_pi_predrnnv2_ladder/make_figures.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Переиспользовать сборщики exp16**

`docs/experiments/16_model_ablation_ladder/collect_metrics.py` и `make_figures.py` тянут метрики из Comet по префиксу имени рана и сворачивают по-уровневые RMSE в базовые переменные. Прочитать их и, если различие только в префиксе имён и списке арм, — параметризовать, а не копировать (DRY: CLAUDE.md §1). Если структура метрик совпадает, оба скрипта exp20 сводятся к тонким обёрткам с другим списком арм.

- [ ] **Step 2: Снять таблицу дельт к P0 на общей эпохе + по окну эпох**

Обязательно **обе**: дельта одной эпохи по z шумит на ±5–8 п.п. (§8.5-а exp16), и именно на этом обжёгся exp18.

- [ ] **Step 3: Написать README отчёта**

Структура — по образцу exp16: вопрос, лестница, протокол, отклонения, результаты, выводы, ограничения. Обязательно зафиксировать:
- **главный вопрос exp20:** переносится ли эффект физики (−2 % у PI-IAM4VP) на рекуррентную архитектуру, или он был свойством трансформерного носителя;
- **что P2a/P2 невоспроизводимы** (pre-13 дерево, PI-PredRNNv2 там не существовало);
- **что PredRNNv2 до exp20 обучался как v1** (декаплинг-лосс не доходил до оптимизатора) — то есть все прежние «бэйзлайны PredRNNv2», если бы они были запущены, измеряли бы не то, что заявлено;
- отклонения от протокола exp16 (шаги физики, batch, эпохи), если Task 5 их потребовал.

- [ ] **Step 4: Обновить CHANGELOG и закрыть ветку**

Секции `Added` (PI-PredRNNv2, миксин, exp20), `Fixed` (декаплинг-лосс PredRNNv2), `Changed` (контракт forward у семейства PredRNN).

---

## Открытые вопросы (решаются данными, не планом)

1. **Цена физики в рекуррентном носителе.** Физядро зовётся 23 раза за forward против 12 у PI-IAM4VP. Если налог окажется ≫3×, честный equal-compute вывод потребует отдельной оговорки — или урезания `physics_residual_hybrid_steps` (тогда лестница сравнима внутри exp20, но не с exp16 поштучно).
2. **Врезка на всех 23 шагах против только на прогнозных 12.** Выбраны все 23 (единообразие: коррекция применяется всюду, где модель что-то предсказывает). Альтернатива — корректировать только шаги ≥ `input_length`; это ближе к PI-IAM4VP, где физика зовётся только на прогнозных шагах. Если смоук покажет, что цена неприемлема, это первый рычаг.
3. **Декаплинг × физика.** Оба аукс-члена теперь живые; их взаимодействие никем не измерено. Арм «v2 с мёртвым декаплингом» в лестницу не вошёл (пользователь выбрал 6 арм) — при необходимости добавляется как `decouple_beta: 0.0`.
