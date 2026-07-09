# Exp 16 Model Ablation Ladder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Обучить лестницу из 8 армов PI-IAM4VP (R0→R5 + A1-pre13 + якорь) на одинаковых эпохах/данных/сиде и измерить вклад каждой ступени эволюции физического ядра (дизайн: `docs/experiments/16_model_ablation_ladder/README.md`).

**Architecture:** Ступени R0/R1/R3 — существующие конфиги + сид/эпохи; R2a/R2 — те же конфиги на worktree до-exp13 коммита; R4/R5 — порт принятых exp14/15-термов как opt-in флагов в `utils/physics_hybrid.PDE_kernel` (зеркало `utils/physics.PurePDEKernel`, дефолты бит-в-бит). Запуск — существующий лончер `sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh <config_path>`.

**Tech Stack:** PyTorch 2.6 (env `pi-iamvp` на cHARISMa), SLURM (`-A proj_1715`), Comet, pytest.

## Global Constraints

- Никаких `try/except`; импорты только в начале файла (CLAUDE.md).
- Дефолты физики **бит-в-бит**: все новые параметры opt-in, выключены по умолчанию; регресс-пин обязателен.
- Train-скрипты (`train.py`, `trainer.py`, `sh_files/launch_train.sh`, `sh_files/train_*.sh`) **не менять**.
- Кластер: sbatch всегда `-A proj_1715`; env `WEATHERPRED_CONDA_ENV_NAME=pi-iamvp`; memmap `~/era5_memmap/predformer_usa_2000_2004.dat` (репак job 4170311); job `aj-exp015` не трогать.
- Все конфиги лестницы: `training.seed: 0`, `training.max_epoch: 30`, `trainer.val_every_n_epochs: 3`, имена `abl16-<арм>-s<seed>`.
- `ruff check .` и `ruff format .` перед каждым коммитом; коммиты conventional commits.
- Ветка: `fix_inline_v2`.

---

### Task 1: Конфиги волны 1 (R0, R1, R3)

**Files:**
- Create: `configs/ablation16/abl16_r0_no_physics_s0.yaml`
- Create: `configs/ablation16/abl16_r1_legacy_hybrid_s0.yaml`
- Create: `configs/ablation16/abl16_r3_a2_exp13_s0.yaml`

**Interfaces:**
- Produces: три YAML, каждый = копия базового конфига с 4 переопределениями. Task 2 сабмитит их по полному пути `configs/ablation16/<name>.yaml`.

- [ ] **Step 1: Сгенерировать три конфига из базовых**

```bash
cd /Users/buzaev-fa/WeatherPredictions
mkdir -p configs/ablation16
python3 - <<'EOF'
import re, pathlib

BASE = {
    "abl16_r0_no_physics_s0":   ("configs/pi_iam4vp_residual_no_physics_usa_v4.yaml",   "abl16-r0-no-physics-s0"),
    "abl16_r1_legacy_hybrid_s0":("configs/pi_iam4vp_residual_legacy_hybrid_usa_v4.yaml","abl16-r1-legacy-hybrid-s0"),
    "abl16_r3_a2_exp13_s0":     ("configs/pi_iam4vp_residual_diabatic_usa_v4.yaml",     "abl16-r3-a2-exp13-s0"),
}
for stem, (src, run_name) in BASE.items():
    text = pathlib.Path(src).read_text()
    text = re.sub(r"(?m)^(  name: ).*$", rf"\g<1>{run_name}", text, count=1)
    text = re.sub(r"(?m)^  max_epoch: \d+$", "  max_epoch: 30", text, count=1)
    assert "max_epoch: 30" in text and run_name in text
    # seed — новый ключ в training: (после строки lr)
    text = re.sub(r"(?m)^(training:\n)", r"\g<1>  seed: 0\n", text, count=1)
    assert "seed: 0" in text
    pathlib.Path(f"configs/ablation16/{stem}.yaml").write_text(text)
    print("written", stem)
EOF
```

Expected: `written abl16_r0_no_physics_s0` (и ещё две строки).

- [ ] **Step 2: Верифицировать переопределения**

```bash
for f in configs/ablation16/*.yaml; do
  echo "== $f"; grep -E "^  (name|seed|max_epoch):" "$f"
  grep -E "val_every_n_epochs" "$f"
done
```

Expected: в каждом — `name: abl16-...`, `seed: 0`, `max_epoch: 30`, `val_every_n_epochs: 3`.

- [ ] **Step 3: Смоук load_config**

```bash
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python -c "
import sys; sys.path.insert(0, '.')
from train import load_config
for s in ['abl16_r0_no_physics_s0','abl16_r1_legacy_hybrid_s0','abl16_r3_a2_exp13_s0']:
    c = load_config(f'configs/ablation16/{s}.yaml')
    print(s, c['experiment']['name'], c['training']['seed'], c['training']['max_epoch'])
"
```

Expected: три строки с именами, `0`, `30`.

- [ ] **Step 4: Commit**

```bash
git add configs/ablation16/
git commit -m "feat(configs): exp 16 ablation ladder wave-1 configs (R0/R1/R3, seed 0, 30 epochs)"
```

---

### Task 2: Волна 1 на кластере (R0, R1, R2a, R2, R3)

Не зависит от Tasks 3–5 — стартует сразу после Task 1. R2a/R2 бегут из worktree
до-exp13 коммита `a827d32` (= `051e350^`: v2 уже есть, знаковых фиксов нет).

**Files:**
- Cluster create: `~/wt_prefix13` (worktree @ a827d32), `~/wt_prefix13/configs/ablation16/{abl16_r2a_a1_pre13_s0,abl16_r2_a2_pre13_s0}.yaml`

**Interfaces:**
- Consumes: конфиги Task 1 (запушены), memmap `predformer_usa_2000_2004.dat` (job 4170311 COMPLETED).
- Produces: 5 SLURM-джоб `abl16-*-s0`; Comet-раны с теми же именами; чекпоинты `$WEATHERPRED_CHECKPOINT_BASE/<run>/...`.

- [ ] **Step 1: Push + обновить кластерный worktree**

```bash
cd /Users/buzaev-fa/WeatherPredictions && git push origin fix_inline_v2
ssh cluster 'cd ~/WeatherPredictions && git fetch origin && git -C ~/wt_fix_v2 checkout --detach origin/fix_inline_v2 && git -C ~/wt_fix_v2 log --oneline -1'
```

Expected: HEAD worktree = последний коммит с `configs/ablation16/`.

- [ ] **Step 2: Проверить memmap 2000–2004**

```bash
ssh cluster 'ls -lh ~/era5_memmap/predformer_usa_2000_2004.dat ~/era5_memmap/predformer_usa_2000_2004.meta.json'
```

Expected: `.dat` ~23–24G + meta. Если нет — дождаться job 4170311 (`sacct -j 4170311`).

- [ ] **Step 3: Worktree до-exp13 + конфиги R2a/R2 в нём**

```bash
ssh cluster '
git -C ~/WeatherPredictions worktree add --detach ~/wt_prefix13 a827d32 2>/dev/null || git -C ~/wt_prefix13 log --oneline -1
mkdir -p ~/wt_prefix13/configs/ablation16
python3 - <<EOF
import re, pathlib
BASE = {
    "abl16_r2a_a1_pre13_s0": ("configs/pi_iam4vp_residual_massconsistent_usa_v4.yaml", "abl16-r2a-a1-pre13-s0"),
    "abl16_r2_a2_pre13_s0":  ("configs/pi_iam4vp_residual_diabatic_usa_v4.yaml",      "abl16-r2-a2-pre13-s0"),
}
root = pathlib.Path.home() / "wt_prefix13"
for stem, (src, run_name) in BASE.items():
    text = (root / src).read_text()
    text = re.sub(r"(?m)^(  name: ).*$", rf"\g<1>{run_name}", text, count=1)
    text = re.sub(r"(?m)^  max_epoch: \d+$", "  max_epoch: 30", text, count=1)
    text = re.sub(r"(?m)^(training:\n)", r"\g<1>  seed: 0\n", text, count=1)
    assert "seed: 0" in text and "max_epoch: 30" in text and run_name in text
    (root / "configs/ablation16" / f"{stem}.yaml").write_text(text)
    print("written", stem)
EOF'
```

Expected: `written abl16_r2a_a1_pre13_s0`, `written abl16_r2_a2_pre13_s0`. ВАЖНО: базой служат конфиги ИЗ worktree (параметры модели той эпохи), меняются только name/seed/max_epoch.

- [ ] **Step 4: Сабмит 5 ранов**

```bash
ssh cluster '
MM=$HOME/era5_memmap/predformer_usa_2000_2004.dat
EXPORT=ALL,WEATHERPRED_USA_MEMMAP=$MM,WEATHERPRED_CONDA_ENV_NAME=pi-iamvp,WEATHERPRED_CHECKPOINT_BASE=$HOME/abl16_ckpt
cd ~/wt_fix_v2
for c in abl16_r0_no_physics_s0 abl16_r1_legacy_hybrid_s0 abl16_r3_a2_exp13_s0; do
  sbatch --parsable -A proj_1715 -J $c --export=$EXPORT sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh configs/ablation16/$c.yaml
done
cd ~/wt_prefix13
for c in abl16_r2a_a1_pre13_s0 abl16_r2_a2_pre13_s0; do
  sbatch --parsable -A proj_1715 -J $c --export=$EXPORT sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh configs/ablation16/$c.yaml
done
squeue -u fa.buzaev -o "%.10i %.26j %.9T %R"'
```

Expected: 5 jobid + очередь с пятью `abl16-*`. Примечание: `sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh` на a827d32 уже содержит spool-fix (a827d32 и есть этот фикс).

- [ ] **Step 5: Монитор первых минут (OOM/ошибки конфига)**

```bash
ssh cluster 'sleep 900; for j in $(squeue -h -u fa.buzaev -n abl16-r0-no-physics-s0,abl16-r1-legacy-hybrid-s0,abl16-r3-a2-exp13-s0,abl16-r2a-a1-pre13-s0,abl16-r2-a2-pre13-s0 -o %i 2>/dev/null); do echo "== $j"; tail -3 ~/wt_fix_v2/logs/slurm-*-$j.out ~/wt_prefix13/logs/slurm-*-$j.out 2>/dev/null | tail -4; done'
```

Expected: у стартовавших — строки прогресса эпохи, без Traceback. FAILED → читать `.err`, чинить, пересабмитить.

---

### Task 3: Порт exp14-термов в `physics_hybrid.PDE_kernel` (opt-in)

Флаги: `advection_form` ('flux' дефолт | 'advective'), `metric_terms`,
`spherical_divergence`, `rayleigh_friction`, `vertical_scheme` ('stencil' | 'lagrange3').
Эталон формул: `utils/physics.py:1040-1090` (get_uv_dt), `:723-745` (буферы), `:783-792` (lagrange3).

**Files:**
- Modify: `utils/physics_hybrid.py` (PDE_kernel.__init__ ~строка 309; get_uv_dt ~455; get_w ~536; get_t_t/q — только `self._d_z`; PDE_block.__init__ ~700; HybridBlock.__init__ ~800)
- Test: `tests/test_physics_hybrid_exp16_port.py` (новый)

**Interfaces:**
- Consumes: `_lagrange3_dz_matrix(pressure_hpa)` — уже в `utils/physics.py` (импортировать оттуда), буфер `pressure` (гПа) и `pixel_z` module-level в physics_hybrid.
- Produces: kwargs `advection_form: str = "flux"`, `metric_terms: bool = False`, `spherical_divergence: bool = False`, `rayleigh_friction: bool = False`, `vertical_scheme: str = "stencil"` на PDE_kernel/PDE_block/HybridBlock (сквозной проброс, паттерн — как у существующего `w_diagnostic`). Task 5 включает их из конфига.

- [ ] **Step 1: Failing-тест (голден дефолтов + TypeError на новые kwargs)**

Создать `tests/test_physics_hybrid_exp16_port.py`:

```python
"""Пины порта exp14/15-термов в PDE_kernel (exp 16). Дефолты — бит-в-бит."""

import torch

from utils.physics_hybrid import HybridBlock, PDE_kernel


def _state(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    B, P, H, W = 2, 13, 8, 16
    z = 5.0e4 + 1.0e3 * torch.randn(B, P, H, W, generator=g)
    t = 260.0 + 15.0 * torch.randn(B, P, H, W, generator=g)
    q = (5.0e-3 + 2.0e-3 * torch.randn(B, P, H, W, generator=g)).clamp_min(1e-6)
    u = 10.0 * torch.randn(B, P, H, W, generator=g)
    v = 10.0 * torch.randn(B, P, H, W, generator=g)
    return torch.cat([z, t, q, u, v], dim=1)


def _kernel(**kw) -> PDE_kernel:
    torch.manual_seed(0)
    kernel = PDE_kernel(
        in_dim=65,
        variable_dim=13,
        physics_part_coef=0.5,
        w_diagnostic="mass_consistent",
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        grid_h=8,
        physical_passthrough=True,
        **kw,
    )
    return kernel.eval()


def test_defaults_bitexact_golden() -> None:
    """Все новые флаги выключены → physics_only_forward бит-в-бит с до-портовым."""
    kernel = _kernel()
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)


def test_new_kwargs_accepted() -> None:
    kernel = _kernel(
        advection_form="advective",
        metric_terms=True,
        spherical_divergence=True,
        rayleigh_friction=True,
        vertical_scheme="lagrange3",
    )
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    assert torch.isfinite(out).all()


def test_metric_terms_sign() -> None:
    """+u·v·tanφ/a в u_t: при u,v>0 в NH u_t(metric) > u_t(base)."""
    base, metric = _kernel(), _kernel(metric_terms=True)
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    u, v = u.abs() + 1.0, v.abs() + 1.0
    w = base.get_w(u, v)
    base.share_z_dxyz(z)
    metric.share_z_dxyz(z)
    ut_base, _ = base.get_uv_dt(u, v, w)
    ut_metric, _ = metric.get_uv_dt(u, v, w)
    assert (ut_metric - ut_base).min() > 0


def test_rayleigh_only_boundary_layer() -> None:
    """k_v = 0 при σ<0.7: верхние уровни не трогаются трением."""
    fric = _kernel(rayleigh_friction=True)
    assert fric.rayleigh_k[0, :5].abs().max() == 0
    assert fric.rayleigh_k[0, -1].item() > 0
```

- [ ] **Step 2: Сгенерировать голден НА ТЕКУЩЕМ коде (до правок)**

```bash
mkdir -p tests/goldens
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
import torch
from tests.test_physics_hybrid_exp16_port import _kernel, _state
kernel = _kernel()
with torch.no_grad():
    out = kernel.physics_only_forward(_state())
torch.save(out, "tests/goldens/exp16_kernel_default_out.pt")
print("golden saved", out.shape)
EOF
```

Expected: `golden saved torch.Size([2, 65, 8, 16])`.

- [ ] **Step 3: Прогнать тест — убедиться, что падает (нет kwargs)**

```bash
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python -m pytest tests/test_physics_hybrid_exp16_port.py -x -q
```

Expected: `test_defaults_bitexact_golden` PASS, `test_new_kwargs_accepted` FAIL с `TypeError: ... unexpected keyword argument 'advection_form'`.

- [ ] **Step 4: Реализация в PDE_kernel.__init__**

После валидации `tendency_limiter` (utils/physics_hybrid.py, в `__init__` PDE_kernel) добавить (сигнатура: `advection_form: str = "flux"`, `metric_terms: bool = False`, `spherical_divergence: bool = False`, `rayleigh_friction: bool = False`, `vertical_scheme: str = "stencil"` — вставить в параметры после `tendency_caps`):

```python
        if advection_form not in ("flux", "advective"):
            raise ValueError(f"Unknown advection_form {advection_form!r}")
        if vertical_scheme not in ("stencil", "lagrange3"):
            raise ValueError(f"Unknown vertical_scheme {vertical_scheme!r}")
        self.advection_form = advection_form
        self.metric_terms = metric_terms
        self.spherical_divergence = spherical_divergence
        self.rayleigh_friction = rayleigh_friction
        self.vertical_scheme = vertical_scheme
```

После блока `self.register_buffer("f_field", ...)` — буферы термов (зеркало `utils/physics.py:732-744`; широты `lat_rad` уже есть выше):

```python
        # exp 14 (порт из PurePDEKernel): кривизна сферы, трение Хелда–Суареса,
        # 3-точечная Лагранжева вертикальная производная. Дефолты выключены —
        # существующие армы бит-в-бит.
        if metric_terms or spherical_divergence:
            self.register_buffer(
                "tan_phi_over_a", (torch.tan(lat_rad) / radius).reshape(1, 1, grid_h, 1)
            )
        else:
            self.tan_phi_over_a = None
        if rayleigh_friction:
            sigma = pressure.reshape(1, -1, 1, 1) / 1000.0  # p₀ = 1000 гПа
            k_v = (1.0 / 86400.0) * torch.clamp((sigma - 0.7) / (1.0 - 0.7), min=0.0)
            self.register_buffer("rayleigh_k", k_v)
        else:
            self.rayleigh_k = None
        if vertical_scheme == "lagrange3":
            self.register_buffer("dz_lagrange", _lagrange3_dz_matrix(pressure))
        else:
            self.dz_lagrange = None
```

Импорт наверху файла: `from utils.physics import _lagrange3_dz_matrix` (проверить, что не создаёт цикл: `utils/physics.py` не импортирует `physics_hybrid` — ок).

- [ ] **Step 5: Метод `_d_z` + переключение всех вызовов**

Рядом с `_d_x`/`_d_y` (utils/physics_hybrid.py ~строка 440):

```python
    def _d_z(self, field: torch.Tensor) -> torch.Tensor:
        """Вертикальная производная d_z = −∂/∂p [1/гПа] по self.vertical_scheme."""
        if self.vertical_scheme == "lagrange3":
            return -torch.einsum("qp,bphw->bqhw", self.dz_lagrange, field)
        return d_z(field)
```

Заменить внутри классов PDE_kernel вызовы `d_z(...)` на `self._d_z(...)` в: `get_uv_dt` (`d_z(u * w)`, `d_z(v * w)`), `get_t_t` (`t_z = ...`), `get_q_dt` (`q_z = ...`), `share_z_dxyz` (`self.z_z = ...`). Module-level `d_z` остаётся для внешних потребителей. Проверить знак матрицы: в `PurePDEKernel` `dz_lagrange` строится под конвенцию d_z=−∂/∂p — сверить направление с `utils/physics.py` использованием (`_d_z` там) и запинить тестом ниже.

- [ ] **Step 6: get_uv_dt — advective-форма + термы**

Заменить тело `get_uv_dt` (utils/physics_hybrid.py:455-478) на:

```python
    def get_uv_dt(self, u, v, w):
        if self.advection_form == "advective":
            # Адвективная форма (exp 14, C_best): −(u·∂ₓ + v·∂ᵧ + w·∂_z)
            adv_u = u * self._d_x(u) + v * self._d_y(u) + w * self._d_z(u)
            adv_v = u * self._d_x(v) + v * self._d_y(v) + w * self._d_z(v)
        else:  # flux: консервативное представление (легаси, байт-в-байт)
            adv_u = (
                compute_spatial_derivative(u * u, self._d_x)
                + compute_spatial_derivative(u * v, self._d_y)
                + self._d_z(u * w)
            )
            adv_v = (
                compute_spatial_derivative(u * v, self._d_x)
                + compute_spatial_derivative(v * v, self._d_y)
                + self._d_z(v * w)
            )

        self.u_t = -adv_u + self.f_field * v - self.z_x
        self.v_t = -adv_v - self.f_field * u - self.z_y
        if self.metric_terms:
            # Кривизна сферы: +u·v·tanφ/a (u_t), −u²·tanφ/a (v_t) — H&H (2.19)–(2.20)
            self.u_t = self.u_t + self.tan_phi_over_a * u * v
            self.v_t = self.v_t - self.tan_phi_over_a * u * u
        if self.rayleigh_friction:
            self.u_t = self.u_t - self.rayleigh_k * u
            self.v_t = self.v_t - self.rayleigh_k * v
```

(хвост с eddy_viscosity — без изменений).

- [ ] **Step 7: get_w — spherical_divergence**

В `get_w` после `div = self.u_x + self.v_y` добавить:

```python
        if self.spherical_divergence:
            # ∇·V на сфере: ∂v/∂y − v·tanφ/a (меридиональный метрический член)
            div = div - v * self.tan_phi_over_a
```

- [ ] **Step 8: Проброс kwargs через PDE_block и HybridBlock**

По паттерну `w_diagnostic` (grep `w_diagnostic` по файлу — 6 мест): добавить те же 5 kwargs с теми же дефолтами в сигнатуры `PDE_block.__init__` и `HybridBlock.__init__` и передать вниз в конструкторы `PDE_kernel(...)` / `PDE_block(...)`.

- [ ] **Step 9: Тесты + голден + ruff**

```bash
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python -m pytest tests/test_physics_hybrid_exp16_port.py tests/test_physics_sign_conventions.py -q
ruff check . && ruff format --check .
```

Expected: все PASS (включая старые 31), ruff чисто. Особо: `test_defaults_bitexact_golden` доказывает бит-в-бит.

- [ ] **Step 10: Commit**

```bash
git add utils/physics_hybrid.py tests/test_physics_hybrid_exp16_port.py tests/goldens/exp16_kernel_default_out.pt
git commit -m "feat(physics): port exp14 opt-in terms to hybrid kernel — advective form, sphere metric, HS friction, lagrange3, spherical divergence"
```

---

### Task 4: Порт exp15-термов (omega_free, latent heating, клим-источники)

Эталон: `utils/physics.py:1064-1077` (omega_free u/v), `:1312-1330` (latent + omega_free t, z на полном t_t), `:1203-1211` (omega_free q, конденсация остаётся), `:1285-1334` (sources).

**Files:**
- Modify: `utils/physics_hybrid.py` (PDE_kernel: __init__, get_uv_dt, t_evolution, q_evolution, _evolve_fields; проброс PDE_block/HybridBlock)
- Test: `tests/test_physics_hybrid_exp16_port.py` (дополнить)

**Interfaces:**
- Produces: kwargs `omega_free: tuple[str, ...] = ()`, `latent_heating_coupling: bool = False`, `clim_sources_path: str | None = None`, `clim_sources_prefix: str = "C15_now__"` на PDE_kernel/PDE_block/HybridBlock.
- Клим-источники: annual-приближение exp15 (месячный индекс в модельном forward недоступен — календарного времени в API нет; ограничение фиксируется в отчёте). Буферы из `<prefix>annual_{t,q,z}` формы (13, 32) → усреднение широты 32→grid_h → `(1, 13, grid_h, 1)`.

- [ ] **Step 1: Failing-тесты (дополнить файл)**

```python
def test_omega_free_t_keeps_full_tt_for_z() -> None:
    """omega_free('t'): тенденция t усечена, но z интегрирует ПОЛНЫЙ t_t."""
    base, ofree = _kernel(), _kernel(omega_free=("t",))
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_free = ofree.physics_only_forward(state)
    z_base, t_base = out_base.chunk(5, dim=1)[0], out_base.chunk(5, dim=1)[1]
    z_free, t_free = out_free.chunk(5, dim=1)[0], out_free.chunk(5, dim=1)[1]
    assert torch.equal(z_base, z_free)          # z не изменился (полный t_t)
    assert not torch.equal(t_base, t_free)      # t — изменился


def test_omega_free_q_keeps_condensation() -> None:
    """omega_free('q') убирает w·q_z, но конденсация остаётся (зеркало physics.py)."""
    kernel = _kernel(omega_free=("q",))
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    w = kernel.get_w(u, v)
    q_t = kernel.get_q_dt(u, v, t, w, q)
    assert torch.isfinite(q_t).all()


def test_latent_heating_cools_saturated_ascent_tendency() -> None:
    """Скрытое тепло: t_t(latent) − t_t(base) = −(L/c_p)·cond ≥ 0 (cond ≤ 0)."""
    base, lat = _kernel(), _kernel(latent_heating_coupling=True)
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    q = torch.full_like(q, 0.02)  # форсируем насыщение
    w = base.get_w(u, v)
    base.share_z_dxyz(z)
    lat.share_z_dxyz(z)
    tt_base = base.get_t_t(u, v, w, t)
    tt_lat = lat.get_t_t(u, v, w, t)
    assert (tt_lat - tt_base).min() >= 0


def test_clim_sources_shift_tendency_by_buffer() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    arrays = {
        f"C15_now__annual_{k}": rng.normal(0, 1e-5, size=(13, 32)).astype("float32")
        for k in ("t", "q", "z")
    }
    np.savez("tests/goldens/exp16_clim_stub.npz", **arrays)
    kernel = _kernel(clim_sources_path="tests/goldens/exp16_clim_stub.npz")
    assert kernel.clim_src_t.shape == (1, 13, 8, 1)
    expected_row0 = arrays["C15_now__annual_t"][:, 0:4].mean(axis=1)
    assert torch.allclose(kernel.clim_src_t[0, :, 0, 0], torch.from_numpy(expected_row0))
```

Прогнать: `pytest tests/test_physics_hybrid_exp16_port.py -q` → новые тесты FAIL (`TypeError: unexpected keyword argument 'omega_free'`).

- [ ] **Step 2: __init__ — параметры и буферы**

Сигнатура PDE_kernel: добавить `omega_free: tuple[str, ...] = ()`, `latent_heating_coupling: bool = False`, `clim_sources_path: str | None = None`, `clim_sources_prefix: str = "C15_now__"`. Валидация и буферы (после блока exp14):

```python
        if not set(omega_free) <= {"u", "v", "t", "q"}:
            raise ValueError(f"omega_free must be a subset of u,v,t,q; got {omega_free!r}")
        self.omega_free = tuple(omega_free)
        self.latent_heating_coupling = latent_heating_coupling
        # exp 15: климатологические Q₁/Q₂-источники (annual-приближение S2_map:
        # месячный индекс недоступен в модельном forward — календаря в API нет).
        if clim_sources_path is not None:
            arrays = dict(np.load(clim_sources_path))
            for var in ("t", "q", "z"):
                annual = torch.from_numpy(arrays[f"{clim_sources_prefix}annual_{var}"])
                # (P, H_data) → усреднение широтных строк до grid_h → (1, P, grid_h, 1)
                pooled = F.adaptive_avg_pool1d(annual.unsqueeze(0), grid_h).squeeze(0)
                self.register_buffer(f"clim_src_{var}", pooled.reshape(1, -1, grid_h, 1))
        else:
            self.clim_src_t = None
            self.clim_src_q = None
            self.clim_src_z = None
```

Импорты наверху: `import numpy as np` и `import torch.nn.functional as F` (проверить — F уже импортирован в файле).

- [ ] **Step 3: omega_free в get_uv_dt**

В обеих ветках форм убрать вертикальный член по флагу — в начале `get_uv_dt`:

```python
        w_u = torch.zeros_like(w) if "u" in self.omega_free else w
        w_v = torch.zeros_like(w) if "v" in self.omega_free else w
```

и использовать `w_u`/`w_v` вместо `w` в adv_u/adv_v соответственно (в advective: `w_u * self._d_z(u)`; во flux: `self._d_z(u * w_u)`).

- [ ] **Step 4: latent heating в get_t_t + omega_free('t') с двойным t_t в t_evolution**

Эталон `utils/physics.py:1312-1315`: latent добавляется к t_t **до** z-интеграла
(z интегрирует t_t со скрытым теплом), а omega_free усечёт только ВЫХОДНУЮ
тенденцию. В `get_t_t` перед `return self.t_t` добавить:

```python
        if self.latent_heating_coupling:
            cond = self._condensation_source(t, self._q_for_latent, w)
            self.t_t = self.t_t - (self.L / self.c_p) * cond
```

Заменить `t_evolution`:

```python
    def t_evolution(self, u, v, w, t):
        t_t = self.get_t_t(u, v, w, t)  # полный (+latent): кэш self.t_t уходит в z-интеграл
        if "t" in self.omega_free:
            # Зеркало physics.py rhs: убрать адиабату и вернуть w·t_z из ВЫХОДНОЙ
            # тенденции; self.t_t (для z) остаётся полным.
            omega_pa = -100.0 * w
            pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
            adia = self.R_d * t * omega_pa / (self.c_p * pressure_pa)
            t_t = t_t - adia + w * self._d_z(t)
        return t + self._limit_increment(t_t * self.block_dt, t, "t")
```

Конденсацию вынести из `get_q_dt` в метод `_condensation_source(self, t, q, w)` (тело — внутренние `get_qs`/`get_F`/`delta`-строки текущего `get_q_dt`, возвращает `delta * F_ * omega_pa / pressure_pa`), `get_q_dt` вызывает его. В `_evolve_fields` перед вызовом `t_evolution` добавить строку `self._q_for_latent = q_old` (и `self._q_for_latent = torch.zeros_like(q)` fallback в `__init__` не нужен — `_evolve_fields` единственная точка входа, ставит кэш всегда). В тесте `test_latent_heating_cools_saturated_ascent_tendency` перед вызовами `get_t_t` выставить `base._q_for_latent = q; lat._q_for_latent = q`.

- [ ] **Step 5: omega_free('q') в get_q_dt + клим-источники в эволюциях**

В `get_q_dt` финал:

```python
        cond = self._condensation_source(t, q, w)
        if "q" in self.omega_free:
            q_t = -(u * q_x + v * q_y) + cond
        else:
            q_t = -(u * q_x + v * q_y + w * q_z) + cond
        if self.clim_src_q is not None:
            q_t = q_t + self.clim_src_q
        return q_t
```

В `t_evolution` перед `return`: `if self.clim_src_t is not None: t_t = t_t + self.clim_src_t`. В `z_evolution`: `z_t = self.get_z_t()`; после — `if self.clim_src_z is not None: z_t = z_t + self.clim_src_z`.

- [ ] **Step 6: Проброс через PDE_block/HybridBlock (те же 4 kwargs), тесты, ruff, commit**

```bash
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python -m pytest tests/test_physics_hybrid_exp16_port.py tests/test_physics_sign_conventions.py -q
ruff check . && ruff format --check .
git add utils/physics_hybrid.py tests/
git commit -m "feat(physics): port exp15 opt-in terms to hybrid kernel — omega_free with full-t_t z-integral, latent heating coupling, annual climatological Q1/Q2 sources"
```

Expected: все PASS, включая `test_defaults_bitexact_golden` (дефолты нетронуты).

---

### Task 5: Конфиги R4/R5, проброс из модели, bit-exact, сабмит

**Files:**
- Modify: `Models/IAM4VP.py` (конструктор: новые `physics_*` параметры → HybridBlock, по паттерну `physics_w_diagnostic`)
- Create: `configs/ablation16/abl16_r4_exp14_s0.yaml`, `configs/ablation16/abl16_r5_exp15_s0.yaml`
- Create: `tools/export_abl16_clim.py` (копия клим-NPZ в постоянный путь кластера)

**Interfaces:**
- Consumes: kwargs Task 3/4 на HybridBlock.
- Produces: параметры конфига `physics_advection_form`, `physics_metric_terms`, `physics_spherical_divergence`, `physics_rayleigh_friction`, `physics_vertical_scheme`, `physics_omega_free`, `physics_latent_heating_coupling`, `physics_clim_sources_path`.

- [ ] **Step 1: Проброс в IAM4VP**

Grep `physics_w_diagnostic` в `Models/IAM4VP.py` (параметр конструктора → атрибут → kwarg HybridBlock) и добавить рядом 8 параметров с дефолтами Task 3/4 (`physics_omega_free: tuple[str, ...] | list[str] = ()` — приводить `tuple(...)`). Пины: `test_new_kwargs_accepted`-подобный смоук через `get_model` не нужен — bit-exact harness Step 3 покрывает.

- [ ] **Step 2: Конфиги R4/R5**

R4 = копия `configs/ablation16/abl16_r3_a2_exp13_s0.yaml` с `name: abl16-r4-exp14-s0` и блоком в `model.params`:

```yaml
    physics_advection_form: advective
    physics_metric_terms: true
    physics_spherical_divergence: true
    physics_rayleigh_friction: true
    physics_vertical_scheme: lagrange3
```

R5 = копия R4 с `name: abl16-r5-exp15-s0` плюс:

```yaml
    physics_omega_free: [t, q, u, v]
    physics_latent_heating_coupling: true
    physics_clim_sources_path: /home/fa.buzaev/abl16_data/eq15_clim_summary_usa_2000.npz
```

- [ ] **Step 3: Bit-exact harness всех существующих армов**

```bash
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python tools/sanity_pi_iam4vp_gpu.py --help >/dev/null 2>&1 || true
# основная проверка — голден-тест Task 3 + полный пайтест:
/Users/buzaev-fa/miniconda3/envs/predformer/bin/python -m pytest tests/ -q
ruff check .
```

Expected: все зелёные.

- [ ] **Step 4: Клим-NPZ на кластер + сабмит R4/R5**

```bash
git add Models/IAM4VP.py configs/ablation16/ && git commit -m "feat(models): expose exp14/15 hybrid kernel flags; exp16 R4/R5 configs" && git push origin fix_inline_v2
ssh cluster 'mkdir -p ~/abl16_data && git -C ~/wt_fix_v2 checkout --detach origin/fix_inline_v2 && cp ~/wt_fix_v2/docs/experiments/15_deep_equation_improvement/results/eq15_clim_summary_usa_2000.npz ~/abl16_data/ 2>/dev/null; git -C ~/WeatherPredictions fetch origin && cp ~/WeatherPredictions/docs/experiments/15_deep_equation_improvement/results/eq15_clim_summary_usa_2000.npz ~/abl16_data/ 2>/dev/null; ls -la ~/abl16_data/'
ssh cluster '
MM=$HOME/era5_memmap/predformer_usa_2000_2004.dat
EXPORT=ALL,WEATHERPRED_USA_MEMMAP=$MM,WEATHERPRED_CONDA_ENV_NAME=pi-iamvp,WEATHERPRED_CHECKPOINT_BASE=$HOME/abl16_ckpt
cd ~/wt_fix_v2
for c in abl16_r4_exp14_s0 abl16_r5_exp15_s0; do
  sbatch --parsable -A proj_1715 -J $c --export=$EXPORT sh_files/train_pi_iam4vp_residual_usa_v4_2gpu.sh configs/ablation16/$c.yaml
done'
```

Expected: 2 jobid. Мониторить первые минуты как в Task 2 Step 5.

---

### Task 6: Волна 2 — сиды 1–2 + якорь A (после анализа сида 0)

- [ ] **Step 1: Сиды 1–2**

Скриптом Task 1/2 сгенерировать копии всех 7 конфигов с `seed: 1|2` и суффиксом имени `-s1`/`-s2` (замена `seed: 0`→`seed: 1`, `-s0`→`-s1`), сабмит теми же командами. Запускать ТОЛЬКО если эффекты ступеней на сиде 0 сопоставимы с ожидаемым межсидовым шумом (решение фиксируется в journal).

- [ ] **Step 2: Якорь A**

```bash
# configs/ablation16/abl16_anchor_pi_iam4vp_s0.yaml = копия configs/pi_iam4vp_usa_v4.yaml
# с name: abl16-anchor-pi-iam4vp-s0, seed: 0, max_epoch: 30 (скрипт Task 1)
ssh cluster 'cd ~/wt_fix_v2 && sbatch -A proj_1715 -J abl16-anchor --export=ALL,WEATHERPRED_USA_MEMMAP=$HOME/era5_memmap/predformer_usa_2000_2004.dat,WEATHERPRED_CONDA_ENV_NAME=pi-iamvp,WEATHERPRED_CHECKPOINT_BASE=$HOME/abl16_ckpt sh_files/train_pi_iam4vp_usa_v4_2gpu.sh configs/ablation16/abl16_anchor_pi_iam4vp_s0.yaml'
```

---

### Task 7: Сбор метрик и отчёт

**Files:**
- Create: `docs/experiments/16_model_ablation_ladder/collect_metrics.py`
- Create: `docs/experiments/16_model_ablation_ladder/make_figures.py` (по образцу `docs/experiments/15_deep_equation_improvement/make_figures.py`)
- Modify: `docs/experiments/16_model_ablation_ladder/README.md` (секция «Результаты»)

- [ ] **Step 1: Экспорт per-epoch метрик из Comet**

`collect_metrics.py` (запуск локально, `COMET_API_KEY` из `.env`):

```python
"""Экспорт val-метрик abl16-ранов из Comet в results/abl16_metrics.json."""

import json
import os
import pathlib

from comet_ml.api import API

RUNS = [
    "abl16-r0-no-physics-s0",
    "abl16-r1-legacy-hybrid-s0",
    "abl16-r2a-a1-pre13-s0",
    "abl16-r2-a2-pre13-s0",
    "abl16-r3-a2-exp13-s0",
    "abl16-r4-exp14-s0",
    "abl16-r5-exp15-s0",
]


def main() -> None:
    api = API(api_key=os.environ["COMET_API_KEY"])
    workspace = os.environ["COMET_WORKSPACE"]
    project = os.environ.get("COMET_PROJECT_NAME", "weather-plus-physics")
    out: dict[str, dict] = {}
    for run in RUNS:
        matches = api.get_experiments(workspace, project_name=project, pattern=run)
        if not matches:
            print(f"[collect] MISSING {run}")
            continue
        exp = matches[-1]
        metrics: dict[str, list] = {}
        for m in exp.get_metrics_summary():
            name = m["name"]
            if "val" in name.lower() or "rmse" in name.lower():
                metrics[name] = exp.get_metrics(name)
        out[run] = metrics
    dst = pathlib.Path(__file__).parent / "results" / "abl16_metrics.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, default=str))
    print(f"[collect] written {dst}")


if __name__ == "__main__":
    main()
```

Первый запуск покажет реальные имена метрик — при необходимости скорректировать фильтр (имена см. в Comet UI, namespace после 576983d без legacy-префикса).

- [ ] **Step 2: Фигуры + таблица**

`make_figures.py`: (а) линейные графики val-RMSE per variable (z,t,q,u,v) по эпохам 3..30, один цвет на арм (все армы на одном графике); (б) итоговая таблица эпохи 30 (+лучшая эпоха) в Markdown. Записать `fig1_curves_<var>.png`, `results/abl16_final_table.md`.

- [ ] **Step 3: README-результаты + вывод + commit**

Заполнить секцию «Результаты» README таблицей и 3–5 выводами (вклад каждой ступени; сопоставление со ступенями диагностики exp13/14/15; ограничения — annual-климатология, латент-сетка). Обновить `CHANGELOG.md` (Added: exp 16). Прогнать `ruff check .`, коммит:

```bash
git add docs/experiments/16_model_ablation_ladder/ CHANGELOG.md
git commit -m "docs(experiments): exp 16 model ablation ladder — wave-1 results, figures, conclusions"
git push origin fix_inline_v2
```
