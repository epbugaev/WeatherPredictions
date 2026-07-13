# exp19 — level-targeted equations Implementation Plan

> [!CAUTION]
> **ПЛАН ПРИОСТАНОВЛЕН — 2026-07-13.** [Экс. 18 отозван](../../experiments/18_level_resolved_physics/),
> а вся мотивация этого плана выведена из его принципа. На плато валидации посылка
> каждого рычага отпала: маска уровня **вредна**, «r-налог» и «удар R3q по T@850»
> не существуют, а вреда физики в погранслое нет. Фаза A/B **не запускалась**.
> Прежде чем продолжать: снять маску уровня и обучаемый гейт, а Экман и CC-мост
> переобосновать как улучшение *уравнений* (невязка внизу реальна), а не как
> лечение послойного скилла. Подробности — [спека, шапка](../specs/2026-07-13-exp19-level-targeted-equations-design.md).

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать три opt-in физических механизма (маска уровня, Экмановское трение, CC-мост влажности) как переключаемые флаги ядра PI-IAM4VP, измерить кандидатов на данных (фаза A) и обучить прошедшие армы поверх R5 (фаза B).

**Architecture:** Три независимых механизма добавляются как opt-in флаги в `PDE_kernel` → `PDE_block` → `HybridBlock` → `IAM4VP`, дефолты сохраняют текущее поведение бит-в-бит (пин через golden). Фаза A расширяет самодостаточный диагностический скрипт exp18 (`utils.physics.PurePDEKernel`-путь) для измерения Экмана и r-невязки. Фаза B — конфиги `abl19-*` поверх базы R5 (t=12, 90 эпох, сид 0) и тот же rollout/level-profile харнесс, что exp16/18.

**Tech Stack:** PyTorch, numpy, h5netcdf; кластер cHARISMa (sbatch, CPU-партиция для фазы A, GPU для фазы B); локально `uv run --python 3.12`; ruff; pytest.

## Global Constraints

- **Никаких `try/except`** — только LBYL (проверки `if ... is not None`, `assert`, валидация флагов через `raise ValueError`). [CLAUDE.md §2]
- **Никаких локальных `import`** внутри функций/методов — все импорты на модульном уровне. [CLAUDE.md §2]
- **Дефолты всех новых флагов сохраняют поведение бит-в-бит.** После каждой правки ядра `tests/test_physics_hybrid_exp16_port.py::test_defaults_bitexact_golden` обязан проходить (golden `tests/goldens/exp16_kernel_default_out.pt`).
- **Порядок каналов ядра:** `zquvtw` = `[z, t, q, u, v]` по 13 уровней; ядро эволюционирует уже-нарезанные per-переменные поля `(B, 13, H, W)`.
- **Уровни давления (гПа), индексы 0..12:** `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]`. Буфер `pressure` формы `(1, 13, 1, 1)` уже в модуле.
- **Маска НЕ применяется к z** (производный гидростатический интеграл; маскирование рвёт когерентность столба — вывод exp18). Маскируются только u, v, t, q.
- **Train-скрипты не трогаем** — только конфиги `configs/abl19-*`.
- **Все `sbatch` — с `-A proj_1715`.** Не трогать посторонние джобы (`aj-exp015` и др.).
- **Не коммитить** веса/логи/чекпоинты; `docs/` вне ruff-скоупа (T201-print в скриптах легитимны). Не стейджить чужие untracked (`docs/planKDD.md`, `example/`).
- **ruff чист** (`ruff check .` и `ruff format .`) на всех .py вне `docs/`.
- **Коммиты** — conventional commits на английском, заканчиваются строкой:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Локальный `python3` = 3.9** — не тянет `int | None` в рантайме и `zip(strict=)`; запускать скрипты-фигуры/анализ через `uv run --python 3.12 --with numpy --with matplotlib python ...`. pytest ядра работает на проектном окружении.

---

## File Structure

- `utils/physics_hybrid.py` (MODIFY) — три флага в `PDE_kernel`, проброс в `PDE_block`/`HybridBlock`.
- `Models/IAM4VP.py` (MODIFY) — `physics_*` kwargs для трёх флагов + проброс в `HybridBlock`.
- `tests/test_physics_hybrid_exp19.py` (CREATE) — юнит-пины трёх механизмов + сквозной проброс.
- `docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py` (CREATE) — Экман + r-невязка на ERA5 (self-contained, PurePDEKernel).
- `docs/experiments/19_level_targeted_equations/README.md` (CREATE) — отчёт: фаза A (пороги/приёмка) + фаза B (армы, скилл).
- `docs/experiments/19_level_targeted_equations/results/` (CREATE) — JSON фазы A.
- `configs/abl19/abl19_b1_mask_hard_t12.yaml`, `abl19_b1l_mask_learnable_t12.yaml`, `abl19_b2_ekman_t12.yaml`, `abl19_b3_ccbridge_t12.yaml` (CREATE).
- `sh_files/abl19_train_array.sh`, `sh_files/abl19_rollout_t12_array.sh` (CREATE).
- `docs/experiments/README.md` (MODIFY) — строка индекса exp19.

---

## PHASE 0 — Код: opt-in флаги ядра (безусловно)

### Task 1: Жёсткая маска уровня (B1) в PDE_kernel

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_kernel.__init__`, новый метод `_mask_increment`, вызовы в `uv_evolution`/`t_evolution`/`q_evolution`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: модульный `pressure` `(1,13,1,1)`; `_limit_increment(raw, state, var_key) -> Tensor` (возвращает `.detach()`).
- Produces:
  - `PDE_kernel.__init__(..., physics_level_mask: dict[str, float] | None = None)` — ключи ⊆ `{"u","v","t","q"}`, значение = порог давления в гПа; физика активна где `pressure <= порог`, ниже обнулена.
  - буферы `level_mask_u/v/t/q` формы `(1,13,1,1)` (регистрируются только когда `physics_level_mask is not None`).
  - `PDE_kernel._mask_increment(inc: Tensor, var: str) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics_hybrid_exp19.py
import torch

from utils.physics_hybrid import PDE_kernel


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


def test_mask_hard_zeros_boundary_layer_u_only() -> None:
    """physics_level_mask {'u':700}: u не меняется на 850/925/1000; v,t,q — как без маски."""
    base = _kernel()
    masked = _kernel(physics_level_mask={"u": 700})
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_mask = masked.physics_only_forward(state)
    z_b, t_b, q_b, u_b, v_b = out_base.chunk(5, dim=1)
    z_m, t_m, q_m, u_m, v_m = out_mask.chunk(5, dim=1)
    z_in, t_in, q_in, u_in, v_in = state.chunk(5, dim=1)
    # u заморожен на индексах 10,11,12 (850/925/1000):
    assert torch.equal(u_m[:, 10:], u_in[:, 10:])
    # u выше отсечки эволюционирует (как в base):
    assert torch.equal(u_m[:, :10], u_b[:, :10])
    # v,t,q не затронуты маской:
    assert torch.equal(v_m, v_b) and torch.equal(t_m, t_b) and torch.equal(q_m, q_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_mask_hard_zeros_boundary_layer_u_only -v`
Expected: FAIL — `PDE_kernel.__init__() got an unexpected keyword argument 'physics_level_mask'`

- [ ] **Step 3: Write minimal implementation**

В `PDE_kernel.__init__` добавить параметр в сигнатуру (после `clim_sources_prefix`):
```python
        physics_level_mask: dict[str, float] | None = None,
```
После блока валидации `omega_free` (после строки `self.omega_free = tuple(omega_free)`) добавить:
```python
        if physics_level_mask is not None and not set(physics_level_mask) <= {"u", "v", "t", "q"}:
            raise ValueError(
                f"physics_level_mask keys must be subset of u,v,t,q; got {physics_level_mask!r}"
            )
        self.physics_level_mask = physics_level_mask
```
В конце `__init__` (после регистрации clim-буферов, до `self.variable_norm = ...`) добавить регистрацию буферов:
```python
        if physics_level_mask is not None:
            for mask_var in ("u", "v", "t", "q"):
                threshold = physics_level_mask.get(mask_var)
                if threshold is None:
                    mask = torch.ones_like(pressure, dtype=torch.float32)
                else:
                    mask = (pressure <= threshold).to(torch.float32)
                self.register_buffer(f"level_mask_{mask_var}", mask)
```
Добавить метод (рядом с `_limit_increment`):
```python
    def _mask_increment(self, increment: torch.Tensor, var: str) -> torch.Tensor:
        """Обнуляет физическое приращение переменной ниже отсечки давления.

        Маска — буфер ``level_mask_<var>`` формы ``(1, 13, 1, 1)`` из порогов
        ``physics_level_mask``; при выключенной маске приращение возвращается
        без изменений (бит-в-бит). z не маскируется (диагностический интеграл).

        Args:
            increment: приращение поля ``(B, 13, H, W)`` (выход ``_limit_increment``).
            var: ключ переменной, одно из ``{'u','v','t','q'}``.

        Returns:
            ``torch.Tensor`` той же формы — приращение после маски уровня.
        """
        if self.physics_level_mask is None:
            return increment
        return increment * getattr(self, f"level_mask_{var}")
```
Обернуть приращения в трёх методах:
```python
    def uv_evolution(self, u, v, w):
        u_t, v_t = self.get_uv_dt(u, v, w)
        u = u + self._mask_increment(self._limit_increment(u_t * self.block_dt, u, "u"), "u")
        v = v + self._mask_increment(self._limit_increment(v_t * self.block_dt, v, "v"), "v")
        return u, v
```
В `t_evolution` заменить `return`:
```python
        return t + self._mask_increment(self._limit_increment(t_t * self.block_dt, t, "t"), "t")
```
В `q_evolution` заменить `return`:
```python
        return q + self._mask_increment(self._limit_increment(q_t * self.block_dt, q, "q"), "q")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_mask_hard_zeros_boundary_layer_u_only tests/test_physics_hybrid_exp16_port.py::test_defaults_bitexact_golden -v`
Expected: PASS (оба — новый тест и golden бит-в-бит)

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): hard per-level physics mask in PDE_kernel

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Обучаемый гейт маски (B1L) в PDE_kernel

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_kernel.__init__`, `_mask_increment`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: `physics_level_mask` (Task 1) как источник инициализации; буфер `pressure`.
- Produces:
  - `PDE_kernel.__init__(..., physics_level_mask_learnable: bool = False)`.
  - `nn.Parameter` `level_gate_logit` формы `(4, 13)`, строки в порядке `("u","v","t","q")`, α = `sigmoid(logit)`. Инициализация `+4.0` где жёсткая маска = 1, `−4.0` где 0.

- [ ] **Step 1: Write the failing test**

```python
def test_mask_learnable_gate_grad_and_init() -> None:
    """Обучаемый гейт: параметр (4,13) с градиентом; α≈жёсткой маске при init ±4."""
    masked = _kernel(
        physics_level_mask={"u": 700, "v": 700, "t": 850, "q": 850},
        physics_level_mask_learnable=True,
    )
    assert masked.level_gate_logit.shape == (4, 13)
    assert masked.level_gate_logit.requires_grad
    alpha = torch.sigmoid(masked.level_gate_logit)
    # u (строка 0): выше 700 (индексы 0..9) ≈1, ниже ≈0
    assert alpha[0, :10].min() > 0.95 and alpha[0, 10:].max() < 0.05
    # t (строка 2): отсечка 850 → индексы 0..10 ≈1, 11,12 ≈0
    assert alpha[2, :11].min() > 0.95 and alpha[2, 11:].max() < 0.05


def test_mask_learnable_default_off_bitexact() -> None:
    """Без learnable-флага гейт не создаётся; выход бит-в-бит с немаскированным."""
    base = _kernel()
    assert not hasattr(base, "level_gate_logit")
    with torch.no_grad():
        out = base.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_mask_learnable_gate_grad_and_init -v`
Expected: FAIL — `unexpected keyword argument 'physics_level_mask_learnable'`

- [ ] **Step 3: Write minimal implementation**

В сигнатуру `PDE_kernel.__init__` (после `physics_level_mask`):
```python
        physics_level_mask_learnable: bool = False,
```
После валидации `physics_level_mask` добавить:
```python
        if physics_level_mask_learnable and physics_level_mask is None:
            raise ValueError("physics_level_mask_learnable=True requires physics_level_mask init")
        self.physics_level_mask_learnable = physics_level_mask_learnable
```
В блоке регистрации буферов маски, обернуть в `if physics_level_mask is not None:` уже добавлено (Task 1). После цикла регистрации буферов добавить обучаемый параметр:
```python
        if physics_level_mask_learnable:
            mask_rows = torch.stack(
                [getattr(self, f"level_mask_{v}").reshape(-1) for v in ("u", "v", "t", "q")]
            )
            self.level_gate_logit = nn.Parameter(torch.where(mask_rows > 0.5, 4.0, -4.0))
```
Расширить `_mask_increment`:
```python
    def _mask_increment(self, increment: torch.Tensor, var: str) -> torch.Tensor:
        """..."""  # (docstring из Task 1 + строка про learnable)
        if self.physics_level_mask is None:
            return increment
        if self.physics_level_mask_learnable:
            row = ("u", "v", "t", "q").index(var)
            alpha = torch.sigmoid(self.level_gate_logit[row]).reshape(1, -1, 1, 1)
            return increment * alpha
        return increment * getattr(self, f"level_mask_{var}")
```
Обновить docstring `_mask_increment`: добавить строку «при ``physics_level_mask_learnable`` маска — обучаемый гейт α=sigmoid(logit)».

Также добавить `self.physics_level_mask_learnable = False` инициализацию в ветке где `physics_level_mask is None` не нужна — атрибут ставится всегда через строку выше (`self.physics_level_mask_learnable = physics_level_mask_learnable`). ОК.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py tests/test_physics_hybrid_exp16_port.py::test_defaults_bitexact_golden -v`
Expected: PASS (все)

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): learnable per-level physics gate (sigmoid, init from hard mask)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Проброс флагов маски через PDE_block → HybridBlock → IAM4VP

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_block.__init__`, `HybridBlock.__init__`)
- Modify: `Models/IAM4VP.py` (`IAM4VP.__init__` + вызов `HybridBlock(...)`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: `PDE_kernel(..., physics_level_mask, physics_level_mask_learnable)` (Tasks 1–2).
- Produces: те же два kwargs в `PDE_block.__init__`, `HybridBlock.__init__`, и `IAM4VP.__init__` как `physics_level_mask`/`physics_level_mask_learnable`.

- [ ] **Step 1: Write the failing test**

```python
from utils.physics_hybrid import HybridBlock


def test_mask_flags_reach_kernel_through_hybrid_block() -> None:
    """Сквозной проброс: HybridBlock → PDE_block → PDE_kernel."""
    torch.manual_seed(0)
    block = HybridBlock(
        dim=65, zquvtw_channel=13, depth=2, block_dt=300.0, inverse_time=False,
        physics_part_coef=0.5, w_diagnostic="mass_consistent",
        lat_start_deg=18.28125, dlat_deg=5.625, dlon_deg=5.625, grid_h=8,
        physical_passthrough=True,
        physics_level_mask={"u": 700, "v": 700},
        physics_level_mask_learnable=True,
    )
    for kernel in block.pde_block.PDE_kernels:
        assert kernel.physics_level_mask == {"u": 700, "v": 700}
        assert kernel.physics_level_mask_learnable
        assert kernel.level_gate_logit.shape == (4, 13)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_mask_flags_reach_kernel_through_hybrid_block -v`
Expected: FAIL — `HybridBlock.__init__() got an unexpected keyword argument 'physics_level_mask'`

- [ ] **Step 3: Write minimal implementation**

`PDE_block.__init__`: добавить в сигнатуру (после `clim_sources_prefix`):
```python
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
```
и передать в `PDE_kernel(...)` в цикле:
```python
                    physics_level_mask=physics_level_mask,
                    physics_level_mask_learnable=physics_level_mask_learnable,
```
`HybridBlock.__init__`: те же два kwargs в сигнатуру (после `clim_sources_prefix`) и в вызов `PDE_block(...)`. Найти вызов `self.pde_block = PDE_block(` и добавить:
```python
            physics_level_mask=physics_level_mask,
            physics_level_mask_learnable=physics_level_mask_learnable,
```
`Models/IAM4VP.py`: в `__init__` (рядом с `physics_clim_sources_path`):
```python
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
```
сохранить: `self.physics_level_mask = physics_level_mask`, `self.physics_level_mask_learnable = physics_level_mask_learnable`.
В вызове `HybridBlock(...)` (строки ~503–511) добавить:
```python
            physics_level_mask=self.physics_level_mask,
            physics_level_mask_learnable=self.physics_level_mask_learnable,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py tests/test_physics_hybrid_exp16_port.py -v`
Expected: PASS (все, включая golden)

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py Models/IAM4VP.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): plumb level-mask flags through HybridBlock and IAM4VP

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Экмановское трение (вертикальная диффузия импульса) в PDE_kernel

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_kernel.__init__`, `get_uv_dt`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: `_d_z(field) -> Tensor` (= −∂/∂p); буфер `pressure`.
- Produces:
  - `PDE_kernel.__init__(..., ekman_K_profile: tuple[float, ...] | None = None)` — 13 коэффициентов вихревой вязкости K(p) (Па²/с); None → выключено (бит-в-бит).
  - буфер `ekman_K` `(1,13,1,1)`; член `F = _d_z(K * _d_z(field))` = `∂/∂p(K ∂field/∂p)` добавляется к `u_t`, `v_t`.

- [ ] **Step 1: Write the failing test**

```python
def test_ekman_default_off_bitexact() -> None:
    """ekman_K_profile=None → выход бит-в-бит golden."""
    kernel = _kernel()
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)


def test_ekman_adds_finite_boundary_layer_term() -> None:
    """Профиль K, пиковый у поверхности: u_t меняется, наверху (K=0) — нет."""
    profile = tuple([0.0] * 10 + [1.0e6, 2.0e6, 3.0e6])  # ненуль на 850/925/1000
    base, ekman = _kernel(), _kernel(ekman_K_profile=profile)
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    w = base.get_w(u, v)
    base.share_z_dxyz(z)
    ekman.share_z_dxyz(z)
    ut_base, _ = base.get_uv_dt(u, v, w)
    ut_ekman, _ = ekman.get_uv_dt(u, v, w)
    diff = ut_ekman - ut_base
    assert torch.isfinite(diff).all()
    assert diff[:, :10].abs().max() == 0  # K=0 наверху → член нулевой
    assert diff[:, 10:].abs().max() > 0   # погранслой затронут
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_ekman_adds_finite_boundary_layer_term -v`
Expected: FAIL — `unexpected keyword argument 'ekman_K_profile'`

- [ ] **Step 3: Write minimal implementation**

Сигнатура `PDE_kernel.__init__` (после `physics_level_mask_learnable`):
```python
        ekman_K_profile: tuple[float, ...] | None = None,
```
Валидация + буфер (рядом с блоком rayleigh в `__init__`):
```python
        if ekman_K_profile is not None:
            if len(ekman_K_profile) != variable_dim:
                raise ValueError(
                    f"ekman_K_profile must have {variable_dim} entries; got {len(ekman_K_profile)}"
                )
            self.register_buffer(
                "ekman_K",
                torch.tensor(ekman_K_profile, dtype=torch.float32).reshape(1, -1, 1, 1),
            )
        else:
            self.ekman_K = None
```
В `get_uv_dt`, после блока `if self.rayleigh_friction:` (перед `if self.eddy_viscosity > 0:`):
```python
        if self.ekman_K is not None:
            # Экмановское трение: ∂/∂p(K ∂/∂p u). _d_z = −∂/∂p, поэтому двойное
            # применение восстанавливает знак: _d_z(K·_d_z(u)) = ∂/∂p(K ∂u/∂p).
            self.u_t = self.u_t + self._d_z(self.ekman_K * self._d_z(u))
            self.v_t = self.v_t + self._d_z(self.ekman_K * self._d_z(v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py tests/test_physics_hybrid_exp16_port.py::test_defaults_bitexact_golden -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): Ekman friction (vertical momentum diffusion) in PDE_kernel

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CC-мост влажности в PDE_kernel

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_kernel.__init__`, `_condensation_source` рефактор q_s, `q_evolution`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: кэш `self.t_t` (полный, после `get_t_t`, кэшируется в `t_evolution`); `_condensation_source`.
- Produces:
  - `PDE_kernel.__init__(..., humidity_evolution: str = "as_is")` — `"as_is"` (текущее) или `"cc_bridge"`.
  - в режиме `cc_bridge` q-приращение получает связку Клаузиуса–Клапейрона: `q_t += q · (L/(R_v·T²)) · t_t` (чтобы наблюдаемая r сохранялась под изменением T).

**Замечание по порядку в `_evolve_fields`:** `t_evolution` вызывается ДО `q_evolution`, и `get_t_t` кэширует `self.t_t` — так что `self.t_t` доступен в `q_evolution`. Это уже так (см. `_evolve_fields`: t_new раньше q_new).

- [ ] **Step 1: Write the failing test**

```python
def test_cc_bridge_default_off_bitexact() -> None:
    """humidity_evolution='as_is' (дефолт) → выход бит-в-бит golden."""
    kernel = _kernel()
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)


def test_cc_bridge_warming_adds_positive_q_tendency() -> None:
    """cc_bridge: при прогреве (t_t>0) добавка к q_t = q·(L/R_v T²)·t_t > 0."""
    base, cc = _kernel(), _kernel(humidity_evolution="cc_bridge")
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_cc = cc.physics_only_forward(state)
    q_base = out_base.chunk(5, dim=1)[2]
    q_cc = out_cc.chunk(5, dim=1)[2]
    assert torch.isfinite(out_cc).all()
    assert not torch.equal(q_base, q_cc)
    # z,t,u,v не затронуты CC-мостом (меняется только q-путь):
    for idx in (0, 1, 3, 4):
        assert torch.equal(out_base.chunk(5, dim=1)[idx], out_cc.chunk(5, dim=1)[idx])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_cc_bridge_warming_adds_positive_q_tendency -v`
Expected: FAIL — `unexpected keyword argument 'humidity_evolution'`

- [ ] **Step 3: Write minimal implementation**

Сигнатура `PDE_kernel.__init__` (после `ekman_K_profile`):
```python
        humidity_evolution: str = "as_is",
```
Валидация (рядом с другими строковыми флагами в `__init__`):
```python
        if humidity_evolution not in ("as_is", "cc_bridge"):
            raise ValueError(
                f"Unknown humidity_evolution {humidity_evolution!r}; expected 'as_is' or 'cc_bridge'"
            )
        self.humidity_evolution = humidity_evolution
```
Выделить q_s в `_condensation_source` в отдельный метод, чтобы CC-мост переиспользовал его без дублирования (DRY, CLAUDE.md §1). Добавить метод:
```python
    def _saturation_specific_humidity(self, t: torch.Tensor) -> torch.Tensor:
        """q_s(T,p) по Магнусу (СИ), detached. Общий для конденсации и CC-моста.

        Args:
            t: температура ``(B, 13, H, W)``, K.

        Returns:
            ``torch.Tensor`` той же формы — насыщающая удельная влажность, кг/кг.
        """
        pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
        t_c = t - 273.15
        exponent = torch.clamp(17.67 * t_c / self.avoid_inf(t_c + 243.5), min=-20.0, max=20.0)
        e_s = 6.112 * torch.exp(exponent) * 100
        q_s = (0.622 * e_s / self.avoid_inf(pressure_pa - 0.378 * e_s)).detach()
        return torch.maximum(q_s, torch.ones_like(q_s) * 1e-6)
```
В `_condensation_source` заменить строки вычисления `e_s`/`q_s` (от `t_c = t - 273.15` до `q_s = torch.maximum(...)`) на:
```python
        pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
        omega_pa = -100.0 * w
        q_s = self._saturation_specific_humidity(t)
```
(проверить golden — рефактор обязан быть бит-в-бит: та же арифметика, тот же порядок операций.)
В `q_evolution` перед `return`:
```python
    def q_evolution(self, u, v, t, w, q):
        q_t = self.get_q_dt(u, v, t, w, q)
        if self.humidity_evolution == "cc_bridge":
            # Мост Клаузиуса–Клапейрона: наблюдаемая величина — r=q/q_s; чтобы r
            # эволюционировала под изменением T, к q-тенденции добавляется
            # q·(L/(R_v T²))·t_t (dq_s/dT-связка). self.t_t — полный кэш из t_evolution.
            cc_term = q * (self.L / (self.R_v * t * t)) * self.t_t
            q_t = q_t + cc_term
        return q + self._mask_increment(self._limit_increment(q_t * self.block_dt, q, "q"), "q")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py tests/test_physics_hybrid_exp16_port.py -v`
Expected: PASS (включая golden — рефактор q_s бит-в-бит)

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): Clausius-Clapeyron humidity bridge in PDE_kernel

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Проброс Экмана + CC-моста через PDE_block → HybridBlock → IAM4VP

**Files:**
- Modify: `utils/physics_hybrid.py` (`PDE_block.__init__`, `HybridBlock.__init__`)
- Modify: `Models/IAM4VP.py` (`IAM4VP.__init__` + вызов `HybridBlock`)
- Test: `tests/test_physics_hybrid_exp19.py`

**Interfaces:**
- Consumes: `PDE_kernel(..., ekman_K_profile, humidity_evolution)` (Tasks 4–5).
- Produces: те же kwargs на всех трёх уровнях; IAM4VP-имена `physics_ekman_K_profile`, `physics_humidity_evolution`.

- [ ] **Step 1: Write the failing test**

```python
def test_ekman_cc_flags_reach_kernel_through_hybrid_block() -> None:
    """Сквозной проброс Экмана и CC-моста до ядра."""
    torch.manual_seed(0)
    profile = tuple([0.0] * 10 + [1.0e6, 2.0e6, 3.0e6])
    block = HybridBlock(
        dim=65, zquvtw_channel=13, depth=2, block_dt=300.0, inverse_time=False,
        physics_part_coef=0.5, w_diagnostic="mass_consistent",
        lat_start_deg=18.28125, dlat_deg=5.625, dlon_deg=5.625, grid_h=8,
        physical_passthrough=True,
        ekman_K_profile=profile,
        humidity_evolution="cc_bridge",
    )
    for kernel in block.pde_block.PDE_kernels:
        assert kernel.ekman_K is not None and kernel.ekman_K.shape == (1, 13, 1, 1)
        assert kernel.humidity_evolution == "cc_bridge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py::test_ekman_cc_flags_reach_kernel_through_hybrid_block -v`
Expected: FAIL — `unexpected keyword argument 'ekman_K_profile'`

- [ ] **Step 3: Write minimal implementation**

`PDE_block.__init__` и `HybridBlock.__init__`: добавить в сигнатуру (после `physics_level_mask_learnable`):
```python
        ekman_K_profile: tuple[float, ...] | None = None,
        humidity_evolution: str = "as_is",
```
и пробросить в `PDE_kernel(...)` (в `PDE_block`) / `PDE_block(...)` (в `HybridBlock`):
```python
                    ekman_K_profile=ekman_K_profile,
                    humidity_evolution=humidity_evolution,
```
`Models/IAM4VP.py`: в `__init__`:
```python
        physics_ekman_K_profile: tuple[float, ...] | None = None,
        physics_humidity_evolution: str = "as_is",
```
сохранить в `self.` и передать в `HybridBlock(...)`:
```python
            ekman_K_profile=self.physics_ekman_K_profile,
            humidity_evolution=self.physics_humidity_evolution,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_hybrid_exp19.py tests/test_physics_hybrid_exp16_port.py -v && ruff check utils/physics_hybrid.py Models/IAM4VP.py tests/test_physics_hybrid_exp19.py`
Expected: PASS + `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add utils/physics_hybrid.py Models/IAM4VP.py tests/test_physics_hybrid_exp19.py
git commit -m "feat(exp19): plumb Ekman and CC-bridge flags through HybridBlock and IAM4VP

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## PHASE A — Диагностика кандидатов на данных (кластер CPU)

### Task 7: phaseA_diagnostics.py — Экмановская невязка + калибровка K

**Files:**
- Create: `docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py`
- (self-contained, путь через `utils.physics.PurePDEKernel`, как exp18 `level_diagnostics.py`)

**Interfaces:**
- Consumes: `utils.physics.PurePDEKernel`, `utils.physics.Grid/GridConfig`, `utils.physics_hybrid.relative_to_specific_humidity`; ERA5 USA-кроп (env `ERA5_ROOT`, `YEAR`, `STRIDE`, `SYNTHETIC`, `MAX_TRIPLES`, `OUT`).
- Produces: функции `ekman_term(k, s, K_profile) -> dict[str, Tensor]` (u_t/v_t добавки), `run_ekman_sweep(...) -> dict` — per-level residual_rel u,v для набора K-профилей.

- [ ] **Step 1: Write the failing smoke test (inline, synthetic)**

Скрипт должен уметь smoke-прогон без ERA5:
Run: `cd docs/experiments/19_level_targeted_equations && OUT=/tmp/exp19A.json SYNTHETIC=1 MAX_TRIPLES=4 uv run --python 3.12 --with numpy --with torch --with h5netcdf python phaseA_diagnostics.py`
Expected (до реализации): FAIL — файла нет.

- [ ] **Step 2: Write implementation**

Скопировать каркас `18_level_resolved_physics/level_diagnostics.py` (REPO_ROOT-ленивый, `Era5Reader`, `synthetic_state`, `LevelAccum`, `build_kernel`, `data_latitudes_deg`), затем добавить Экман:
```python
# K(σ)-профиль: пиковый в погранслое, ноль в свободной тропосфере.
# σ = p/1000; profile[i] = K0 · max(0, (σ_i − σ_b)/(1 − σ_b)).
def ekman_profile(K0: float, sigma_b: float = 0.7) -> torch.Tensor:
    sigma = torch.tensor(PRESSURE_HPA, dtype=torch.float32).reshape(1, -1, 1, 1) / 1000.0
    return K0 * torch.clamp((sigma - sigma_b) / (1.0 - sigma_b), min=0.0)


def ekman_term(k, s: dict, K: torch.Tensor) -> dict:
    """∂/∂p(K ∂/∂p u,v) через k._d_z (= −∂/∂p). Добавка к u_t/v_t."""
    return {
        "u_t": k._d_z(K * k._d_z(s["u"])),
        "v_t": k._d_z(K * k._d_z(s["v"])),
    }
```
В `run()` для якоря `C15_now` (R5) прогнать sweep по `K0 ∈ [0, 5e5, 1e6, 2e6, 5e6, 1e7]`: для каждого K0 аккумулировать residual_rel u,v с добавкой `ekman_term`. Записать в JSON `ekman_sweep: {K0: {u:[...], v:[...]}}`.
Критерий приёмки (печатать в конце): найти K0, минимизирующий среднюю невязку u,v на индексах 11,12 (925/1000); принят, если падение ≥20% против K0=0 И невязка на индексах 0..9 (≤700) не выросла >5%.

- [ ] **Step 3: Run smoke**

Run: `cd docs/experiments/19_level_targeted_equations && OUT=/tmp/exp19A.json SYNTHETIC=1 MAX_TRIPLES=4 uv run --python 3.12 --with numpy --with torch --with h5netcdf python phaseA_diagnostics.py`
Expected: PASS — печатает `WROTE /tmp/exp19A.json`, JSON содержит `ekman_sweep`.

- [ ] **Step 4: ruff**

Run: `ruff check docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py --select F` (docs вне общего скоупа; проверяем только F-ошибки)
Expected: чисто (T201-print легитимны в docs-скриптах).

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py
git commit -m "feat(exp19): phase A Ekman friction residual diagnostics (self-contained)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: phaseA_diagnostics.py — r-пространственная невязка + CC-мост

**Files:**
- Modify: `docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py`

**Interfaces:**
- Consumes: `Era5Reader` читает `r` напрямую (ERA5 relative_humidity); `PurePDEKernel.rhs` даёт `q_t`, `t_t`.
- Produces: функция `r_space_residual(...)` — сравнивает r-невязку схемы `as_is` (q→r конверсия) против `cc_bridge` на уровнях 600–1000.

- [ ] **Step 1: Extend smoke expectation**

Тот же smoke-прогон должен дополнительно писать `r_residual: {as_is:[...], cc_bridge:[...]}`.

- [ ] **Step 2: Write implementation**

В `Era5Reader.state` уже читается `r` (VARS содержит relative_humidity). Сохранить `r` в state-словарь (сейчас конвертируется в q; добавить ключ `"r"`).
Добавить:
```python
def r_tendency_from_q(q_t, t_t, q, t, q_s):
    """as_is: r_t = d(q/q_s)/dt ≈ q_t/q_s − q·q_s_t/q_s². cc_bridge: r_t = q_t/q_s
    (T-зависимость снята мостом). q_s_t = q_s·(L/R_v T²)·t_t."""
    L, R_v = 2.5e6, 461.5
    q_s_t = q_s * (L / (R_v * t * t)) * t_t
    r_as_is = q_t / q_s - q * q_s_t / (q_s * q_s)
    r_cc = q_t / q_s
    return r_as_is, r_cc
```
Для якоря `C15_now`: obs r-тенденция = `(s1["r"] − s0["r"]) / DT_OBS / 100` (r в %, привести к доле). Аккумулировать residual_rel r по обеим схемам. Записать `r_residual`.
Критерий приёмки (печать): cc_bridge r-невязка < as_is на индексах 8..12 (600–1000).

- [ ] **Step 3: Run smoke**

Run: `cd docs/experiments/19_level_targeted_equations && OUT=/tmp/exp19A.json SYNTHETIC=1 MAX_TRIPLES=4 uv run --python 3.12 --with numpy --with torch --with h5netcdf python phaseA_diagnostics.py`
Expected: PASS — JSON содержит `ekman_sweep` И `r_residual`.

- [ ] **Step 4: ruff**

Run: `ruff check docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py --select F`
Expected: чисто.

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/19_level_targeted_equations/phaseA_diagnostics.py
git commit -m "feat(exp19): phase A r-space residual + CC-bridge diagnostics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Прогон фазы A на кластере + пороги отсечки + решение-гейт

**Files:**
- Create: `docs/experiments/19_level_targeted_equations/results/phaseA_usa_2000.json`, `phaseA_usa_2004.json` (артефакты прогона, НЕ коммитить если >их обычного размера JSON — эти малые, коммитить)
- Create: `docs/experiments/19_level_targeted_equations/README.md` (раздел «Фаза A»)

**Interfaces:**
- Consumes: `phaseA_diagnostics.py` (Tasks 7–8); готовый `18_.../results/eq18_usa_2004.json` для A-отсечки.
- Produces: таблица порогов маски (u,v из невязки; t,q — заявлены как гиперпараметр), вердикты приёма A-Ekman/A-CCмост, K-профиль для конфига B2.

- [ ] **Step 1: Запустить на кластере (реальный ERA5, 2000 + 2004)**

```bash
ssh cluster 'cd ~/wt_fix_v2/docs/experiments/19_level_targeted_equations && \
  sbatch -A proj_1715 -p cpu-e-quick --wrap "OUT=results/phaseA_usa_2004.json YEAR=2004 STRIDE=1 REPO_ROOT=~/wt_fix_v2 python phaseA_diagnostics.py"'
```
(и аналогично YEAR=2000). Дождаться завершения (`~10 мин`, как exp18).

- [ ] **Step 2: A-отсечки из eq18**

Извлечь пороги: последний уровень с `residual_rel ≲ 1` по u,v из `18_.../results/eq18_usa_2004.json` (арм C15_now). Известно из exp18: **u,v → 700 гПа** (индекс 9). Записать таблицу порогов в README.

- [ ] **Step 3: Вердикты приёма**

Прочитать `phaseA_usa_2004.json`: `ekman_sweep` (принят если ≥20% падения на 925–1000 без роста ≤700 >5%) и `r_residual` (принят если cc_bridge < as_is на 600–1000). Записать оба вердикта + выбранный K-профиль в README «Фаза A».

- [ ] **Step 4: Decision gate (документируется, не код)**

- B1, B1L — запускаются в фазе B **безусловно** (маска невидима в невязке).
- B2 (Ekman) — в фазу B **только если A-Ekman принят** (Step 3).
- B3 (CC-bridge) — в фазу B **только если A-CCмост принят** (Step 3).
Зафиксировать решение по каждому арму в README.

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/19_level_targeted_equations/results/phaseA_usa_2000.json \
        docs/experiments/19_level_targeted_equations/results/phaseA_usa_2004.json \
        docs/experiments/19_level_targeted_equations/README.md
git commit -m "docs(exp19): phase A results — Ekman/CC verdicts + mask thresholds

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## PHASE B — Конфиги и обучение (база R5, t=12, 90 эпох, сид 0)

### Task 10: Конфиги abl19-b1 (жёсткая маска) и abl19-b1l (обучаемый гейт)

**Files:**
- Create: `configs/abl19/abl19_b1_mask_hard_t12.yaml`
- Create: `configs/abl19/abl19_b1l_mask_learnable_t12.yaml`

**Interfaces:**
- Consumes: `configs/abl16_long/abl16_r5_exp15_t12.yaml` как базу (копия + добавление флагов маски).
- Produces: два конфига с `physics_level_mask`/`physics_level_mask_learnable`.

- [ ] **Step 1: Создать abl19_b1 (жёсткая маска)**

Скопировать `configs/abl16_long/abl16_r5_exp15_t12.yaml`, изменить `experiment.name: abl19-b1-mask-hard-t12-s0`, в `model.params` добавить (пороги: u,v из A-отсечки = 700; t,q — гиперпараметр = 850):
```yaml
    physics_level_mask:
      u: 700
      v: 700
      t: 850
      q: 850
    physics_level_mask_learnable: false
```
(база R5 уже имеет `static_graph: false`, `find_unused_parameters: true` — DDP-совместимость сохранена.)

- [ ] **Step 2: Создать abl19_b1l (обучаемый гейт)**

Копия abl19_b1, `experiment.name: abl19-b1l-mask-learnable-t12-s0`, `physics_level_mask_learnable: true` (тот же `physics_level_mask` как init).

- [ ] **Step 3: Локальная валидация конфигов (парсинг + сборка модели smoke)**

Run: `uv run --python 3.12 --with pyyaml python -c "import yaml; [print(yaml.safe_load(open(f))['experiment']['name']) for f in ['configs/abl19/abl19_b1_mask_hard_t12.yaml','configs/abl19/abl19_b1l_mask_learnable_t12.yaml']]"`
Expected: печатает оба имени, без ошибок парсинга.

- [ ] **Step 4: Commit**

```bash
git add configs/abl19/abl19_b1_mask_hard_t12.yaml configs/abl19/abl19_b1l_mask_learnable_t12.yaml
git commit -m "feat(exp19): configs B1 (hard mask) and B1L (learnable gate) on R5 base

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Конфиги abl19-b2 (Ekman) и abl19-b3 (CC-bridge) — только для принятых A

**Files:**
- Create: `configs/abl19/abl19_b2_ekman_t12.yaml` (только если A-Ekman принят)
- Create: `configs/abl19/abl19_b3_ccbridge_t12.yaml` (только если A-CCмост принят)

**Interfaces:**
- Consumes: K-профиль из Task 9 Step 3 (для B2); база R5 t12.
- Produces: конфиги с `physics_ekman_K_profile` / `physics_humidity_evolution: cc_bridge`.

- [ ] **Step 1: Проверить гейт (Task 9 Step 4)**

Если A-Ekman отклонён — B2 не создаётся; зафиксировать в README «отклонён измерением, выключить лучше чинить». Аналогично B3. Создавать только принятые.

- [ ] **Step 2: Создать abl19_b2 (если Ekman принят)**

Копия abl19_b1 (маска u,v ≤700 сохраняется — Экман работает В погранслое, где маска и так режет прогностику; но Экман добавляется к u_t ДО маски, поэтому даёт вклад). Уточнение: для чистой изоляции эффекта Экмана B2 — БЕЗ маски (`physics_level_mask` убрать), `experiment.name: abl19-b2-ekman-t12-s0`, добавить:
```yaml
    physics_ekman_K_profile: [<13 значений K из Task 9 Step 3>]
```

- [ ] **Step 3: Создать abl19_b3 (если CC принят)**

Копия базы R5, `experiment.name: abl19-b3-ccbridge-t12-s0`, добавить:
```yaml
    physics_humidity_evolution: cc_bridge
```

- [ ] **Step 4: Валидация парсинга (тех, что созданы)**

Run: `uv run --python 3.12 --with pyyaml python -c "import yaml,glob; [print(yaml.safe_load(open(f))['experiment']['name']) for f in glob.glob('configs/abl19/abl19_b[23]*.yaml')]"`
Expected: печатает имена созданных конфигов.

- [ ] **Step 5: Commit**

```bash
git add configs/abl19/abl19_b2_ekman_t12.yaml configs/abl19/abl19_b3_ccbridge_t12.yaml
git commit -m "feat(exp19): configs B2 (Ekman) and B3 (CC-bridge) for accepted candidates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Обучение (job-array), rollout, level-profile eval, отчёт README

**Files:**
- Create: `sh_files/abl19_train_array.sh`, `sh_files/abl19_rollout_t12_array.sh`
- Modify: `docs/experiments/19_level_targeted_equations/README.md` (раздел «Фаза B» + синтез)
- Modify: `docs/experiments/README.md` (строка индекса exp19)

**Interfaces:**
- Consumes: конфиги abl19 (Tasks 10–11); rollout/level_profile харнесс exp18 (`18_.../level_profile_analysis.py`).
- Produces: обученные армы, rollout npz, per-level фигуры/таблицы, финальный отчёт.

- [ ] **Step 1: Написать job-array обучения**

`sh_files/abl19_train_array.sh` — sbatch job-array по созданным конфигам abl19 (2–4 арма), `-A proj_1715`, GPU-партиция, тот же launcher, что exp16-long t12 (не трогая train-скрипты — только `--config`). Массив покрывает ровно созданные конфиги.

- [ ] **Step 2: Запустить обучение**

```bash
ssh cluster 'cd ~/wt_fix_v2 && sbatch -A proj_1715 sh_files/abl19_train_array.sh'
```
Дождаться эпохи 90 (общая с exp16 t12-лестницей для сравнимости; ~8–10ч как long-волна, но можно оценивать на промежуточной эпохе 90 как exp18).

- [ ] **Step 3: Rollout + level-profile**

Скопировать `18_.../abl16_rollout_t12_array.sh`-подход → `sh_files/abl19_rollout_t12_array.sh` (job-array rollout по армам abl19). Запустить, собрать npz. Прогнать `level_profile_analysis.py` exp18 с новыми армами (+ R0, R5 из exp16 как якоря).
Run (локально, после стягивания npz): `uv run --python 3.12 --with numpy --with matplotlib python docs/experiments/18_level_resolved_physics/level_profile_analysis.py` (или его exp19-адаптацию с abl19-путями).

- [ ] **Step 4: Написать отчёт**

В `19_.../README.md` раздел «Фаза B»: per-level скилл B1/B1L (и B2/B3 если были) против R0/R5; прошла ли погранслойная деградация к ~0; сохранились ли верхние выигрыши R5; выученный профиль α(p) для B1L (совпал ли с residual-порогом u,v=700 и скилловым t/q). Синтез: подтверждает/опровергает принцип exp18.
Добавить строку exp19 в `docs/experiments/README.md` (таблица «Карта подпапок»).

- [ ] **Step 5: Commit**

```bash
git add sh_files/abl19_train_array.sh sh_files/abl19_rollout_t12_array.sh \
        docs/experiments/19_level_targeted_equations/README.md docs/experiments/README.md
git commit -m "docs(exp19): phase B training/rollout scripts, level-profile report, index

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (проверка плана против спеки)

**Spec coverage:**
- §3.1 маска (жёсткая B1 / обучаемая B1L, z исключён) → Tasks 1, 2, 3, 10. ✓
- §3.2 Экман → Tasks 4, 6 (код), 7 (диагностика), 11 (конфиг). ✓
- §3.3 CC-мост → Tasks 5, 6 (код), 8 (диагностика), 11 (конфиг). ✓
- §4 фаза A (Ekman/CCмост/отсечки, 2000+2004, критерии приёма) → Tasks 7, 8, 9. ✓
- §5 фаза B (база R5 t12 90эп s0, B1/B1L безусловно, B2/B3 гейт, DDP static_graph) → Tasks 10, 11, 12; DDP уже в базе R5. ✓
- §6 критерии успеха → Task 12 Step 4. ✓
- §7 границы (opt-in дефолт бит-в-бит, LBYL, ruff, папка exp19) → Global Constraints + пины golden в Tasks 1–6. ✓

**Placeholder scan:** K-профиль в Task 11 берётся из Task 9 Step 3 (конкретный выход прогона, не TBD); пороги t/q явно помечены гиперпараметром. Нет «add error handling»/«similar to». ✓

**Type consistency:** `physics_level_mask: dict[str,float]|None`, `physics_level_mask_learnable: bool`, `ekman_K_profile: tuple|None`, `humidity_evolution: str` — единообразны в Tasks 1–6; IAM4VP-имена с префиксом `physics_`. `_mask_increment(inc, var)`, `_saturation_specific_humidity(t)` — согласованы. ✓
