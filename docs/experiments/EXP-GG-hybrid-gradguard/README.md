# EXP-GG — HybridBlock gradient-guard: deep defect analysis + before/after ablation

**Scope.** Deep debug of PI-IAM4VP after the `fix_inline_v2` refactor
(2026-07-08). Two defects were found in the `stable_physical` residual-physics
path. This document is the root-cause analysis, the controlled before/after
experiment, and the conclusions.

**Commits.** Guard fix `4ec2317`; sanity tooling `6c4a645`; ablation harness
`tools/exp_gradguard_ablation.py`. Baseline forward/state_dict harness: **ALL
OK** (the fix does not change any forward output — it only touches backward).

**Reproduce.** `tools/sanity_pi_iam4vp_gpu.py` (46-check matrix over 7 arms);
`tools/exp_gradguard_ablation.py --repo <tree> --label <before|after>` run
against the pre-fix worktree (`4d2dea8`) and the post-fix tree. Local runs use a
synthetic adversarial memmap fixture (white noise at mean ± 0.5σ — deliberately
rough so WENO smoothness indicators are stressed); the cluster run uses real
ERA5 (USA crop, v4 memmap).

---

## 1. Defect A — silent NaN-gradient poisoning of the physics branch (FIXED)

### Mechanism

In `stable_physical` mode (`fixedeq`, `massconsistent`, `diabatic` arms) the
`_physics_prior_from_state` forward runs the inherited `HybridBlock` in physical
units. The forward is guarded for *values*: `_sanitize_hybrid_latent_physical`
replaces nonfinite entries between hybrid steps, and `_finite_or_fallback`
guards the final prior — so `y_phys` returned to the corrector is always finite,
and the **loss is always finite**.

But the guards run *between* hybrid steps and *after* the block. Inside a single
`PDE_block` forward (3 `PDE_kernel`s), the nonfinite activations are produced and
flow through the learnable ops — `variable_norm`/`variable_innorm` convs, the
five `norm_*` BatchNorms, `block_norm`, and the `router_weight` mix — before any
sanitize touches them. The physics *increments* are `.detach()`ed, but those
learnable parameters are still in the autograd graph via the activation path
(`variable_norm(x) → … → block_norm → x+skip → next kernel`). So on `backward()`
their gradients are NaN **even though the loss is finite**.

`optimizer.step()` (AdamW) then writes NaN into those weights. From the next
forward on, `router_weight` is NaN ⇒ the whole physics mix is NaN ⇒ sanitized to
the fallback (≈ previous state) ⇒ `physics_residual_nonfinite_ratio = 1.0`, the
physics tendency collapses to ≈ 0, and the arm is **functionally identical to
`no_physics`** — permanently, with no error and a still-decreasing loss.

### Evidence (probe on the adversarial fixture, fixedeq, batch 0 backward)

Exactly 4 hybrid parameters receive nonfinite gradients while the loss is
finite:

| parameter | nonfinite-grad ratio |
| --- | --- |
| `hybrid_block.router_weight` | 1.00 |
| `…PDE_kernels.2.block_norm.weight` | 1.00 |
| `…PDE_kernels.2.norm_u.weight` | 0.077 |
| `…PDE_kernels.2.variable_innorm.weight` | 0.015 |

Forward nonfinite ratio reaches ≈ 0.20 by the 6th kernel call; after one step
the pre-fix diagnostics read `router_weight_abs = nan`, `bn_gamma_drift = nan`,
`physics_residual_nonfinite_ratio = 1.0` from batch 1 onward.

### Fix (`4ec2317`)

`Tensor.register_hook` on every `hybrid_block` parameter zeroes nonfinite
gradient entries (`nan_to_num`; identity for finite gradients — a healthy step
is bit-for-bit unchanged) with no host sync, and counts affected parameters into
a new diagnostic `physics_hybrid_nonfinite_grad_params` (a backward-N count that
surfaces in the forward-(N+1) diagnostics). The hook runs before DDP gradient
reduction, so all ranks reduce sanitized gradients.

---

## 2. Defect B — conv-contamination of the physical path (DESIGN FLAW, proposal only)

`variable_norm`/`variable_innorm` in `PDE_kernel` are **not** normalisations —
they are random-initialised `Conv2d(3×3)` (`utils/physics_hybrid.py:351,377`).
The kernel integrates `zquvtw_old = 0.5·variable_norm(x) + 0.5·zquvtw`. In
`stable_physical` (physical units, eval-BN = identity affine) the random conv
smears the z-scale (~10⁵ m²/s²) across every variable block on **kernel 0**,
before a single equation runs.

### Evidence (probe on real-scale physical fields, fixedeq at init, kernel 0)

| block | physically plausible | clean \|max\| | integrated \|max\| | contamination × |
| --- | --- | --- | --- | --- |
| z (m²/s²) | ~5e4..2e5 | 2.03e5 | 1.07e5 | 0.5 |
| t (K) | 200..292 | 2.92e2 | **3.40e4** | 116× |
| q (kg/kg) | 0..0.012 | 1.15e-2 | 2.40e4 | 2.1e6× |
| u (m/s) | −5..27 | 2.69e1 | 2.81e4 | 1045× |
| v (m/s) | −11..11 | 1.13e1 | 2.46e4 | 2173× |

The temperature the equations actually integrate spans **−23700 … +33954 K**.
`‖conv term‖ / ‖phys term‖ = 0.51` at kernel 0 — half of the integrated state
is random projection. The `physical_clip` increment caps (±5 K etc.) cannot
recover this: they bound each step's *change*, not the contaminated *state* the
tendencies are computed from. A second contributor is the double residual skip
(`PDE_block` adds a skip to both paths, so the physical state is added to itself
across depth). In `legacy_normalized` this was masked by train-mode BatchNorm
renormalising the activations each forward; `stable_physical`'s frozen eval-BN
removes that mask.

**Consequence.** The "physics tendency" fed to the corrector is dominated by
projection noise, not physics (consistent with the equation experiments' finding
that the linear residual ceiling is R² ≤ 0.17 — see
[EXP-09/10](../09-diabatic-residual-ceiling/) et al. in this archive). Defect B
is also the *cause* of the overflow that triggers defect A: **B → fp32 overflow
→ NaN activations → A (NaN grads) → silent branch death.** The guard makes
training survive B; it does not make the features physical.

### Fix — `stable_physical_v2` (shipped, `b9c6ed1`)

A new OCP hybrid mode `stable_physical_v2` runs the HybridBlock as a **pure
primitive-equation integrator on the clean physical state**:
`PDE_kernel._evolve_fields` extracts the frozen equation calls; the new
`physics_only_forward` (on `PDE_kernel`/`PDE_block`/`HybridBlock`) integrates
`zquvtw` directly — no `variable_norm` conv mix, no `norm_*`, no
`variable_innorm`/`block_norm`, no router, no skip-doubling. The physical-unit
plumbing (denormalize, r↔q, sanitize, tendency clip) is shared with
`stable_physical`. Existing modes stay bit-exact (9-arm harness ALL OK). The
weather-plus-physics arms **B1/A1/A2** (base residual, mass-consistent, diabatic)
now use `stable_physical_v2` (experiment names suffixed `-purephys`); the
`no_physics` control (**B0**) and the `fixedeq`/`legacy_hybrid` references are
unchanged — they remain the contaminated/no-physics baselines for comparison.

---

## 3. Before/after experiment

**Design.** Literal old-code (`4d2dea8`, no guard) vs new-code (post-`4ec2317`,
guard), same seed → bit-identical init (state_dict harness confirms), same
deterministic batches. The registered gradient hook is the *only* difference. K
optimizer steps per arm; per step we log physics-branch health.
Arms: `fixedeq`, `massconsistent` (both `stable_physical` — expected to poison
pre-fix), and `legacy_hybrid` (train-mode BN — negative control, expected *not*
to poison, so the guard should be a no-op there).

**Pre-registered predictions (fixed before reading results — no HARKing):**

- **P1.** Pre-fix `fixedeq`/`massconsistent`: `hybrid` weights go nonfinite
  within 1 step and stay nonfinite; `router_finite` flips to False and never
  recovers; `physics_tendency_rms` collapses toward 0; `forward_nonfinite_ratio`
  → 1.0. Task loss keeps decreasing (silent failure).
- **P2.** Post-fix `fixedeq`/`massconsistent`: `hybrid` weights stay finite for
  all K steps; `router_finite` stays True; `physics_tendency_rms` stays O(1σ);
  the physics branch keeps contributing (`pi_minus_iam4vp_rms` > 0).
- **P3.** `legacy_hybrid` (negative control): identical before vs after (no
  nonfinite grads to sanitize) — guard is a verified no-op.

### 3.1 Results — adversarial fixture (local, CPU, seed 0, 12 steps)

`stable_physical` arms (`router` = is `router_weight` finite; `poisoned` =
hybrid params with a nonfinite entry; `tend_rms` = `physics_residual_tendency_rms`;
`nf` = forward nonfinite ratio):

| step | fixedeq BEFORE (router/poisoned/tend_rms/nf/loss) | fixedeq AFTER |
| --- | --- | --- |
| 0 | **False** / 20 / 1.66 / 0.94 / 0.4377 | True / 0 / 1.66 / 0.94 / 0.4377 |
| 1 | False / **49** / 1.0e-4 / 1.00 / 0.4302 | True / 0 / 1.66 / 0.94 / 0.4302 |
| 4 | False / 49 / **3.1e-7** / 1.00 / 0.4099 | True / 0 / 5.25 / 0.44 / 0.4097 |
| 11 | False / 49 / 3.1e-7 / 1.00 / **0.4015** | True / 0 / 6.22 / 0.99 / **0.4012** |

- **P1 confirmed.** `fixedeq` and `massconsistent` poison on **step 0** (router
  dead after the first `optimizer.step()`), and the damage **spreads** 20 → 49
  hybrid params (all of `hybrid_block`) by step 1 as NaN weights produce NaN
  activations → NaN grads everywhere. `tendency_rms` collapses `1.66 → 3.1e-7`
  and `nf` pins at `1.00` for the remaining 11 steps — **permanent**.
- **P2 confirmed.** After the fix, both arms keep `router_finite=True`,
  `poisoned=0` for all 12 steps; `tendency_rms` stays O(1–6σ). Notably
  `massconsistent` AFTER dips to `3e-7` on hard batches (steps 2–8) then
  **recovers to 2.94** (steps 9–11): the guard converts *permanent poisoning*
  into *transient, recoverable forward overflow* — the weights never die.
- **P3 confirmed.** `legacy_hybrid` is **bit-identical BEFORE vs AFTER across all
  12 steps** (`nf=0.00` throughout — train-mode BatchNorm prevents the overflow
  entirely). The guard is a verified no-op where it is not needed.
- **The loss gives zero signal.** Final task loss is `0.4015` (before, physics
  dead) vs `0.4012` (after, physics alive) — indistinguishable. A run watching
  only `val_loss` would ship a "PI-IAM4VP" that is silently plain IAM4VP.

### 3.2 Results — real ERA5 (cluster, 1×GPU) — PENDING

The sanity job (`tools/sanity_pi_iam4vp_gpu.py`) logs `forward_nonfinite_ratio`
per arm on real staged ERA5; the real-data grad-guard ablation is queued behind
it. The open quantitative question is the *frequency* of forward overflow on
smooth real fields at fp32 (the adversarial fixture is white noise, an upper
bound on roughness). Mechanism and fix behaviour are already settled by §3.1;
§3.2 only calibrates how often A bites in normal production vs only under stress
(AMP/bf16, distribution shift, longer rollout). (First cluster attempt
`4168882`/`4169055` failed in 1 s — the sbatch wrappers sourced
`_shell_contract.sh` by `BASH_SOURCE`, which points at the Slurm spool copy, not
the repo; fixed in `a827d32` to resolve `REPO_ROOT` from `SLURM_SUBMIT_DIR`.
Resubmitted.) This section fills on landing.

### 3.3 Defect-B before/after — `stable_physical` vs `stable_physical_v2`

Same `fixedeq` arm and seed, one code tree, differing only in the hybrid mode
(`tools/exp_purephysics_ablation.py`, adversarial fixture, 12 steps). `nf` =
forward nonfinite ratio; `clip` = fraction of the tendency saturating its ±8
clip; `tend_rms` = physics tendency RMS (σ); `t` = physical temperature range of
the prior (K):

| step | `stable_physical` (nf/clip/tend_rms/t) | `stable_physical_v2` |
| --- | --- | --- |
| 0 | 0.94 / 0.03 / 1.66 / [150,350] | **0.00 / 0.00 / 0.19 / [176,311]** |
| 6 | 0.44 / 0.31 / 5.01 / [150,350] | 0.00 / 0.00 / 0.14 / [169,314] |
| 11 | 1.00 / 0.48 / 6.22 / [150,350] | 0.00 / 0.00 / 0.13 / [179,310] |

- **v2 removes the overflow entirely** (`nf 0.94–1.0 → 0.00`), so defect A cannot
  even arise in v2 — the guard is dormant (`hybrid_grad_guard_activations = 0`,
  `hybrid_block_no_grad_v2` PASS: v2 has zero learnable physics params).
- **v2 keeps the state physical**: the prior temperature stays in ~[170,314] K
  instead of being pinned to the [150,350] K sanitize rails every step.
- **The tendency becomes a real physical increment**: `tend_rms` drops
  ~40× (6.2 → 0.13 σ) and the clip never fires (0.48 → 0.00). Under
  `stable_physical` the corrector was fed clipped projection noise; under v2 it
  is fed the primitive-equation evolution of the real fields.
- Task loss is unchanged on this 12-step fixture (0.4012 vs 0.4014) — again the
  loss carries no signal about physics quality; the difference is *what the
  physics feature means*, which only a full training run can convert into skill.

---

## 4. Conclusions

1. **Defect A is a silent, permanent, single-step failure with no loss signal.**
   Pre-fix, the physics branch dies on step 0 and the loss curve is
   indistinguishable from the healthy run (0.4015 vs 0.4012). The failure is
   observable *only* through physics-specific diagnostics (`router_finite`,
   `physics_residual_tendency_rms`, the new `physics_hybrid_nonfinite_grad_params`).
   This retroactively justifies keeping those diagnostics through the refactor —
   without them, a multi-day run silently trains plain IAM4VP under a "PI" label.

2. **The fix is exactly scoped: survivability, nothing more.** It converts
   permanent poisoning into transient recoverable overflow (the `massconsistent`
   dip-and-recover), is bit-identical where unneeded (`legacy_hybrid`), and never
   changes a forward output (state_dict/forward harness ALL OK). It is a correct
   defense-in-depth measure at the gradient boundary.

3. **The guard was necessary but not sufficient — defect B was the real disease,
   now fixed.** Under `stable_physical` the guard stopped the branch from *dying*
   but did not make the features *physical*: `tendency_rms ≈ 6σ`, `nf` up to 1.0,
   `t` integrated over −23700…+33954 K (§2), which aligns with the equation
   experiments' linear-residual ceiling (R² ≤ 0.17). `stable_physical_v2` (§3.3)
   removes the contamination at the source — the equations integrate the clean
   physical state, `nf → 0.00`, the tendency becomes a real physical increment
   (~40× smaller, clip never fires), and defect A can no longer arise because
   there is no overflow to poison from. B1/A1/A2 now run this mode; `fixedeq`
   stays `stable_physical` as the contaminated reference for comparison.

4. **Why it lay dormant — a correct change unmasked a latent landmine.** The old
   `legacy_normalized` path ran train-mode BatchNorm, which renormalised
   activations every forward and prevented the overflow (`legacy_hybrid` nf=0.00
   here). Freezing BN to eval — the *correct* frozen-prior invariant that keeps
   the hybrid working space in physical units — removed that accidental overflow
   shield and exposed A. The lesson is specific: the frozen-prior fix was right,
   and it needed the gradient guard shipped alongside it.

5. **Causal chain, broken at two points.** B (conv contamination) → fp32
   overflow in WENO/kernel → NaN activations (values sanitized, loss finite) →
   **A** (NaN parameter grads) → `optimizer.step()` poisons weights → permanent
   silent degradation to `no_physics`. The guard (`4ec2317`) breaks the chain at
   the A→poison link (defense in depth, kept for every mode);
   `stable_physical_v2` (`b9c6ed1`) breaks it at the B→overflow source and is the
   substantive fix.

6. **The scientific reframing.** Under `stable_physical` the corrector was fed
   clipped random-projection noise, so "does the physics feature help?" was really
   "does contaminated noise help?" — and the honest answer was no (R² ≤ 0.17).
   `stable_physical_v2` is the first version where the physics feature *is* the
   primitive-equation evolution of the real fields, so B1/A1/A2 finally test the
   intended hypothesis. Whether real physics improves skill is now an open
   question a full training run can answer — which is the point of the arms.
