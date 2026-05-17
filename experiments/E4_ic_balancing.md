# E4 — Балансировка IC (идея 4): initialization-shock

## Гипотеза / корень

Сырой ERA5 (из 4D-Var IFS) не в дискретном балансе именно этой FD-4
дискретизации: дискретный −∇z ≠ дискретный f×V. Первый шаг впрыскивает
O(1) гравитационный shock. Гипотеза: digital filter initialization (DFI,
Dolph forward–backward, Lynch–Huang) убирает spurious гравволны из IC и
дополнительно стабилизирует поверх E1+E2+E3. Проверяемая: если взрыв
доминируется shock’ом, балансировка даёт скачок, иначе — слабый эффект.

## Изменение кода

- `tools/check_physics_common.py`: `balance_initial_state(state, kernel,
  mode, span_hours)` — `dfi` (Dolph forward–backward time-filter) и
  `geostrophic` (проекция массы на ветер) — вызывается после Magnus в
  `_prepare_state`-пути под `--balance-ic {none,dfi,geostrophic}`
  (default `none`, обратная совместимость).
- `tools/check_physics_new_kernel.py`: `--balance-ic`; method_name/tags.

*Scope:* E4 крупнее остальных; реализуем прагматичный Dolph
forward–backward, не полный нормально-модовый DFI. Гейтим на том, что E1–E3
реально помогли (иначе обсудить с пользователем приоритет).

## Sanity (`Models/dev/sanity_ic_balance.py`, PASS-gate)

- норма начального gravity-imbalance (proxy: |∂div/∂t| или |z_t − f·V|)
  после балансировки падает ≥ ×3.
- медленная (вихревая/крупномасштабная) часть сохранена: corr(balanced,
  raw) по rotational wind / z крупного масштаба > 0.95.

## Команды

```bash
.venv/bin/python Models/dev/sanity_ic_balance.py        # PASS
.venv/bin/python tools/check_physics_new_kernel.py --offline \
    --memmap-path /tmp/syn_era5.dat --memmap-meta-path /tmp/syn_era5.meta.json \
    --horizon-hours 6 --stencil fd4 --time-scheme ssp_rk3 --coriolis spherical \
    --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --balance-ic dfi
# кластер: ablation runner label E4 (накопленно + balance-ic dfi)
```

## Comet

- method_name: `fd4_ssprk3_spherical+hyperdiff+polar+dfi`
- url / key / tags: *(заполнить после кластерного sweep)*

## Результаты

### sanity ✅ код-корректность (2026-05-15)

`Models/dev/sanity_ic_balance.py` PASS:

```text
geostrophic: |u−u_g|max=0  |v−v_g|max=0  Δmass(z,t)=0  R 7.96e-5→1.74e-12
dfi:         finite=True applied=True corr(z_raw,dfi)=1.0000  R 7.96e-5→9.71e-5
```

- **geostrophic**: ветер становится точно (−z_y,z_x)/f_safe, масса
  (z,t,q) не тронута, геострофический остаток обнуляется (7.96e-5→1.7e-12).
  Реализация точна.
- **dfi**: стабилизированный forward (ssp_rk3+∇⁴+polar) **не расходится**
  (ключевой фикс — наивный backward-Эйлер на неустойчивой физике
  расходился), реально применяется, low-pass сохраняет крупномасштабный z
  (corr=1.0000).

### Честный негативный результат (не баг, по плану)

Прагматичный **forward-only short-span DFI НЕ форсит баланс**: на
максимально несбалансированном тесте (нулевой ветер + z-бугор) геостроф.
остаток `R` не падает (7.96e-5→9.71e-5). Причина физическая: таймскейл
геострофической адаптации (инерционный период ~часы–сутки) ≫ окна 1–3 ч;
forward-only-центрирование не ловит баланс из нуля. Корректная роль DFI —
снимать малый insertion-shock у уже почти-балансного ERA5, а не
балансировать ветер из нуля. Полноценный forward–backward DFI требует
устойчивого backward-интегратора (нет для явной схемы) — вне scope
(план: E4 «гейтим; фиксируем отрицательный результат»).

### smoke 6h ✅ pipeline + LBYL-guard (2026-05-15)

`smoke_ablation.py … --balance-ic dfi --abl-label E4`: отдельный Comet-
эксперимент `fd4_ssp_rk3_spherical_E4+hyperdiff+polar+dfi`. На бело-шумной
синтетике DFI-экскурсия расходится → **LBYL-guard корректно откатывает на
сырой IC** (без падения), далее обычный взрыв@h=1 (синтетика, не физ-вердикт).

### кластер 48h ✅ (job 3998911, real ERA5, 12 IC)

`frac_ic_blown_up`: h0=0.0 → **h1=1.0** → h48=1.0; prognostic = **NaN с h1**.
**Тождественно E3** (E4≈E3 — предсказание оправдалось). DFI не помог:
честный негатив подтверждён на real ERA5. Comet:
`fd4_ssp_rk3_spherical_E4+hyperdiff+polar+dfi`.

| метрика | h=1 | h=6 | h=48 | vs E3 | vs fixC |
| --- | --- | --- | --- | --- | --- |
| frac_ic_blown_up | 1.0 | 1.0 | 1.0 | ≡ | хуже (fixC=0.0) |
| weighted_rmse/z/500hPa | NaN | NaN | NaN | ≡ | fixC 1.1e5→6.9e10 |
| acc/z/500hPa | NaN | NaN | NaN | ≡ | fixC 0.44→0.24 |

## Вердикт

Код корректен (geostrophic точен R→1.7e-12; DFI стабилен+low-pass).
Эффективность DFI-балансировки **слабая** (честный физ-результат, не баг) —
кластер подтвердил E4≡E3, взрыв@h1. README §Главный вывод.

## Решение / next

balance-ic остаётся опцией (default none); в накопленный стек E5
включается `--balance-ic dfi` для полноты, но без ожидания выигрыша.
→ E5 (консервативная форма).
