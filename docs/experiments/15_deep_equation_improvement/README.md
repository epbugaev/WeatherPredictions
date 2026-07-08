# Эксперимент 15: почленный аудит и глубокое улучшение уравнений ядра

**Вопрос.** Какие члены пяти прогностических уравнений ядра (u, v, T, q, z)
отсутствуют или сформулированы хуже эталона примитивных уравнений в
изобарических координатах, и насколько снижается невязка (residual) каждого
уравнения на ERA5 2000–2002 (USA-кроп и глобус) от добавления недостающей
физики — диабатики, источников влаги, баротропного якоря гидростатики,
поправки О'Брайена к ω и климатологических поправок, построенных на 2000 годе.

**Ответ.** Заполняется по результатам прогонов: см.
[EQUATIONS_2000_2002.md](EQUATIONS_2000_2002.md).

## Файлы

| Файл | Что делает |
|---|---|
| [EQUATIONS_2000_2002.md](EQUATIONS_2000_2002.md) | Отчёт: аудит, литобзор, журнал, таблицы, разбор гипотез, фигуры |
| [physics_stats_eq.py](physics_stats_eq.py) | Матрица вариантов всех пяти уравнений: невязки, разложения, климатологии |
| [make_figures.py](make_figures.py) | Рендер фигур из results-JSON и NPZ |
| `results/eq15_*_{usa,globe}_{2000,2001,2002}_*.json` | Результаты прогонов |
| `results/eq15_*_maps_*.npz` | Карты накопленной невязки по вариантам |
| `fig*.png` | Фигуры |

## Варианты уравнений (все — opt-in параметры `PurePDEKernel`)

| Вариант | Параметр ядра | Физика | Источник |
|---|---|---|---|
| `T_hs` | `newtonian_relaxation` | −k_T(φ,σ)(T−T_eq) — ньютоновская релаксация | Held & Suarez (1994) |
| `T_lh` | `latent_heating_coupling` | +(L/c_p)·(−dq/dt\|cond) — скрытое тепло конденсации | Yanai et al. (1973) |
| `W_obrien` | `w_diagnostic='obrien'` | квадратичная поправка ω, w(p_top)=0 | O'Brien (1970) |
| `Z_ps` | `z_anchor='kinematic_ps'` | баротропный якорь R_d·T_s/p_s·∂p_s/∂t | IFS Cy47r1, ур. (2.8) |
| `S1_zm`/`S2_map`/`S12` | `rhs(sources=...)` | климатологии невязки (Q₁/Q₂), построены ТОЛЬКО на 2000 | Yanai et al. (1973) — методология |

## Воспроизведение

```bash
# раунд 1 (2000, построение климатологий):
REMOTE_USER=fa.buzaev bash sh_files/remote_submit.sh sh_files/physics_stats_eq15_cpu.sh \
  docs/experiments/15_deep_equation_improvement/physics_stats_eq.py 2000 globe 1 0 r1 \
  /home/fa.buzaev/WeatherPredictions/logs/eq15_clim_globe_2000.npz -
# раунд 2 (2001/2002, вневыборочная оценка климатологий):
REMOTE_USER=fa.buzaev bash sh_files/remote_submit.sh sh_files/physics_stats_eq15_cpu.sh \
  docs/experiments/15_deep_equation_improvement/physics_stats_eq.py 2001 globe 1 0 r2 - \
  /home/fa.buzaev/WeatherPredictions/logs/eq15_clim_globe_2000.npz
# локальный smoke без данных:
OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=4 python docs/experiments/15_deep_equation_improvement/physics_stats_eq.py
# фигуры (после скачивания results/):
python docs/experiments/15_deep_equation_improvement/make_figures.py
```

Дефолтное поведение ядра бит-в-бит прежнее; новые члены запинены
`tests/test_physics_sign_conventions.py::Exp15EquationVariants`.
