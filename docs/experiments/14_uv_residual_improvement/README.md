# Эксперимент 14: снижение невязки уравнений движения (u, v)

**Вопрос.** Почему относительная невязка (residual) уравнений движения на
ERA5-2000 была 3.9–4.3 для u и 10.2–12.0 для v (эксперимент 13) — хуже
нулевого прогноза в разы — и какие члены уравнений нужно исправить или
добавить, чтобы снизить её кратно на USA-кропе и глобусе за 2000–2002 гг.,
не ухудшив уравнения T, q, z.

**Ответ.** Основной источник — не физика, а ориентация оси широты: данные
WeatherBench упорядочены юг→север, а FD-4-производная `d_y` диагностической
библиотеки предполагает север→юг и возвращала −∂/∂y. Перевёрнутый
меридиональный барический градиент ломал геострофический баланс в v-уравнении
(ошибка ≈ 2·f·u). Вторая геометрическая ошибка — широты `Grid` не совпадали с
широтами строк данных (USA-кроп сдвинут на 1.5 ячейки, глобальная сетка имеет
шаг 1.3953° вместо 1.40625°). Поверх исправленной геометрии добавлены члены
кривизны сферы, метрический член континуити, трение Хелда–Суареса и
массо-согласованная ω. Итоги, числа и разбор гипотез — в
[UV_RESIDUAL_2000_2002.md](UV_RESIDUAL_2000_2002.md).

## Файлы

| Файл | Что делает |
|---|---|
| [UV_RESIDUAL_2000_2002.md](UV_RESIDUAL_2000_2002.md) | Отчёт: суть, журнал прогонов, таблицы, разбор гипотез, фигуры |
| [physics_stats_uv.py](physics_stats_uv.py) | Матрица вариантов уравнений: невязки, разложение по членам/уровням/широтам, карты |
| [make_figures.py](make_figures.py) | Рендер фигур из results-JSON и NPZ |
| `results/physics_stats_uv_{usa,globe}_{2000,2001,2002}.json` | Результаты прогонов (все часовые тройки года) |
| `results/physics_maps_uv_{usa,globe}_2000.npz` | Карты накопленной невязки base/C_best + lat/lon/lsm |
| `fig*.png` | Фигуры |

## Воспроизведение

```bash
# на кластере (через remote_submit с локальной машины):
YEAR=2000 DOMAIN=globe STRIDE=1 bash sh_files/remote_submit.sh sh_files/physics_stats_uv_cpu.sh
# локальный smoke без данных:
OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=2 python docs/experiments/14_uv_residual_improvement/physics_stats_uv.py
# фигуры (после скачивания results/):
python docs/experiments/14_uv_residual_improvement/make_figures.py
```

Новые члены уравнений живут в `utils/physics.py::PurePDEKernel` за явными
параметрами (`rows_south_to_north`, `metric_terms`, `spherical_divergence`,
`vertical_scheme`, `rayleigh_friction`) и в `GridConfig.latitudes_deg`;
дефолтное поведение бит-в-бит, запинено
`tests/test_physics_sign_conventions.py::Exp14MomentumVariants`.
