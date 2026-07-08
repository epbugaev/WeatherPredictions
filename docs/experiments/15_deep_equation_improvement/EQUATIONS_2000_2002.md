# Эксперимент 15: почленный аудит и улучшение уравнений ядра (ERA5 2000–2002)

Статус: черновик, числа заполняются из JSON прогонов.

## 1. Суть эксперимента

(заполняется после прогонов)

## 2. Аудит: эталонная система против кода

### 2.1. Эталон

Эталон — гидростатические примитивные уравнения в изобарических координатах на
сфере. Формулировка сверена с документацией оперативной модели ECMWF: IFS
Documentation Cy47r1, Part III «Dynamics and Numerical Procedures», гл. 2
«Basic equations and discretisation», ур. (2.1)–(2.10) (гибридная координата
η; на чистых уровнях давления η-члены переходят в p-форму). В обозначениях
кода (u, v — ветер, T — температура, q — влажность, Φ — геопотенциал,
ω = dp/dt):

Импульс (IFS ур. 2.1–2.2, форма на сфере):

$$\frac{\partial u}{\partial t} = -\mathbf{V}\cdot\nabla u - \omega\frac{\partial u}{\partial p} + fv + \frac{u v \tan\varphi}{a} - \frac{1}{a\cos\varphi}\frac{\partial \Phi}{\partial \lambda} + P_u$$

$$\frac{\partial v}{\partial t} = -\mathbf{V}\cdot\nabla v - \omega\frac{\partial v}{\partial p} - fu - \frac{u^2 \tan\varphi}{a} - \frac{1}{a}\frac{\partial \Phi}{\partial \varphi} + P_v$$

Термодинамика (IFS ур. 2.3):

$$\frac{\partial T}{\partial t} = -\mathbf{V}\cdot\nabla T - \omega\frac{\partial T}{\partial p} + \frac{\kappa T_v \omega}{(1+(\delta-1)q)\,p} + P_T,\qquad \kappa = R_d/c_p$$

Влажность (IFS ур. 2.4):

$$\frac{\partial q}{\partial t} = -\mathbf{V}\cdot\nabla q - \omega\frac{\partial q}{\partial p} + P_q$$

Гидростатика (IFS ур. 2.6) и неразрывность с ω и тенденцией приземного
давления (IFS ур. 2.5, 2.7, 2.8):

$$\frac{\partial \Phi}{\partial p} = -\frac{R_d T_v}{p},\qquad \omega(p) = -\int_{0}^{p}\nabla\cdot\mathbf{V}\,dp',\qquad \frac{\partial p_s}{\partial t} = -\int_{0}^{p_s}\nabla\cdot\mathbf{V}\,dp'$$

P-члены — параметризованная физика (радиация, конвекция, конденсация,
пограничный слой, трение). T_v = T[1+{(R_v/R_d)−1}q] — виртуальная
температура.

### 2.2. Система «как в коде» (после эксперимента 14, конфигурация C_best)

`utils/physics.py::PurePDEKernel`, знаковая конвенция ζ = −p (d_z = −∂/∂p,
get_w = −ω в гПа/с), см. docs/equations.md. C_best: точные широты строк
данных, d_y = +∂/∂y (юг→север), f = 2Ω·sinφ, члены кривизны ±uv·tanφ/a,
сферическая дивергенция в континуити, трение Хелда–Суареса, ω
массо-согласованная.

### 2.3. Почленный diff против эталона

| Член эталона | В коде (C_best) | Расхождение | Действие эксперимента 15 |
|---|---|---|---|
| Адвекция во всех уравнениях | есть (advective form, FD-4) | — | — |
| Кориолис + кривизна | есть (exp 14) | — | — |
| Барический градиент −∇Φ | есть | эталон добавляет R_d·T_v·∇ln p — на уровнях постоянного p равен нулю | — |
| Энергоконверсия κT_vω/((1+(δ−1)q)p) | κTω/p — сухая T, без влажностного множителя | T_v−T ≤ 1,2 % (q ≤ 0,02); (δ−1)q ≤ 1,7 % | задокументировано; эффект O(1 %), ниже разрешающей способности эксперимента |
| P_T: диабатика (радиация, конвекция, LH) | НЕТ | главный недостающий источник T-бюджета | `T_hs` (Held–Suarez), `T_lh` (скрытое тепло конденсации), `S1/S2` (климатология Q₁) |
| P_q: испарение, ПС-перемешивание | только крупномасштабная конденсация | нет источника снизу | `S1/S2` (климатология Q₂); оценка Q₂ из невязки |
| P_u, P_v: трение/ГВС | рэлеевское трение (exp 14) | нет орографического ГВС-драга | документируется предел |
| Гидростатика: T_v | сухая T | ≤ 1,2 % | задокументировано |
| Якорь Φ_t(p_s) | Φ_t(p_s) = 0 | эталон (IFS 2.8) прогнозирует p_s из колоночной дивергенции; наш якорь выбрасывает баротропную компоненту (прилив, барометрическая тенденция) | `Z_ps`: R_d·T_s/p_s·∂p_s/∂t с кинематической ∂p_s/∂t |
| ω: граничные условия | w(p_s) = 0 снизу, верх свободен | эталон: ω(0) = 0 сверху | `W_obrien` (O'Brien 1970), сравнение с mass_consistent |
| Прогноз p_s / ln p_s | отсутствует (13 фиксированных уровней) | структурное ограничение датасета | документируется; частично компенсируется `Z_ps` |

## 3. Обзор литературы

Фактически открытые и прочитанные источники:

1. **IFS Documentation Cy47r1 — Part III: Dynamics and Numerical Procedures**,
   ECMWF, 2020, гл. 2. https://www.ecmwf.int/sites/default/files/elibrary/2020/Part-III-Dynamics_and_Numerical_Procedures.pdf
   — эталонная система (ур. 2.1–2.10); подтверждение якоря ∂p_s/∂t (ур. 2.8) и
   вертикального интеграла гидростатики от поверхности (ур. 2.21).
2. **Held, I. M., Suarez, M. J. (1994)**. A Proposal for the Intercomparison of
   the Dynamical Cores of Atmospheric General Circulation Models. *BAMS* 75(10),
   1825–1830. Спецификация воспроизведена по документации MITgcm §4.7
   (https://mitgcm.readthedocs.io/en/latest/examples/held_suarez_cs/held_suarez_cs.html):
   T_eq = max{200·(p₀/p)^κ, 315 − 60·sin²φ − 10·log(p/p₀)·cos²φ}·(p/p₀)^κ (θ-форма),
   k_T = k_a + (k_s−k_a)·cos⁴φ·max(0,(σ−σ_b)/(1−σ_b)), k_a=1/40 сут, k_s=1/4 сут,
   k_v = k_f·max(0,(σ−σ_b)/(1−σ_b)), k_f=1/1 сут, σ_b=0.7.
3. **O'Brien, J. J. (1970)**. Alternative Solutions to the Classical Vertical
   Velocity Problem. *J. Appl. Meteor.* 9(2), 197–203.
   https://journals.ametsoc.org/view/journals/apme/9/2/1520-0450_1970_009_0197_asttcv_2_0_co_2.xml
   (прочитан абстракт; формула поправки восстановлена из заявленной гипотезы
   «ошибка дивергенции линейна по давлению»: квадратичный по массе вес).
4. **Yanai, M., Esbensen, S., Chu, J.-H. (1973)** — определения Q₁ (apparent
   heat source) и Q₂ (apparent moisture sink); формулы сверены по примеру NCL
   (https://www.ncl.ucar.edu/Applications/wind.shtml): Q₁ = c_p·[∂T/∂t + V·∇T −
   ω(κT/p − ∂T/∂p)], Q₂ = −L·[∂q/∂t + V·∇q + ω·∂q/∂p].
5. **Trenberth, K. E. (1991)**. Climate Diagnostics from Global Analyses:
   Conservation of Mass in ECMWF Analyses. *J. Climate* 4(7), 707–722
   (https://journals.ametsoc.org/view/journals/clim/4/7/1520-0442_1991_004_0707_cdfgac_2_0_co_2.xml,
   прочитан абстракт): локальные невязки уравнения неразрывности в анализах на
   уровнях давления достигают 60–100 % величины дивергентного члена (тропики) —
   количественный ориентир того, что кинематическая ω из анализов замыкается
   плохо даже в оперативных центрах.
6. **Kochkov, D., Yuval, J., Langmore, I., et al. (2024)**. Neural General
   Circulation Models for Weather and Climate. *Nature* 619, 544–549;
   arXiv:2311.07222 (прочитан абстракт): дифференцируемое динамическое ядро +
   обучаемая колоночная физика — вся диабатика (наши P-члены) там целиком
   обучаемая; чистое ядро без P-членов и не предполагается замкнутым.
7. **Verma, Y., Heinonen, M., Garg, V. (2024)**. ClimODE: Climate and Weather
   Forecasting with Physics-informed Neural ODEs. ICLR 2024; arXiv:2404.10024
   (прочитан абстракт): перенос как value-conserving neural flow + обучаемый
   источник — та же структура «адвекция + источник».

(дополняется по ходу)

## 4. Полный журнал прогонов

| # | Job ID | Что | Статус | Итог |
|---|---|---|---|---|
| 1 | 4169258 | smoke: матрица exp 14 (18 ядер), глобус-2000, 30 троек, STRIDE=4 | done (28 с) | скорость 0,94 с/тройку; C_best: u 0.99, v 0.93, t 2.03, q 1.03, z 3.18 |
| 2 | 4169268 | r1: eq15-матрица, глобус-2000, STRIDE=1 (8782 тройки), CLIM_OUT | — | — |
| 3 | 4169269 | r1: eq15-матрица, USA-2000, STRIDE=1, CLIM_OUT | — | — |

(дополняется)

## 5. Таблицы невязок

(заполняется из JSON)

## 6. Разбор гипотез

(заполняется)

## 7. Фигуры

(заполняется)

## 8. Ограничения и что проверить дальше

(заполняется)
