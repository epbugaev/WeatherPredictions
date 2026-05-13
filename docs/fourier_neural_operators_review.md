# Fourier Neural Operators: обзор архитектур и применений

Дата: 2026-05-13

## 1. Введение и мотивация

Классические задачи обучения с учителем учат отображение $f: \mathbb{R}^n \to \mathbb{R}^m$ между конечномерными векторами. При работе с уравнениями в частных производных (PDE) и физическими полями естественнее формулировать задачу как обучение **оператора** $\mathcal{G}: \mathcal{A} \to \mathcal{U}$, где $\mathcal{A}$ и $\mathcal{U}$ — бесконечномерные функциональные пространства (например, пространство начальных условий $a(x)$ и пространство решений $u(x)$). Дискретизация — артефакт численного представления, а не часть задачи: модель должна быть способна работать на разных сетках без переобучения.

Стандартные CNN и MLP **не являются дискретизационно-инвариантными**. Сверточное ядро размера $k \times k$ привязано к шагу сетки $h$: уменьшая $h$ при фиксированном $k$, мы сужаем рецептивное поле в физических единицах, а параметры ядра не имеют корректной интерпретации как дискретизация непрерывного интегрального ядра. То же относится к MLP: число входов жёстко привязано к числу пикселей. Следствие — модель, обученная на $64^2$, не переносится на $256^2$ без потери точности и без интерполяции.

**Mesh-invariance** (или resolution-invariance) означает, что один и тот же набор параметров корректно работает на любой дискретизации области, причём с улучшением сетки ошибка не растёт, а часто и убывает (super-resolution). Эта свойство — ключевое теоретическое и практическое обоснование нейронных операторов.

Идейный фундамент — теорема Чена и Чена 1995 года (T. Chen, H. Chen, IEEE TNN), которая обобщает универсальную аппроксимацию Цыбенко с функций на **нелинейные непрерывные операторы**. Грубо: трёхслойная сеть с подходящей функцией активации может приблизить произвольный непрерывный нелинейный оператор $\mathcal{G}$ на компактном подмножестве банахового пространства. Эта теорема — формальное оправдание DeepONet и, опосредованно, всех нейронных операторов.

В этом обзоре практико-ориентированный разбор семейства FNO: математическая формулировка, основные варианты, сравнение со смежными подходами (DeepONet, GNO, PINO), применения в погоде (FourCastNet, SFNO, GraphCast, Pangu, NeuralGCM) и практические рецепты обучения.

## 2. Базовый FNO

### 2.1 Постановка

Задача: построить оператор $\mathcal{G}_\theta: a \mapsto u$, аппроксимирующий истинный $\mathcal{G}^*$, на парах $(a_i, u_i)$, где $a_i \in \mathcal{A}$ — параметр/начальное условие (например, поле проницаемости в Darcy flow, или $\omega(t_0)$ в Navier-Stokes), а $u_i \in \mathcal{U}$ — решение.

Общая структура нейронного оператора ([Li et al. 2021, "Neural Operator"](https://arxiv.org/abs/2108.08481)):

$$v_{\ell+1}(x) = \sigma\Big(W_\ell\, v_\ell(x) + b_\ell + (\mathcal{K}_\ell v_\ell)(x)\Big),$$

где $W_\ell$ — поточечная (1×1) линейная карта в канальном измерении, $\mathcal{K}_\ell$ — **интегральный оператор**

$$(\mathcal{K} v)(x) = \int_D \kappa(x, y)\, v(y)\, dy,$$

а $\sigma$ — поэлементная нелинейность (GeLU/ReLU). Разные параметризации $\kappa$ порождают разные семейства: ядро на графе (GNO), низкоранговое разложение (LNO), Фурье-параметризация (FNO).

### 2.2 Архитектура FNO

FNO состоит из трёх частей:

1. **Lift** $P: \mathbb{R}^{d_a} \to \mathbb{R}^{d_v}$ — поточечный MLP, поднимающий вход $a(x)$ в более широкий канал $v_0(x) \in \mathbb{R}^{d_v}$. Обычно `width = 32...64` для 2D, `64...128` для 3D и weather.
2. $L$ **Fourier-слоёв** вида выше с $\mathcal{K}_\ell$ = спектральная свёртка.
3. **Projection** $Q: \mathbb{R}^{d_v} \to \mathbb{R}^{d_u}$ — поточечный MLP, проектирующий в выходное пространство.

### 2.3 Спектральная свёртка

Ключевая идея: в стационарном случае $\kappa(x, y) = \kappa(x - y)$ интеграл $(\mathcal{K}v)(x) = (\kappa * v)(x)$ — это свёртка, а по теореме о свёртке

$$(\kappa * v)(x) = \mathcal{F}^{-1}\big(\mathcal{F}(\kappa) \cdot \mathcal{F}(v)\big)(x).$$

FNO параметризует $\mathcal{F}(\kappa)$ напрямую — не учит ядро в пространстве, а учит **тензор спектральных весов** $R \in \mathbb{C}^{k_{\max} \times d_v \times d_v}$ на низких частотах. Высокие частоты $|k| > k_{\max}$ обнуляются (mode truncation). Один слой делает:

$$(\mathcal{K} v)(x) = \mathcal{F}^{-1}\big(R \cdot \mathcal{F}(v)\big)(x),$$

где $\cdot$ — поканальное матричное умножение в комплексном пространстве на каждой моде $k$. На дискретной сетке $\mathcal{F}$ заменяется на FFT.

Pseudo-код одного спектрального слоя (PyTorch-стиль):

```python
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        # комплексные веса; в реальном коде хранятся как real+imag
        self.weights = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):  # x: (B, C, H, W)
        x_ft = torch.fft.rfft2(x)                       # (B, C, H, W//2+1)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        out_ft[..., :self.modes1, :self.modes2] = torch.einsum(
            "bcxy,coxy->boxy",
            x_ft[..., :self.modes1, :self.modes2],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
```

### 2.4 Гиперпараметры и сложность

- **modes** ($k_{\max}$): сколько низких мод сохраняется по каждому измерению. Типично 12–32 для 2D задач с разрешением 64–256.
- **width** ($d_v$): размерность скрытого канала. 20–64 для классических PDE, 256+ для weather.
- **depth** ($L$): число Fourier-слоёв. Обычно $L = 4$ ([Li et al. 2020](https://arxiv.org/abs/2010.08895)).

Сложность одного слоя на сетке из $N$ точек: FFT — $O(N \log N)$, умножение — $O(k_{\max}^d \cdot d_v^2)$. Итого **квазилинейно** по числу точек. Сравним с CNN: для ядра $k \times k$ стоимость — $O(N \cdot k^2 \cdot d_v^2)$, и для глобального рецептивного поля требуется $k \sim \sqrt{N}$, то есть $O(N^2)$, либо стек $\log N$ слоёв с dilation. Для глобально-связанных операторов (Navier-Stokes, climate) FNO выигрывает на порядки.

Сравнение с GNO ([Li et al. 2020, "Neural Operator: GKN"](https://arxiv.org/abs/2003.03485)): GNO вычисляет интегральное ядро через message passing на графе, $O(N^2)$ в плотном варианте.

### 2.5 Дискретизационная инвариантность на практике

Параметры $R$ зависят только от номера моды $k$, а не от шага сетки $h$. Поэтому модель, обученная на $64^2$, корректно применяется на $128^2$ или $256^2$: FFT по-прежнему даёт коэффициенты на тех же $k_{\max}$ модах, а высокие моды обнуляются. В оригинальной работе показано super-resolution: ошибка на тестовой сетке выше тренировочной может быть **ниже**, потому что FFT более точно представляет непрерывный сигнал.

На практике инвариантность работает строго только для **периодических** областей. На непериодических Li et al. рекомендуют zero-padding до большего домена либо явное добавление пространственных координат $(x, y)$ в каналы входа — это даёт сети возможность учить границы.

## 3. Варианты и расширения FNO

### 3.1 2D / 3D FNO

В оригинальной работе тестировались 1D (Burgers), 2D (Darcy) и 3D (Navier-Stokes как пространство-время). 3D FNO дороже памяти, потому что комплексный тензор весов растёт как $k_{\max}^3 \cdot d_v^2$ и FFT — трёхмерный. Альтернатива: 2D + рекуррентный шаг во времени (Markov assumption), используется почти во всех weather-моделях.

### 3.2 Tensorized FNO (TFNO/T-FNO)

[Kossaifi et al. 2023, "MG-TFNO"](https://arxiv.org/abs/2310.00120) предложили заменить полный тензор $R \in \mathbb{C}^{d_v \times d_v \times k_1 \times k_2 \times \dots}$ на тензорное разложение (Tucker, CP или Tensor-Train). Идея: на больших сетках и каналах $R$ доминирует в параметрах и часто избыточен.

Эффект: компрессия параметров до **150×** на турбулентных Navier-Stokes без потери точности (и часто с её повышением, поскольку низкоранговое ограничение действует как регуляризация). В реализации `neuraloperator` это стандартная опция (`TFNO`).

### 3.3 Factorized FNO (F-FNO)

[Tran et al. 2023](https://arxiv.org/abs/2111.13802) ввели **сепарабельную спектральную свёртку**: вместо общего $R(k_1, k_2)$ применяется композиция $R_1(k_1) \cdot R_2(k_2)$, т.е. одномерные операторы по каждому измерению поочерёдно. Плюс:

- параметров $O(d_v^2 (k_1 + k_2))$ вместо $O(d_v^2 k_1 k_2)$;
- глубокие residual connections поверх стандартной структуры FNO.

Авторы сообщают **сокращение ошибки на 83% на Navier-Stokes** относительно стандартного FNO, при увеличении глубины до 24 слоёв (с residual-блоками). Дополнительные приёмы: Markov assumption, инъекция гауссова шума, косинусный LR.

### 3.4 AFNO — Adaptive Fourier Neural Operator

[Guibas et al. 2021](https://arxiv.org/abs/2111.13587) сформулировали FNO как замену self-attention в ViT-стиле. AFNO применяет спектральное преобразование на токены изображения и добавляет два важных приёма:

1. **Block-diagonal weights** в канальном измерении ($R$ блочно-диагональна), сокращающие параметры.
2. **Soft-thresholding** в частотной области:

$$\tilde{R}(k) = \mathrm{SoftShrink}_\lambda(R(k)) = \mathrm{sign}(R(k)) \cdot \max(|R(k)| - \lambda, 0),$$

что обнуляет малозначимые моды и адаптивно разреживает спектр.

Сложность — квазилинейная по числу токенов, в отличие от $O(N^2)$ self-attention. AFNO стал базовым блоком в FourCastNet v1 и был мотивирован способностью обрабатывать длинные последовательности (Cityscapes сегментация с 65k токенов).

### 3.5 Geo-FNO

[Li et al. 2022](https://arxiv.org/abs/2207.05209) решают проблему **нерегулярных областей**. FFT требует регулярной сетки, поэтому Geo-FNO учит **деформацию** $\phi: \Omega_{\text{phys}} \to \Omega_{\text{latent}}$, отображающую физическую (нерегулярную) область в латентную регулярную сетку, на которой работает обычный FNO. $\phi$ — обучаемая, реализуется как neural network или fixed coordinate transform.

Применения: эластичность, plasticity, аэродинамика airfoil, инверсное проектирование. Авторы заявляют ускорение $10^5\times$ по сравнению с численными решателями и 2× точность относительно интерполяции на сетку.

### 3.6 U-FNO

[Wen et al. 2022](https://arxiv.org/abs/2109.03697) — гибрид U-Net и FNO для multiphase flow (CO2/water в пористой среде). Идея: добавить ветвь U-Net параллельно/после Fourier-слоёв, чтобы захватить **локальные высокочастотные** детали, которые FNO склонен сглаживать из-за mode truncation. На multiphase flow с гетерогенной проницаемостью даёт лучшую точность при 3× меньшем объёме данных против чистой CNN.

### 3.7 Spherical FNO (SFNO)

[Bonev et al. 2023](https://arxiv.org/abs/2306.03838) — критическое расширение для **погоды и климата**. Земля — сфера $S^2$, на ней обычный FFT даёт артефакты на полюсах (FFT неявно предполагает планарную периодичность). SFNO заменяет FFT на **дискретное преобразование по сферическим гармоникам**:

$$f(\theta, \phi) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} \hat{f}_\ell^m\, Y_\ell^m(\theta, \phi),$$

где $Y_\ell^m$ — сферические гармоники. Спектральная свёртка становится умножением на $\ell$-зависимый оператор $R(\ell)$ (по теореме Funk-Hecke свертка на $S^2$ диагонализуется в базисе $Y_\ell^m$ при условии, что ядро зависит только от угла). Это даёт **rotational equivariance** на сфере.

Результаты:
- Стабильные авторегрессионные роллауты длиной 1460 шагов (≈ год при шаге 6 ч), тогда как FFT-FNO стабилен ~100 шагов (25 дней) и далее коллапсирует с energy dissipation.
- Сохранение физически реалистичного спектра.

SFNO — backbone FourCastNet v2 (FCN2).

### 3.8 Group-equivariant FNO

[Helwig et al. 2023](https://arxiv.org/abs/2306.05697) расширяют FNO до **эквивариантности относительно групп вращений/отражений/трансляций** через group convolution в частотной области. Применяется там, где симметрии физического оператора известны (изотропная диффузия, гидродинамика без выделенного направления).

### 3.9 CoDA-NO

[Rahman et al. 2024](https://arxiv.org/abs/2403.12553) — Codomain Attention Neural Operator. Токенизирует функции **по канальному (codomain) измерению**, не по пространственному. Это позволяет pretrain на одной физической системе и fine-tune на другой с разным числом переменных — направление multi-physics foundation models. Опубликован на NeurIPS 2024.

### 3.10 Соседние спектральные/operator-подходы

- **LSM (Latent Spectral Models)** ([Wu et al. 2023, arXiv:2301.12664](https://arxiv.org/abs/2301.12664)) — U-Net-подобная архитектура с обучаемыми spectral basis в латентном пространстве. SOTA на solid+fluid PDE при ICML 2023.
- **GNOT** ([Hao et al. 2023, arXiv:2302.14376](https://arxiv.org/abs/2302.14376)) — трансформер-нейрооператор с гетерогенным normalized attention, поддерживает нерегулярные меши и multiple input functions.
- **MgNO** — multigrid neural operator, иерархия V-cycle поверх спектральных блоков.
- **OFormer**, **Galerkin Transformer** — линейные attention-операторы, концептуально близки к FNO через спектральное разложение attention.

## 4. Сравнение с другими нейронными операторами

### 4.1 DeepONet

[Lu et al. 2019](https://arxiv.org/abs/1910.03193) — другая ветвь, прямо реализующая теорему Чена и Чена. Архитектура: две сети,

$$\mathcal{G}(a)(y) \approx \sum_{k=1}^{p} \underbrace{b_k(a(x_1), \dots, a(x_m))}_{\text{branch}} \cdot \underbrace{t_k(y)}_{\text{trunk}},$$

где **branch** кодирует входную функцию $a$ на фиксированных sensor-точках $\{x_i\}$, а **trunk** — координату оценки $y$. Сумма произведений — обобщённое разложение оператора.

Отличия от FNO:
- DeepONet не требует регулярной сетки на входе (sensors могут быть произвольны), но **число и положение sensors фиксированы**. Дискретизационная инвариантность по выходу — да; по входу — частично.
- FNO глобально-конволюционен; DeepONet — функционал branch берёт $a$ как вектор.
- DeepONet проще для нерегулярных областей и инверсных задач, FNO — для регулярных сеток и крупных полей.

### 4.2 Graph Neural Operator / Multipole GNO

GNO ([Li et al. 2020](https://arxiv.org/abs/2003.03485)) реализует $(\mathcal{K} v)(x) = \int \kappa(x, y) v(y) dy$ через message passing на $k$-NN графе. Дорого: плотный вариант — $O(N^2)$, разреженный — $O(N \cdot k)$ при ограниченном радиусе, но тогда теряется глобальность.

Multipole GNO ([Li et al. 2020, NeurIPS](https://arxiv.org/abs/2006.09535)) использует иерархию графов в духе fast multipole method: короткодействующая часть — на мелкой сетке, длиннодействующая — на грубой. Сложность линейна, $O(N)$.

### 4.3 PINO — Physics-Informed Neural Operator

[Li et al. 2021](https://arxiv.org/abs/2111.03794) комбинирует FNO с physics-informed loss. Полный лосс:

$$\mathcal{L} = \mathcal{L}_{\text{data}}(u, u^*) + \lambda \cdot \mathcal{L}_{\text{PDE}}(u),$$

где $\mathcal{L}_{\text{PDE}}$ оценивается на более высоком разрешении, чем данные. Получаем zero-shot super-resolution и работу в режиме нулевых данных (только PDE).

PINO отличается от классического PINN: PINN учит точечную функцию для одной задачи (один оператор не обобщается), PINO — оператор, который применим к любому начальному условию из распределения.

### 4.4 Краткая таблица сравнения

| Подход | Сложность | Меш-инвариантность | Регулярная сетка? | Параметры (типично) |
|---|---|---|---|---|
| FNO | $O(N \log N)$ | да (для регулярной сетки) | требуется | 1–10M |
| T-FNO | $O(N \log N)$ | да | требуется | 0.01–1M |
| F-FNO | $O(N \log N)$ | да | требуется | 1–10M |
| SFNO | $O(N \log N)$ на сфере | да | требуется на $S^2$ | 100M+ (weather) |
| GNO / Multipole | $O(N k)$ / $O(N)$ | да | нет | 0.1–10M |
| DeepONet | $O(N p)$ | по выходу | sensors фикс. | 0.1–10M |
| PINO (FNO+PDE) | $O(N \log N)$ | да | требуется | 1–10M |
| GraphCast (для контраста) | $O(N \log N)$ через multimesh | через icosaedr меш | да (через граф) | 36M |

## 5. Применения в погоде и климате

### 5.1 FourCastNet (v1)

[Pathak et al. 2022](https://arxiv.org/abs/2202.11214) (NVIDIA, LBNL) — первая крупная демонстрация нейрооператора как глобальной модели погоды. Backbone — **AFNO** поверх ViT-патчей $8 \times 8$ на ERA5 разрешения **0.25°** (720 × 1440). 20 каналов: u/v ветер на 10/1000 hPa, температура, влажность, осадки, давление.

Ключевые числа:
- ~1 неделя глобального прогноза **за 2 секунды на 1 GPU** против часов на IFS кластере.
- Сопоставимая с IFS точность на коротких сроках (1–3 дня) для крупномасштабных переменных.
- Превосходит IFS по осадкам — переменной с тонкой пространственной структурой.
- Используется для ensemble в страховании катастроф (JBA, AXA).

### 5.2 FourCastNet v2 / SFNO

[Bonev et al. 2023](https://arxiv.org/abs/2306.03838) — переход с AFNO на SFNO. Модель FCN2 обучена на 73-канальной выборке ERA5 (single levels + pressure levels). Эффекты сферичности:
- Стабильные год-долгие роллауты против ~25 дней у FCN1.
- Корректное поведение на полюсах, отсутствие artefacts.
- Сохранение энергетического спектра.

Используется в NVIDIA Earth-2 и Earth2Studio для multi-GPU инференса и subseasonal forecasting.

### 5.3 FourCastNet 3

[Karras et al. 2025, arXiv:2507.12144](https://arxiv.org/abs/2507.12144) — переход к **геометрической ML + probabilistic ensemble**. Конволюционная архитектура на сфере (а не чисто спектральная), оптимизированная для масштабирования.

Числа:
- **60-дневный глобальный прогноз 0.25°, 6h** за **<4 минут** на 1 GPU.
- 8–60× быстрее ведущих ensemble/diffusion моделей.
- Тренировка с domain-decomposition parallelism на 1024+ GPU.
- Калиброванный probabilistic skill вплоть до 60 дней.

### 5.4 GraphCast (контраст)

[Lam et al. 2022, arXiv:2212.12794](https://arxiv.org/abs/2212.12794) (DeepMind) — **GNN**, не FNO. Архитектура encode-process-decode на icosaedral multi-mesh (несколько уровней разрешения, объединённых в один граф). 36.7M параметров, ERA5 0.25°, 10-дневный прогноз за <1 минуту. Побил IFS на 90% из 1380 метрик.

GraphCast vs SFNO:
- Граф эффективно реализует **локальные + глобальные** связи через multi-mesh.
- SFNO явно использует сферическую геометрию через harmonics.
- GraphCast лучше на средние сроки (5–10 дней) по большинству верификаций; SFNO — на длинных стабильных роллаутах.

### 5.5 Pangu-Weather

[Bi et al. 2022, arXiv:2211.02556](https://arxiv.org/abs/2211.02556) (Huawei) — **3D Earth-Specific Transformer**, 256M параметров. 13 уровней давления, 0.25°. Hierarchical temporal aggregation: отдельные модели для 1ч, 3ч, 6ч, 24ч, комбинация для произвольного горизонта снижает накопление ошибки. Первая ML-модель, превзошедшая IFS по всем переменным во всех горизонтах (1ч–7 дней).

Pangu — трансформер на токенах патчей, не FNO. Их сравнение с FCN2 в [Nature npj Climate eval](https://www.nature.com/articles/s41612-024-00769-0): FengWu > FuXi > GraphCast > FCN2 > Pangu для Восточной Азии и западного Тихого океана летом-осенью 2023, но это региональный benchmark.

### 5.6 NeuralGCM

[Kochkov et al. 2024, Nature; arXiv:2311.07222](https://arxiv.org/abs/2311.07222) (Google/ECMWF) — **гибрид**: дифференцируемый dynamical core (spectral atmospheric dynamics) + ML параметризация подсеточных процессов (конвекция, облака). Тренируется end-to-end через differentiable ODE-solver.

Результаты:
- Конкурентоспособен с лучшими ML и физическими моделями на 1–10 дней.
- Ensemble лучше ECMWF-ENS в **95% случаев** на 2–15 дней.
- Стабилен на климатических масштабах (десятилетия).
- На порядки дешевле классических GCM.

Контраст с FNO-семейством: NeuralGCM **сохраняет физическую структуру** (spectral solver на сферических гармониках для динамики), а ML учит то, что физика не закрывает явно.

### 5.7 Что FNO/SFNO даёт именно погоде

Преимущества:
- **Глобальные дальние связи** через FFT за $O(N \log N)$ — погода глобально связана (телесвязи, ENSO).
- **Резолюционная гибкость**: можно дообучать на более высоком разрешении.
- **Скорость инференса**: 1–4 минуты против часов IFS для 10-дневного прогноза, при сопоставимой или лучшей точности по большинству метрик.
- **На сфере** (SFNO) — стабильные годовые роллауты, физически осмысленный спектр.

Недостатки:
- Требует регулярной сетки lat-lon (или сферической) — нет ассимиляции данных естественным образом.
- Низкочастотный bias — мелкие явления (фронты, осадки на 10 км) сглаживаются.
- Энергетическая диссипация на длинных роллаутах — частично решено SFNO + Sobolev loss.

## 6. Практические детали обучения

### 6.1 Лоссы

**Relative L2** — стандарт для PDE benchmarks:

$$\mathcal{L}_{\text{rel}}(u, u^*) = \frac{\| u - u^* \|_2}{\| u^* \|_2}.$$

**Sobolev / H1 loss** добавляет норму градиента:

$$\mathcal{L}_{H^1}(u, u^*) = \| u - u^* \|_2 + \alpha \| \nabla u - \nabla u^* \|_2,$$

либо в спектральной форме — взвешивание мод по $|k|$:

$$\mathcal{L}_{H^s}(u, u^*) = \sum_k (1 + |k|^2)^s \big| \hat{u}(k) - \hat{u}^*(k) \big|^2.$$

H1 и Sobolev критичны для подавления спектрального bias FNO: чистый L2 поощряет совпадение по низким модам, что приводит к гладким, "размытым" решениям и плохой передаче высокочастотных деталей. В weather это особенно важно для фронтов, осадков и turbulent kinetic energy.

### 6.2 Нормализация

Для multi-variable weather критично:
- **Per-variable, per-level z-score**: $\hat{x}_v = (x_v - \mu_v) / \sigma_v$ для каждой переменной (T2m, u10, z500, …) и уровня давления отдельно.
- **Climatology subtraction**: вычитать сезонную/суточную climatology перед обучением и добавлять обратно — модель учит **аномалии**, а не абсолютные значения.
- Latitude weighting в лоссе: $\cos(\text{lat})$ компенсирует неравномерность площади ячеек lat-lon сетки.

### 6.3 Roll-out / autoregressive training

Базовая схема: предсказываем $u(t+\Delta t)$ из $u(t)$, на инференсе автоматически разматываем. Проблема — exposure bias и накопление ошибки.

**Two-step loss** (используется в GraphCast, FCN2, Pangu):

$$\mathcal{L}_{\text{2step}} = \mathcal{L}(u_{t+1}, \hat{u}_{t+1}) + \mathcal{L}(u_{t+2}, f(\hat{u}_{t+1})),$$

где второй шаг считается через собственное предсказание. Расширяется до $k$-step rollout fine-tuning: сначала обучаем 1-step, затем дообучаем 2-step, 4-step, ... до 10–20 шагов. Каждое продление удваивает требуемую память.

**Markov assumption**: модель видит только $u(t-\delta, t)$ как вход. Это упрощает архитектуру и согласуется со структурой PDE первого порядка по времени.

### 6.4 Padding и непериодические границы

FFT неявно предполагает периодичность. Для непериодических областей:
- **Zero-padding** перед FFT: расширить домен на $p$ ячеек нулями, применить FNO, обрезать. Снижает edge artifacts.
- Включить координаты $(x, y)$ как добавочные каналы входа — даёт сети geometric embedding.
- Для weather на lat-lon: по долготе сетка периодична естественно, по широте — нет. SFNO решает это сферическими гармониками радикально.

### 6.5 Mixed precision и память

FFT в `torch.fft` поддерживает `cfloat` (complex64), причём autocast в bf16 для FFT не всегда стабилен — лучше держать FFT в fp32, а conv/linear блоки в bf16. Комплексные тензоры $R$ занимают вдвое больше памяти, чем real. Для 3D FNO и weather память — главное ограничение; T-FNO и F-FNO существенно его снижают.

### 6.6 Подводные камни

- **Spectral bias**: FNO легко учит низкие моды, плохо — высокие ([Qin et al. 2024](https://arxiv.org/abs/2404.07200)). Высокие моды дают артефакты dissipation/aliasing. Mitigation: Sobolev loss, увеличение modes, residual/U-Net блоки (U-FNO).
- **Aliasing**: если в данных есть энергия выше $k_{\max}$, она aliases на низкие моды. До FFT желательно применять anti-aliasing filter или предупредить, что данные band-limited.
- **Energy dissipation в длинных rollouts**: модель теряет high-k энергию, спектр падает. Решается SFNO + Markov Neural Operator regularization (статистическая верность аттрактора), spectrogram loss ([arXiv:2511.08753](https://arxiv.org/abs/2511.08753)).
- **Граничные артефакты** на непериодических задачах: zero-padding и явные координаты обязательны.
- **Переобучение на низких модах**: малая ёмкость в высоких модах + большая в низких. Регуляризация — soft-thresholding (AFNO), tensor decomposition (TFNO).

## 7. Открытые проблемы и направления

1. **Стабильность длинных roll-out**. Даже SFNO теряет энергию на горизонтах годов в климатическом режиме. Подходы: гибридизация с физическим solver (NeuralGCM), Markov regularization, diffusion-based ensembles (как в FourCastNet 3).
2. **Conservation laws**. FNO не гарантирует сохранение массы, энергии, момента. Hard constraints (через проекцию в подпространство допустимых решений) и soft penalties в лоссе — активная область. Применимо для baroclinic-conservative климата.
3. **Mesh-invariance на нерегулярных сетках**. FFT требует регулярности. Geo-FNO решает через деформацию; GNOT/CoDA-NO/DeepONet через attention или kernel integration. Универсальной победы пока нет.
4. **Масштабирование**. Big-FNO с миллиардами параметров (как foundation models) пока редкость. AFNO/SFNO scaling — основное направление NVIDIA Earth-2.
5. **Attention-FNO гибриды**. Объединить локальную spectral conv (для глобальности) и attention (для адаптивности) — GNOT, Galerkin Transformer, CoDA-NO.
6. **Inverse problems и data assimilation**. FNO как differentiable surrogate в variational DA — пока экспериментально. Pangu и FCN2 интегрируются с EnKF/4D-Var.
7. **Probabilistic ensembles**. От детерминированных FNO к probabilistic (diffusion-based как GenCast, FCN3). Калиброванный uncertainty — практическая необходимость для прогноза экстремумов.

## 8. Полезные ресурсы

### 8.1 Ключевые статьи (arXiv)

- [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al. 2020)](https://arxiv.org/abs/2010.08895)
- [Neural Operator: Learning Maps Between Function Spaces (Kovachki, Li et al. 2021)](https://arxiv.org/abs/2108.08481)
- [GKN — Neural Operator: Graph Kernel Network (Li et al. 2020)](https://arxiv.org/abs/2003.03485)
- [Multipole Graph Neural Operator (Li et al. 2020)](https://arxiv.org/abs/2006.09535)
- [DeepONet (Lu et al. 2019)](https://arxiv.org/abs/1910.03193)
- [Adaptive Fourier Neural Operator (Guibas et al. 2021)](https://arxiv.org/abs/2111.13587)
- [Factorized FNO (Tran et al. 2023)](https://arxiv.org/abs/2111.13802)
- [MG-Tensorized FNO (Kossaifi et al. 2023)](https://arxiv.org/abs/2310.00120)
- [Geo-FNO (Li et al. 2022)](https://arxiv.org/abs/2207.05209)
- [U-FNO (Wen et al. 2022)](https://arxiv.org/abs/2109.03697)
- [Spherical FNO (Bonev et al. 2023)](https://arxiv.org/abs/2306.03838)
- [Group-Equivariant FNO (Helwig et al. 2023)](https://arxiv.org/abs/2306.05697)
- [CoDA-NO (Rahman et al. 2024)](https://arxiv.org/abs/2403.12553)
- [GNOT (Hao et al. 2023)](https://arxiv.org/abs/2302.14376)
- [Latent Spectral Models (Wu et al. 2023)](https://arxiv.org/abs/2301.12664)
- [PINO (Li et al. 2021)](https://arxiv.org/abs/2111.03794)
- [FourCastNet (Pathak et al. 2022)](https://arxiv.org/abs/2202.11214)
- [FourCastNet 3 (2025)](https://arxiv.org/abs/2507.12144)
- [GraphCast (Lam et al. 2022)](https://arxiv.org/abs/2212.12794)
- [Pangu-Weather (Bi et al. 2022)](https://arxiv.org/abs/2211.02556)
- [NeuralGCM (Kochkov et al. 2023)](https://arxiv.org/abs/2311.07222)
- [Spectral perspective on FNO (Qin et al. 2024)](https://arxiv.org/abs/2404.07200)

### 8.2 Реализации

- [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) — основная PyTorch-библиотека от группы Anandkumar: FNO, TFNO, SFNO, U-FNO, GNO, dataloaders, лоссы (H1, L2 relative), training loops.
- [NVIDIA Modulus / PhysicsNeMo](https://github.com/NVIDIA/modulus) — production-grade фреймворк для научного ML: FNO, AFNO, SFNO, FourCastNet, weather pipelines, multi-GPU.
- [FourCastNet (NVlabs)](https://github.com/NVlabs/FourCastNet) — публичный код, веса, инструкции инференса.
- [Earth2Studio](https://github.com/NVIDIA/earth2studio) — multi-GPU inference и S2S forecasting.
- [GraphCast (DeepMind)](https://github.com/google-deepmind/graphcast) — JAX-реализация и публичные веса.

### 8.3 Бенчмарки

- [PDEBench (Takamoto et al. 2022)](https://arxiv.org/abs/2210.07182) — 1D/2D/3D PDE (advection, Burgers, diffusion-reaction, Darcy, compressible NS, shallow water) с FNO/TFNO/U-Net baselines.
- [The Well (Polymathic AI, 2024)](https://polymathic-ai.org/the_well/) — 15 ТБ spatiotemporal physics, 16 наборов данных от astrophysics до biology, baselines на FNO/TFNO/ConvNeXt-U-Net.
- [WeatherBench 2](https://sites.research.google/weatherbench/) — глобальный benchmark forecasts на ERA5, deterministic + ensemble метрики, включая GraphCast, Pangu, FCN, NeuralGCM.

## Источники

- https://arxiv.org/abs/2010.08895 — Li et al., Fourier Neural Operator for Parametric PDEs
- https://arxiv.org/abs/2108.08481 — Kovachki, Li et al., Neural Operator: Learning Maps Between Function Spaces
- https://arxiv.org/abs/2003.03485 — Li et al., Neural Operator: Graph Kernel Network
- https://arxiv.org/abs/2006.09535 — Li et al., Multipole Graph Neural Operator
- https://arxiv.org/abs/1910.03193 — Lu et al., DeepONet
- https://arxiv.org/abs/2111.13587 — Guibas et al., Adaptive Fourier Neural Operator
- https://arxiv.org/abs/2111.13802 — Tran et al., Factorized FNO
- https://arxiv.org/abs/2310.00120 — Kossaifi et al., MG-Tensorized FNO
- https://arxiv.org/abs/2207.05209 — Li et al., Geo-FNO
- https://arxiv.org/abs/2109.03697 — Wen et al., U-FNO
- https://arxiv.org/abs/2306.03838 — Bonev et al., Spherical FNO
- https://arxiv.org/abs/2306.05697 — Helwig et al., Group-Equivariant FNO
- https://arxiv.org/abs/2403.12553 — Rahman et al., CoDA-NO
- https://arxiv.org/abs/2302.14376 — Hao et al., GNOT
- https://arxiv.org/abs/2301.12664 — Wu et al., LSM
- https://arxiv.org/abs/2111.03794 — Li et al., PINO
- https://arxiv.org/abs/2202.11214 — Pathak et al., FourCastNet
- https://arxiv.org/abs/2507.12144 — FourCastNet 3
- https://arxiv.org/abs/2212.12794 — Lam et al., GraphCast
- https://arxiv.org/abs/2211.02556 — Bi et al., Pangu-Weather
- https://arxiv.org/abs/2311.07222 — Kochkov et al., NeuralGCM
- https://arxiv.org/abs/2404.07200 — Toward a Better Understanding of FNO from a Spectral Perspective
- https://arxiv.org/abs/2511.08753 — FNO for Structural Dynamics, Spectrogram Loss
- https://arxiv.org/abs/2210.07182 — Takamoto et al., PDEBench
- https://zongyi-li.github.io/blog/2020/fourier-pde/ — Zongyi Li, blog on FNO
- https://neuraloperator.github.io/dev/theory_guide/fno.html — neuraloperator documentation
- https://github.com/neuraloperator/neuraloperator — neuraloperator library
- https://github.com/NVlabs/FourCastNet — FourCastNet repository
- https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/ — NVIDIA blog on FCN3
- https://polymathic-ai.org/the_well/ — The Well benchmark
- https://www.nature.com/articles/s41586-024-07744-y — NeuralGCM in Nature
- https://www.nature.com/articles/s41612-024-00769-0 — Evaluation of five global AI models, npj Climate
- IEEE Transactions on Neural Networks 6(4) 1995 — Chen & Chen, Universal Approximation to Nonlinear Operators
