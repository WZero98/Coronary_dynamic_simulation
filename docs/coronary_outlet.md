# `coronary_outlet.py` 使用说明

本文档说明冠状动脉**出口管腔内压力**边界条件模块 `coronary_outlet.py` 的物理模型、单位约定、函数 API 与典型用法。该模块以脉动流量 \(Q(t)\) 驱动 **二元（RC）** 与 **三元（RCR）** Windkessel 电路，数值积分得到出口压力时间序列 \(P_{\mathrm{out}}(t)\)，供 1D/0D 耦合或后处理使用。

---

## 目录

1. [概述](#1-概述)
2. [环境与依赖](#2-环境与依赖)
3. [物理模型](#3-物理模型)
4. [单位与默认参数](#4-单位与默认参数)
5. [快速开始](#5-快速开始)
6. [函数 API 参考](#6-函数-api-参考)
7. [与入口模块的衔接](#7-与入口模块的衔接)
8. [参数标定与调优](#8-参数标定与调优)
9. [与 CFD 求解器的对接思路](#9-与-cfd-求解器的对接思路)
10. [脚本入口 `__main__`](#10-脚本入口-__main__)
11. [常见问题](#11-常见问题)

---

## 1. 概述

`coronary_outlet.py` 用于生成**某段冠状动脉出口处**管腔内压力随时间变化的边界条件：

| 模型 | 电路元件 | 出口压力定义 |
|------|----------|--------------|
| **二元 Windkessel**（`2wk` / `rc`） | 远端阻力 \(R_d\) + 顺应性 \(C\) | \(P_{\mathrm{out}} = P\)（顺应性节点压） |
| **三元 Windkessel**（`3wk` / `rcr`） | 近端阻力 \(R_p\) + 远端阻力 \(R_d\) + 顺应性 \(C\) | \(P_{\mathrm{out}} = P_{\mathrm{wk}} + R_p Q(t)\) |

**驱动流量**默认来自同仓库 `coronary_inlet.py` 的 `coronary_inlet_flow`（傅里叶级数生理波形）；也可传入任意与 `t` 等长的 `q_in` 数组。

**静脉压**：不作为可调参数；电路出口按零压参考，\(R_d\) 由参考管腔压 \(P_{\mathrm{ref}}\) 与平均流量标定。

**数值积分**：对 Windkessel 常微分方程采用 **RK4**（四阶 Runge–Kutta），在用户提供的时间网格上逐步推进。

**对外公开 API**（以下划线 `_` 开头的为内部实现，文档不展开）：

| 函数 | 作用 |
|------|------|
| `windkessel2_outlet_pressure` | 二元模型，给定 \(t, Q(t)\) |
| `windkessel3_outlet_pressure` | 三元模型，给定 \(t, Q(t)\) |
| `coronary_outlet_pressure` | 统一入口：自动生成或接收 \(Q(t)\)，选择模型 |
| `coronary_outlet_pressure_stats` | 压力波形统计量 |
| `plot_coronary_outlet_pressure` | 二元/三元对比可视化 |

---

## 2. 环境与依赖

| 依赖 | 用途 |
|------|------|
| Python 3.10+（建议） | 类型注解 `float \| None` 等 |
| `numpy` | 数组与 RK4 积分 |
| `matplotlib` | `plot_coronary_outlet_pressure` 绘图（可选） |
| `coronary_inlet.py` | 默认驱动流量 `coronary_inlet_flow` |
| `demo/units.py` | 参考动脉压常量 `P_INLET_REF_MMHG` |

安装示例：

```bash
pip install numpy matplotlib
```

无图形界面时：

```bash
# Windows PowerShell
$env:MPLBACKEND='Agg'
python coronary_outlet.py
```

---

## 3. 物理模型

### 3.1 二元 Windkessel（RC）

等效电路：动脉出口 → 顺应性 \(C\)（压力 \(P\)）→ 远端阻力 \(R_d\) → **零压参考**（静脉压相对管腔压可忽略，不单独建模）。

微分方程：

\[
C \frac{dP}{dt} = Q(t) - \frac{P}{R_d}
\]

**出口管腔压**即 \(P_{\mathrm{out}}(t) = P(t)\)。

稳态（\(dP/dt = 0\)）时：

\[
P_{\mathrm{ss}} = \bar{Q}\, R_d
\]

其中 \(\bar{Q}\) 为流量时间均值。

### 3.2 三元 Windkessel（RCR）

在二元模型基础上，在流量路径上串联**近端阻力** \(R_p\)，用于表征特征阻抗/近端反射；微循环侧仍由 \(R_d\)–\(C\) 描述，状态变量为 \(P_{\mathrm{wk}}\)：

\[
C \frac{dP_{\mathrm{wk}}}{dt} = Q(t) - \frac{P_{\mathrm{wk}}}{R_d}
\]

\[
P_{\mathrm{out}}(t) = P_{\mathrm{wk}}(t) + R_p\, Q(t)
\]

- \(P_{\mathrm{wk}}\)：顺应性节点（微循环）压力  
- \(R_p Q(t)\)：近端电阻上的瞬时压降，使出口压脉动通常**强于**纯二元模型（在相同 \(R_d, C\) 下）

### 3.3 与 `demo/outlet_boundary_condition.py` 的关系

`demo/outlet_boundary_condition.RCWindkesselMurrayOutletBC` 在 **1D 求解器每个时间步**内联更新 Windkessel 状态，并叠加 Murray 流量分配与血压匹配。

`coronary_outlet.py` 则提供**离线、批量**的压力曲线生成，便于：

- 预先构造 Dirichlet 压力边界 \(P_{\mathrm{out}}(t)\)
- 与入口流量波形对比、标定参数
- 不依赖网格的 0D 分析

demo 中仍保留静脉压 \(P_{\mathrm{ven}}\) 用于 1D 耦合；**本模块**为简化出口边界生成，方程中不含 \(P_{\mathrm{ven}}\)。本模块默认 \(C = 0.8\,\mathrm{mL/mmHg}\)，与 demo 中 `wk_C=0.8` 对齐。

---

## 4. 单位与默认参数

| 量 | 符号 | 单位 |
|----|------|------|
| 时间 | \(t\) | s |
| 流量 | \(Q\) | mL/s |
| 压力 | \(P\) | mmHg |
| 远端/近端阻力 | \(R_d,\, R_p\) | mmHg·s/mL |
| 顺应性 | \(C\) | mL/mmHg |
| 心率 | `heart_rate` | bpm（次/分） |
| 心输出量 | `cardiac_output` | L/min |

**模块内常量**（来自 `demo/units.py` 与模块默认值）：

| 名称 | 默认值 | 说明 |
|------|--------|------|
| `P_INLET_REF_MMHG` | 80.0 | 标定 \(R_d\) 时用的参考管腔压 (mmHg) |
| `_DEFAULT_C_ML_PER_MMHG` | 0.8 | 顺应性 \(C\) |
| `_DEFAULT_Q_MEAN_ML_S` | 3.33 | 仅用于未显式给流量时的阻力标定参考 |

**默认远端阻力**（当 `r_distal_mmhg_s_per_ml=None`）：

\[
R_d = \frac{P_{\mathrm{ref}}}{\bar{Q}}
\]

其中 \(\bar{Q} = \mathrm{mean}(Q(t))\)，\(P_{\mathrm{ref}} = 80\,\mathrm{mmHg}\)。稳态时 \(\bar{P} \approx P_{\mathrm{ref}}\)。

**默认近端阻力**（三元模型，`r_proximal_mmhg_s_per_ml=None`）：

\[
R_p = 0.12 \times R_d
\]

比例由参数 `proximal_fraction` 控制（默认 `0.12`）。

---

## 5. 快速开始

### 5.1 最简调用（自动流量 + 二元模型）

```python
import numpy as np
from coronary_outlet import coronary_outlet_pressure

t = np.linspace(0, 6, 4800)  # 6 s，约 800 Hz 采样
p_out = coronary_outlet_pressure(
    t,
    model="2wk",
    heart_rate=75,
    cardiac_output=5.0,
    coronary_fraction=0.03,
)
# p_out.shape == t.shape，单位 mmHg
```

### 5.2 三元模型（返回微循环节点压）

```python
p_out, p_wk = coronary_outlet_pressure(
    t,
    model="3wk",
    heart_rate=75,
    cardiac_output=5.0,
    coronary_fraction=0.03,
)
```

### 5.3 自定义流量驱动

```python
from coronary_inlet import coronary_inlet_flow
from coronary_outlet import windkessel2_outlet_pressure

q = coronary_inlet_flow(t, heart_rate=60, coronary_fraction=0.04)
p = windkessel2_outlet_pressure(t, q, compliance_ml_per_mmhg=0.05)
```

### 5.4 统计与绘图

```python
from coronary_outlet import coronary_outlet_pressure_stats, plot_coronary_outlet_pressure

stats = coronary_outlet_pressure_stats(
    heart_rate=75,
    cardiac_output=5.0,
    coronary_fraction=0.03,
    model="3wk",
)
print(stats)

plot_coronary_outlet_pressure(duration_s=4.0)  # 弹出 2×2 对比图
```

---

## 6. 函数 API 参考

### 6.1 `windkessel2_outlet_pressure`

**二元 Windkessel**出口管腔压 \(P_{\mathrm{out}}(t)\)。

```python
windkessel2_outlet_pressure(
    t,
    q_in,
    r_distal_mmhg_s_per_ml=None,
    compliance_ml_per_mmhg=0.8,
    p_init_mmhg=None,
)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `t` | array-like | （必填） | 时间序列 (s)，**单调递增**；相邻点间距可不等，RK4 使用实际 \(\Delta t\) |
| `q_in` | array-like | （必填） | 驱动流量 \(Q(t)\) (mL/s)，**与 `t` 等长** |
| `r_distal_mmhg_s_per_ml` | float or None | None | 远端阻力 \(R_d\)。`None` 时用 \(P_{\mathrm{ref}}=80\) 与 `mean(q_in)` 标定：\(R_d=P_{\mathrm{ref}}/\bar{Q}\) |
| `compliance_ml_per_mmhg` | float | 0.8 | 顺应性 \(C\) (mL/mmHg) |
| `p_init_mmhg` | float or None | None | \(t=0\) 时刻压力；`None` 时取稳态 \(\bar{Q} R_d\) |

#### 返回值

- `ndarray`：与 `t` 同形的 \(P_{\mathrm{out}}(t)\) (mmHg)

#### 示例

```python
import numpy as np
from coronary_outlet import windkessel2_outlet_pressure

t = np.linspace(0, 2, 2000)
q = 2.5 * (1 + 0.6 * np.sin(2 * np.pi * t))  # 示意脉动流量
p = windkessel2_outlet_pressure(
    t, q,
    r_distal_mmhg_s_per_ml=25.0,
    compliance_ml_per_mmhg=0.1,
)
```

#### 注意

- `t` 仅含 1 个点时，返回常数数组（由 `p_init` 或 \(Q R_d\) 决定），不进行步进积分。
- 大顺应性 \(C\) 使 \(R_d C \gg\) 心动周期时，\(P_{\mathrm{out}}\) 脉动会被强烈平滑（见 [§8](#8-参数标定与调优)）。

---

### 6.2 `windkessel3_outlet_pressure`

**三元 Windkessel**出口管腔压及微循环节点压。

```python
windkessel3_outlet_pressure(
    t,
    q_in,
    r_proximal_mmhg_s_per_ml=None,
    r_distal_mmhg_s_per_ml=None,
    compliance_ml_per_mmhg=0.8,
    proximal_fraction=0.12,
    p_wk_init_mmhg=None,
)
```

#### 参数

在 [§6.1](#61-windkessel2_outlet_pressure) 基础上增加：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `r_proximal_mmhg_s_per_ml` | float or None | None | 近端阻力 \(R_p\)。`None` 时 \(R_p = \texttt{proximal\_fraction} \times R_d\) |
| `proximal_fraction` | float | 0.12 | 仅当 `r_proximal_mmhg_s_per_ml=None` 时生效 |
| `p_wk_init_mmhg` | float or None | None | \(t=0\) 时 \(P_{\mathrm{wk}}\)；`None` 时为稳态 \(\bar{Q} R_d\) |

（`r_distal_mmhg_s_per_ml`、`compliance_ml_per_mmhg` 含义同二元。）

#### 返回值

- `tuple[ndarray, ndarray]`：
  - `P_out`：出口管腔压 \(P_{\mathrm{wk}} + R_p Q\) (mmHg)
  - `P_wk`：顺应性节点压 (mmHg)

#### 示例

```python
p_out, p_wk = windkessel3_outlet_pressure(
    t, q,
    r_proximal_mmhg_s_per_ml=3.0,
    r_distal_mmhg_s_per_ml=28.0,
    compliance_ml_per_mmhg=0.8,
)
# 近端压降分量
dp_prox = p_out - p_wk  # 等于 R_p * q（数值上）
```

---

### 6.3 `coronary_outlet_pressure`

**推荐的一站式接口**：自动调用 `coronary_inlet_flow`（或使用自定义 `q_in`），再进入二元或三元 Windkessel。

```python
coronary_outlet_pressure(
    t,
    model="2wk",
    heart_rate=75.0,
    cardiac_output=5.0,
    coronary_fraction=0.03,
    q_in=None,
    **windkessel_kwargs,
)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `t` | array-like | （必填） | 时间 (s) |
| `model` | str | `"2wk"` | `"2wk"` / `"rc"` → 二元；`"3wk"` / `"rcr"` → 三元 |
| `heart_rate` | float | 75.0 | 心率 (bpm)，仅当 `q_in is None` |
| `cardiac_output` | float | 5.0 | 心输出量 (L/min)，仅当 `q_in is None` |
| `coronary_fraction` | float | 0.03 | 冠脉流量占 CO 比例，仅当 `q_in is None` |
| `q_in` | ndarray or None | None | 若提供，则**忽略**上述三个血流参数，直接作为 \(Q(t)\) |
| `**windkessel_kwargs` | — | — | 传递给 `windkessel2_outlet_pressure` 或 `windkessel3_outlet_pressure`，例如 `r_distal_mmhg_s_per_ml`、`compliance_ml_per_mmhg`、`proximal_fraction` 等 |

#### 返回值

| `model` | 返回值 |
|---------|--------|
| `"2wk"`, `"rc"` | `ndarray` \(P_{\mathrm{out}}(t)\) |
| `"3wk"`, `"rcr"` | `tuple` \((P_{\mathrm{out}}, P_{\mathrm{wk}})\) |

#### 示例：与入口相位对齐

```python
p = coronary_outlet_pressure(
    t,
    model="2wk",
    heart_rate=75,
    cardiac_output=5.0,
    coronary_fraction=0.03,
    # 下列关键字会传入 coronary_inlet_flow（若未提供 q_in）
    # 需在 coronary_inlet 中支持；通过 stats 时同样筛选
)
# Windkessel 专用参数
p3, pwk = coronary_outlet_pressure(
    t,
    model="3wk",
    q_in=my_q,
    r_distal_mmhg_s_per_ml=30.0,
    compliance_ml_per_mmhg=0.2,
)
```

#### 错误

- `model` 非上述四种之一时：`ValueError: 不支持的 Windkessel 模型: ...`

---

### 6.4 `coronary_outlet_pressure_stats`

在内部构造时间网格（`duration_cycles` 个心动周期），计算压力波形统计量。

```python
coronary_outlet_pressure_stats(
    heart_rate=75.0,
    cardiac_output=5.0,
    coronary_fraction=0.03,
    duration_cycles=3.0,
    model="2wk",
    **kwargs,
) -> dict
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `heart_rate` | float | 75.0 | 心率 (bpm) |
| `cardiac_output` | float | 5.0 | 心输出量 (L/min) |
| `coronary_fraction` | float | 0.03 | 冠脉流量比例 |
| `duration_cycles` | float | 3.0 | 模拟时长 = `duration_cycles × (60/HR)` 秒 |
| `model` | `"2wk"` or `"3wk"` | `"2wk"` | Windkessel 类型 |
| `**kwargs` | — | — | **分流**：<br>• 键名为 `fourier_coefficients`、`min_flow_fraction`、`phase_offset` → 传给 `coronary_inlet_flow`<br>• 其余键（如 `r_distal_mmhg_s_per_ml`、`compliance_ml_per_mmhg`）→ 传给 `windkessel2/3_outlet_pressure` |

#### 返回值（`dict`）

**二元模型**：

| 键 | 含义 |
|----|------|
| `model` | `"2wk"` |
| `P_mean_mmHg` | 压力时间均值 |
| `P_max_mmHg` | 压力最大值 |
| `P_min_mmHg` | 压力最小值 |
| `pulsatility_index` | 脉动指数 PI = \((P_{\max}-P_{\min})/P_{\mathrm{mean}}\) |
| `Q_mean_mL_s` | 驱动流量均值 |

**三元模型**额外包含：

| 键 | 含义 |
|----|------|
| `P_wk_mean_mmHg` | \(P_{\mathrm{wk}}\) 均值 |
| `P_wk_max_mmHg` | \(P_{\mathrm{wk}}\) 最大值 |
| `P_wk_min_mmHg` | \(P_{\mathrm{wk}}\) 最小值 |

#### 示例

```python
stats2 = coronary_outlet_pressure_stats(
    heart_rate=75,
    model="2wk",
    compliance_ml_per_mmhg=0.1,
)
stats3 = coronary_outlet_pressure_stats(
    heart_rate=75,
    model="3wk",
    proximal_fraction=0.15,
    phase_offset=0.5,  # 传给入口傅里叶波形
)
```

---

### 6.5 `plot_coronary_outlet_pressure`

绘制 **2×2** 子图：入口流量、二元/三元出口压对比、单周期细节、三元模型 \(P_{\mathrm{wk}}\) 与 \(R_p Q\) 分解；并在控制台打印两种模型的 `coronary_outlet_pressure_stats` 结果。

```python
plot_coronary_outlet_pressure(
    heart_rate=75.0,
    cardiac_output=5.0,
    coronary_fraction=0.03,
    duration_s=4.0,
    r_distal_mmhg_s_per_ml=None,
    r_proximal_mmhg_s_per_ml=None,
    compliance_ml_per_mmhg=0.8,
)
```

#### 参数

| 参数 | 说明 |
|------|------|
| `heart_rate`, `cardiac_output`, `coronary_fraction` | 与 `coronary_inlet_flow` 一致 |
| `duration_s` | 绘图时间长度 (s)；内部采样约 `1000 × duration_s` 点 |
| `r_distal_mmhg_s_per_ml` | 远端阻力；`None` 为自动标定 |
| `r_proximal_mmhg_s_per_ml` | 近端阻力（仅三元）；`None` 为 \(0.12 R_d\) |
| `compliance_ml_per_mmhg` | 顺应性 \(C\) |

#### 返回值

无（`plt.show()` 阻塞显示；打印统计字典到 stdout）。

#### 子图说明

1. **冠脉入口驱动流量** \(Q(t)\)  
2. **出口管腔内压力** \(P_{\mathrm{out}}(t)\)：二元 vs 三元，虚线为各自时间均值  
3. **单周期出口压力**（一个心动周期）  
4. **三元模型分解**：\(P_{\mathrm{wk}}\) 与 \(P_{\mathrm{out}} - P_{\mathrm{wk}} \approx R_p Q\)

---

## 7. 与入口模块的衔接

出口压力由**流量**驱动，默认流量来自 `coronary_inlet.coronary_inlet_flow`：

\[
\bar{Q} = \frac{\mathrm{CO} \times 1000 \times \texttt{coronary\_fraction}}{60}
\quad (\mathrm{mL/s})
\]

建议在配对使用时保持 **`heart_rate`、`cardiac_output`、`coronary_fraction` 一致**，以便 \(R_d\) 标定中的 \(\bar{Q}\) 与入口波形一致。

**典型工作流**：

```text
coronary_inlet_flow(t, ...)  →  Q(t)
        ↓
windkessel2/3_outlet_pressure(t, Q(t), ...)  →  P_out(t)
```

或使用 `coronary_outlet_pressure(t, model=..., heart_rate=..., ...)` 一步完成。

**相位对齐**：若压力波形需相对 ECG/主动脉压平移，应在入口侧设置 `coronary_inlet_flow(..., phase_offset=...)`，或在生成 `q_in` 后自行偏移时间（本模块不内置相位参数）。

---

## 8. 参数标定与调优

### 8.1 时间常数与脉动幅度

微循环时间常数 \(\tau = R_d C\)。当 \(\tau \gg T_{\mathrm{beat}} = 60/\mathrm{HR}\)（默认 \(R_d \approx P_{\mathrm{ref}}/\bar{Q}\)，如 \(\bar{Q}\approx2.5\,\mathrm{mL/s}\) 时 \(R_d\sim 32\,\mathrm{mmHg·s/mL}\)，\(C=0.8\) → \(\tau \sim 26\,\mathrm{s}\)）时：

- **二元模型**：\(P_{\mathrm{out}}\) 接近稳态，脉动指数 PI 很小  
- **三元模型**：\(P_{\mathrm{out}} = P_{\mathrm{wk}} + R_p Q\) 中 \(R_p Q(t)\) 仍携带流量脉动，PI 通常明显大于二元

若需增强二元出口脉动，可尝试：

- 减小 `compliance_ml_per_mmhg`（如 `0.05`–`0.15`），使 \(\tau\) 接近 1–2 个心动周期  
- 显式增大 `r_distal_mmhg_s_per_ml` 或减小 \(\bar{Q}\) 标定基准（需保持生理合理性）

### 8.2 阻力标定

| 目标 | 做法 |
|------|------|
| 平均出口压 ≈ 80 mmHg | 保持 `r_distal_mmhg_s_per_ml=None`，并令 `mean(q_in)` 等于目标平均流量 |
| 固定外周阻力 | 直接传入 `r_distal_mmhg_s_per_ml=R_phys` |
| 增强近端脉动贡献 | 增大 `r_proximal_mmhg_s_per_ml` 或 `proximal_fraction` |

### 8.3 初始条件

默认从**稳态**开始积分，避免起始瞬态过大。若需从实测初始压开始：

```python
windkessel2_outlet_pressure(t, q, p_init_mmhg=78.0)
windkessel3_outlet_pressure(t, q, p_wk_init_mmhg=75.0)
```

建议模拟时长 ≥ 2–3 个心动周期，使周期性稳定。

### 8.4 时间网格

- 不均匀 `t` 支持，但应保证 \(\Delta t\) 足够小以解析 \(Q(t)\) 谐波（傅里叶入口含多谐波时，建议每周期 ≥ 500–1000 点）  
- `t` 必须单调递增；重复或倒序时间点会导致积分异常

---

## 9. 与 1D 求解器的对接思路

本模块输出的是**标量出口边界** \(P_{\mathrm{out}}(t)\) 或 Windkessel 参数约定，不含空间分布。

### 9.1 与 `navier_stokes_1d.py`（推荐）

根目录 **`navier_stokes_1d.py`** 已在线耦合 Windkessel + tube law：

| 角色 | 实现 |
|------|------|
| 驱动流量 | \(Q_{\mathrm{drive}} = Q(L)\)，出口管腔 PDE 流量 |
| 出口管腔压 | `P_lumen` 来自 ODE 状态，经 `area_from_lumen_pressure_mmhg` 得 \(A(L)\) |
| 管腔 \(Q(L)\) | **不由** \(P_{\mathrm{wk}}/R_d\) 强制，由 1D PDE 与入口传播决定 |
| 离线参考 | `NavierStokes1D.reference_outlet_pressure(t, par)` 调用本模块 `windkessel2/3_outlet_pressure` |

详见 [`navier_stokes_1d.md`](navier_stokes_1d.md) §5.2、§11。

### 9.2 与 `demo/navier_stokes.py`（早期原型）

`demo/` 为旧原型，接口不同，**勿与根目录求解器混用**。历史上常见两种方式：

1. **Dirichlet 压力边界**  
   将 `P_out(t)` 插值到求解器时间步，在出口施加管腔压；截面积由 tube law 反解。

2. **0D–1D 耦合**  
   每步用管腔流量推进 Windkessel（demo 中 `RCWindkesselMurrayOutletBC.advance`）；本模块适合**预先验证**参数或生成参考曲线。

**注意**：demo 出口边界含 Murray 多分支分配；本模块为**单出口、单路** Windkessel。

---

## 10. 脚本入口 `__main__`

直接运行：

```bash
python coronary_outlet.py
```

将依次：

1. 在 `duration_s=6` s 上计算二元/三元 `coronary_outlet_pressure`  
2. 打印 `coronary_outlet_pressure_stats`（二元、三元）及压力范围  
3. 调用 `plot_coronary_outlet_pressure` 显示图形

修改默认演示参数可编辑文件末尾：

```python
if __name__ == "__main__":
    hr, co, frac = 75.0, 5.0, 0.03
    duration_s = 6.0
    ...
```

---

## 11. 常见问题

### Q1：`coronary_outlet_pressure` 二元与三元返回值形状不同？

是的。请根据 `model` 分支处理：

```python
result = coronary_outlet_pressure(t, model="3wk", ...)
if isinstance(result, tuple):
    p_out, p_wk = result
else:
    p_out = result
```

或始终使用 `windkessel2_outlet_pressure` / `windkessel3_outlet_pressure`。

### Q2：为什么二元出口压几乎不变，三元却有明显脉动？

见 [§8.1](#81-时间常数与脉动幅度)。默认大 \(C\) 使二元电路对 \(Q(t)\) 低通滤波；三元通过 \(R_p Q(t)\) 直接注入流量脉动。

### Q3：`q_in` 与 `t` 长度不一致会怎样？

`numpy` 数组广播不自动对齐；请保证 `len(t) == len(q_in)`，否则 `interp` / 索引可能报错或产生错误结果。

### Q4：`coronary_outlet_pressure_stats` 里 `phase_offset` 是否有效？

有效。该键会传入 `coronary_inlet_flow`；Windkessel 参数应使用其他关键字（如 `compliance_ml_per_mmhg`），不要与入口参数名冲突。

### Q5：能否只用 Windkessel、不用 `coronary_inlet`？

可以。仅调用 `windkessel2_outlet_pressure(t, q_in, ...)` 并自备任意 \(Q(t)\)，无需导入入口模块（除非使用 `coronary_outlet_pressure` 且 `q_in=None`）。

### Q6：为何没有静脉压参数？

冠脉/动脉出口管腔压通常在数十至百余 mmHg，静脉压（约数 mmHg）相对可忽略。本模块在 Windkessel 出口端采用**零压参考**，方程与标定中均不含 \(P_{\mathrm{ven}}\)。若需与含静脉压的 demo 求解器严格一致，请使用 `demo/outlet_boundary_condition.py`。

### Q7：与 `demo/units.py` 的关系？

压力、阻力单位与全项目一致（mmHg、mL/s）。修改全局参考压应编辑 `demo/units.py` 中的 `P_INLET_REF_MMHG`，或在调用时显式指定 `r_distal_mmhg_s_per_ml` 覆盖自动标定。

---

## 附录：函数一览

| 函数 | 输入要点 | 输出 |
|------|----------|------|
| `windkessel2_outlet_pressure(t, q_in, ...)` | 必填 \(t, Q\) | `P_out` (mmHg) |
| `windkessel3_outlet_pressure(t, q_in, ...)` | 必填 \(t, Q\) | `(P_out, P_wk)` |
| `coronary_outlet_pressure(t, model, ...)` | 可选自动 \(Q\) | 见 `model` |
| `coronary_outlet_pressure_stats(...)` | 周期数 `duration_cycles` | `dict` 统计 |
| `plot_coronary_outlet_pressure(...)` | 绘图时长 `duration_s` | 无（显示图形） |

相关文档：`coronary_inlet.py`（入口流量）、`demo/units.py`（参考压常量）、`demo/outlet_boundary_condition.py`（1D 求解器联立 Windkessel，含静脉压）。


