# `navier_stokes.py` 使用说明

本文档说明一维血流守恒方程 TVD 求解器 `navier_stokes.py` 的物理模型、数值方法、API 与典型用法。

---

## 目录

1. [概述](#1-概述)
2. [环境与依赖](#2-环境与依赖)
3. [物理模型与单位](#3-物理模型与单位)
4. [数值方法](#4-数值方法)
5. [快速开始](#5-快速开始)
6. [类与 API 参考](#6-类与-api-参考)
7. [血管参数与狭窄建模](#7-血管参数与狭窄建模)
8. [边界条件](#8-边界条件)
9. [求解与结果输出](#9-求解与结果输出)
10. [后处理与 FFR](#10-后处理与-ffr)
11. [参考求解器 `OneDBloodFlowLF`](#11-参考求解器-onedbloodflowlf)
12. [参数调优建议](#12-参数调优建议)
13. [限制与扩展方向](#13-限制与扩展方向)
14. [常见问题](#14-常见问题)

---

## 1. 概述

`navier_stokes.py` 实现**一维可变形动脉管**中的血流传播，以截面积 `A` 和体积流量 `Q` 为守恒变量，适用于：

- 脉动血流与压力波传播
- **各网格中心点压力 \(p(x_i)\) 的数值解**（由 tube law 从 \(A\) 闭合求得）
- 血管狭窄段的流体力学模拟
- 与 FFR（血流储备分数）相关的压降/流量分析（简化版）

**主求解器**：`OneDBloodFlowTVD`  
- 空间：MUSCL 线性重构 + **minmod** 限制器 + **HLL** 黎曼数值通量（TVD 型，抑制振荡）  
- 时间：**SSP-RK2**（二阶强稳定保持 Runge-Kutta）  
- 时间步：CFL 自适应  

**参考求解器**：`OneDBloodFlowLF`（一阶 Lax-Friedrichs，扩散较大，仅建议用于对比）

**向后兼容别名**：

| 别名 | 实际类 |
|------|--------|
| `TVDBloodFlow` | `OneDBloodFlowTVD` |
| `OneDBloodFlow` | `OneDBloodFlowLF` |

---

## 2. 环境与依赖

| 依赖 | 用途 |
|------|------|
| Python 3.8+ | 运行环境 |
| `numpy` | 数组与数值计算 |
| `scipy` | 入口波形辅助函数（`jv`、`trapezoid` 等） |
| `matplotlib` | `visualize()` 绘图 |
| `outlet_boundary_condition.py` | 出口流量分配（Murray 定律等） |

安装示例：

```bash
pip install numpy scipy matplotlib
```

无图形界面环境（如远程服务器）可设置非交互后端后再绘图：

```bash
set MPLBACKEND=Agg    # Windows CMD
export MPLBACKEND=Agg # Linux / macOS
python navier_stokes.py
```

---

## 3. 物理模型与单位

### 3.1 控制方程（守恒形式）

\[
\frac{\partial U}{\partial t} + \frac{\partial F}{\partial x} = S,
\quad
U = \begin{bmatrix} A \\ Q \end{bmatrix}
\]

\[
F = \begin{bmatrix} Q \\ \dfrac{\alpha Q^2}{A} + \dfrac{p(A)\,A}{\rho} \end{bmatrix},
\quad
S = \begin{bmatrix} 0 \\ -\dfrac{8\pi\nu Q}{A} \end{bmatrix}
\]

其中：

- \(A\)：血管截面积  
- \(Q\)：体积流量（截面法向，取正向为沿 \(x\) 出口方向）  
- \(\rho\)：血液密度  
- \(\alpha\)：动量修正系数（层流常取 1.1）  
- \(\nu\)：运动粘度  
- 摩擦项采用 Poiseuille 型阻力 \(-8\pi\nu Q/A\)

### 3.2 管壁弹性律（Tube Law）

\[
p(A) = \beta\left(\sqrt{A} - \sqrt{A_0}\right)
\]

- \(A_0(x)\)：参考（无应力）截面积分布  
- \(\beta(x)\)：管壁弹性系数  

**压力数值解**：求解器在每个时间步由当前 \(A(x_i)\) 在网格中心 \(x_i\) 上计算

\[
p_i^n = p\bigl(A_i^n\bigr),\quad i=0,\ldots,n_x-1
\]

并存入 `solver.p`（当前步）与 `solver.p_history`（历史，与 `solve()` 保存策略一致）。压力**不单独推进偏微分方程**，而与弹性管律闭合。

小扰动下的**压力波速**：

\[
c(A) = \sqrt{\frac{\beta}{2\rho}}\, A^{-1/4}
\]

### 3.3 默认单位制（CGS）

脚本内物理量默认采用 **CGS（厘米–克–秒）** 制，与心血管 1D 模型文献常见约定一致：

| 量 | 符号 | 单位 |
|----|------|------|
| 长度 | \(x\) | cm |
| 截面积 | \(A\) | cm² |
| 体积流量 | \(Q\) | cm³/s |
| 密度 | \(\rho\) | g/cm³（默认 1.05） |
| 压强 | \(p\) | dyne/cm²（= 0.1 Pa） |
| 弹性系数 | \(\beta\) | dyne/cm² |
| 运动粘度 | \(\nu\) | cm²/s |
| 时间 | \(t\) | s |

> **注意**：若从 SI（m, Pa, m³/s）迁移参数，需自行换算；混用单位会导致波速、压强和 FFR 数值错误。

---

## 4. 数值方法

### 4.1 空间离散（TVD / MUSCL）

1. 在内部单元用中心差分估计梯度，经 **minmod** 限制得到斜率 \(\sigma\)  
2. 在界面 \(j+\tfrac{1}{2}\) 重构左右状态：
   - \(U_L = U_j + \tfrac{1}{2}\sigma_j\)
   - \(U_R = U_{j+1} - \tfrac{1}{2}\sigma_{j+1}\)
3. 边界附近模板不足时退化为**分段常数**（一阶）
4. 用 **HLL** 近似黎曼解计算界面数值通量 \(\hat{F}_{j+1/2}\)

### 4.2 时间离散（SSP-RK2）

\[
\begin{aligned}
U^{(1)} &= U^n + \Delta t\, \mathcal{L}(U^n) \\
U^{n+1} &= \tfrac{1}{2}U^n + \tfrac{1}{2}\left(U^{(1)} + \Delta t\, \mathcal{L}(U^{(1)})\right)
\end{aligned}
\]

其中 \(\mathcal{L}(U) = -\partial_x \hat{F} + S\)，边界条件在每个子步施加。

### 4.3 CFL 条件

\[
\Delta t = \min\left(\mathrm{CFL}\cdot\frac{\Delta x}{\max_x(|u|+c)},\; \Delta t_{\max}\right),
\quad \Delta t_{\max}=5\times 10^{-4}\,\mathrm{s}
\]

默认 `cfl=0.5`。CFL 过大可能不稳定；过小则计算变慢。

---

## 5. 快速开始

### 5.1 直接运行内置示例

```bash
python navier_stokes.py
```

将执行：

- 30 cm 直管，`nx=151`  
- 12–18 cm 处狭窄（\(A_0\) 缩至 30%，\(\beta\) 增至 5 倍）  
- 模拟 2.4 s（约 3 个心动周期）  
- 弹出 4 子图并打印最小截面积、最大流量、FFR

### 5.2 最小可运行脚本

```python
import numpy as np
from navier_stokes import OneDBloodFlowTVD

L, nx = 30.0, 101
solver = OneDBloodFlowTVD(length=L, nx=nx, cfl=0.45)

A0 = 0.5 * np.ones(nx)
beta = 1e5 * np.ones(nx)
solver.set_vessel_parameters(A0=A0, beta=beta)

A_hist, Q_hist, p_hist, times = solver.solve(T_total=1.6, save_every=20)
solver.visualize()

print("中心点压力 p(x=0):", p_hist[-1, 0])
print("FFR ≈", solver.compute_ffr())
```

### 5.3 典型工作流

```
创建求解器 → set_vessel_parameters() → solve() → 后处理 / visualize() / compute_ffr()
```

---

## 6. 类与 API 参考

### 6.1 `OneDBloodFlowTVD`（主类）

#### 构造函数

```python
OneDBloodFlowTVD(length, nx, rho=1.05, alpha=1.1, nu=0.0035, cfl=0.5)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `length` | float | 血管长度 (cm) |
| `nx` | int | 空间网格节点数（含两端） |
| `rho` | float | 血液密度 (g/cm³) |
| `alpha` | float | 动量修正系数 |
| `nu` | float | 运动粘度 (cm²/s) |
| `cfl` | float | CFL 数，建议 0.3–0.5 |

**类属性（边界相关）**：

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `heart_rate` | `75` | 入口 `coronary_inlet_flow` 的心率 (bpm)，可在实例化后改 `solver.heart_rate = 60` |

**构造后自动生成的属性**：

| 属性 | shape / 类型 | 说明 |
|------|----------------|------|
| `x` | `(nx,)` | 节点坐标 (cm) |
| `dx` | float | 均匀网格间距 |
| `U` | `(2, nx)` | 当前守恒变量 `[A, Q]` |
| `p` | `(nx,)` | 当前时刻各网格中心压力 (dyne/cm²) |
| `A0`, `beta` | `(nx,)` | 参考面积与弹性系数 |
| `current_time` | float | 当前物理时间 (s) |
| `p_history` | `(n_save, nx)` 或 `None` | `solve()` 后的压力时间序列 |

---

#### `set_vessel_parameters(A0=None, beta=None, stenosis=None)`

设置血管几何与弹性，并重置初始状态 \(A=A_0,\, Q=0\)。

**`stenosis` 格式**：`list[tuple]`，每个元组为：

```python
(x_start, x_end, area_factor, beta_factor)
```

- `x_start`, `x_end`：狭窄区间沿血管的位置 (cm)，**闭区间** `[x_start, x_end]`  
- `area_factor`：将该区间内 `A0` **乘以**该因子（`0.3` → 参考面积变为原来的 30%，约 70% 面积狭窄）  
- `beta_factor`：将该区间内 `beta` **乘以**该因子（硬化）

示例：在 12–18 cm 处施加重度狭窄并硬化管壁：

```python
solver.set_vessel_parameters(
    A0=0.5 * np.ones(nx),
    beta=1e5 * np.ones(nx),
    stenosis=[(12.0, 18.0, 0.3, 5.0)],
)
```

可叠加多段：

```python
stenosis=[
    (5.0, 8.0, 0.6, 2.0),
    (20.0, 22.0, 0.4, 3.0),
]
```

---

#### `solve(T_total, save_every=20)`

| 参数 | 说明 |
|------|------|
| `T_total` | 总模拟时间 (s) |
| `save_every` | 每多少个时间步保存一帧到历史数组 |

**返回值**：`(A_history, Q_history, p_history, times)`

| 输出 | shape | 说明 |
|------|--------|------|
| `A_history` | `(n_save, nx)` | 截面积时间序列 |
| `Q_history` | `(n_save, nx)` | 流量时间序列 |
| `p_history` | `(n_save, nx)` | **各网格中心压力**时间序列 (dyne/cm²) |
| `times` | `(n_save,)` | 对应时刻 (s) |

**同时写入对象属性**：`solver.times`, `solver.A_history`, `solver.Q_history`, `solver.p_history`；最后时刻的瞬时压力亦在 `solver.p` 中。

`p_history[k, i]` 表示第 `k` 个保存时刻、位置 `x[i]` 处的压力数值解。

> 保存帧数约为 `1 + floor(n_steps / save_every)`，与总步数 `n_steps` 有关（由 CFL 自适应决定）。

---

#### `visualize()`

绘制 2×2 子图，**每张图均带图例/色标且不遮挡数据区**：

1. \(A(t,x)\) 时空云图 — 右侧 **colorbar**（标注 A 单位）  
2. **\(\langle p(x)\rangle_t\)** — 对保存时刻在 \(t\) 上取平均后的压力沿程曲线 \(p\)–\(x\)  
3. 血管中点 \(Q(t)\)、\(p(t)\) — 双 y 轴 + **子图底部** 横向图例  
4. 末时刻沿程 \(A,\,Q,\,p\) — 双 y 轴 + **子图底部** 横向图例  

需先调用 `solve()`（依赖 `p_history`）。画布边距经 `subplots_adjust` 预留，避免图例压住坐标轴标签。

返回 `matplotlib.figure.Figure`，并调用 `plt.show()`。

---

#### `compute_ffr(n_tail=50)`

简化 FFR 估计：

\[
\mathrm{FFR} \approx \frac{\langle p \rangle_{\text{distal}}}{\langle p \rangle_{\text{proximal}}}
\]

- 取 `A_history` 最后 `n_tail` 帧的时间平均截面积  
- 由 tube law 换算平均压力  
- 近端：前 1/4 网格；远端：后 1/4 网格  

> 此为**教学/原型级**指标，与临床导管测量 FFR 的定义与标定不同，结果仅供相对比较。

---

#### 其他常用方法

| 方法 | 说明 |
|------|------|
| `apply_boundary_conditions(U)` | 施加 §8 入口/出口边界，返回更新后的 `U` |
| `update_pressure(U=None)` | 由当前或给定 `U` 更新 `self.p`（各中心点） |
| `pressure_from_area(A, A0, beta)` | 静态方法，由面积算压力 |
| `pressure_field(A)` | 对整条血管向量 `A` 在**各中心点**算 `p`，与 `self.x` 同长 |
| `wave_speed(A, beta)` | 压力波速 \(c(A)\) |
| `physical_flux(U, ...)` | 守恒通量 \(F(U)\) |
| `source_term(U)` | 单点源项 \(S(U)\) |
| `ssp_rk2_step(dt)` | 推进一个时间步（高级用法） |
| `adaptive_dt(U)` | 当前 CFL 时间步 |

---

### 6.2 模块级函数

#### `minmod(a, b)`

向量化 minmod 限制器，支持 NumPy 数组广播。

---

## 7. 血管参数与狭窄建模

### 7.1 沿程非均匀 `A0` 与 `beta`

`A0`、`beta` 必须与 `nx` 等长，一一对应网格节点：

```python
x = solver.x
A0 = 0.4 + 0.1 * np.sin(np.pi * x / L)   # 示例：沿程变化
beta = 8e4 + 2e4 * (x / L)
solver.set_vessel_parameters(A0=A0, beta=beta)
```

### 7.2 狭窄几何含义

若健康段 \(A_0 = 0.5\,\mathrm{cm}^2\)，`area_factor=0.3` 后狭窄段 \(A_0=0.15\,\mathrm{cm}^2\)。

**面积狭窄率**（按参考面积）：

\[
\text{stenosis\%} \approx \left(1 - \frac{A_{0,\min}}{A_{0,\mathrm{ref}}}\right)\times 100\%
\]

### 7.3 参数数量级参考

| 参数 | 典型量级 | 备注 |
|------|----------|------|
| `A0` | 0.05–1.0 cm² | 取决于血管半径 |
| `beta` | \(10^4\)–\(10^6\) dyne/cm² | 越大管壁越硬、波速越快 |
| `Q`（入口平均量级） | 约 3–4 cm³/s | 由 `coronary_inlet_flow` 按心输出量估算（见 §8） |
| `length` | 10–50 cm | 股动脉–冠脉尺度原型 |
| `heart_rate` | 75 bpm（类属性） | `OneDBloodFlowTVD.heart_rate`，用于入口波形 |

---

## 8. 边界条件

边界在 `OneDBloodFlowTVD.apply_boundary_conditions()` 中施加，并在每个 **SSP-RK2 子步**（预测步、校正步）调用。实现依赖：

- **`coronary_inlet_flow()`**（`navier_stokes.py`）：入口脉动流量  
- **`coronary_outlet_flow_distribution()`**（`outlet_boundary_condition.py`）：出口流量分配  

> **单位**：`coronary_inlet_flow` 注释为 mL/s；在 CGS 中 **1 mL = 1 cm³**，故数值上可与求解器中的 cm³/s 直接对接。

### 8.1 施加位置与变量

| 边界 | 网格索引 | 施加变量 | 说明 |
|------|----------|----------|------|
| 入口 \(x=0\) | `U[:, 0]` | \(A,\,Q\) | 定流量 + 面积外推 |
| 出口 \(x=L\) | `U[:, -1]` | \(A,\,Q\) | 面积按入口缩放 + Murray 型出口流量 |

### 8.2 入口（\(x=0\)）

**算法（源码逻辑）**：

```python
Q_in = coronary_inlet_flow(
    t, heart_rate=self.heart_rate, waveform='physiological'
)
U[0, 0] = U[0, 1]      # 截面积：内侧节点外推（零梯度）
U[1, 0] = Q_in         # 流量：生理脉动波形
```

#### `coronary_inlet_flow` 模型要点

- **平均冠脉流量**（由心输出量估算）  
  \[
  \bar{Q} = \mathrm{CO}\times\frac{1000}{60}\times f_{\mathrm{cor}}
  \]
  默认：`cardiac_output=5.0` L/min，`coronary_fraction=0.04`（约 4% 心输出量进入冠脉）  
  → \(\bar{Q}\approx 3.33\) mL/s（≈ cm³/s）

- **心动周期**：\(T = 60/\mathrm{HR}\)，默认 `heart_rate=75` → \(T=0.8\) s（类属性 `OneDBloodFlowTVD.heart_rate`）

- **`waveform='physiological'`**（当前边界所用）：  
  - 直流分量 + 多谐波正弦叠加（舒张期优势、收缩期减少的冠脉特征）  
  - 谐波系数见 `coronary_inlet_flow` 内 `harmonics` 列表  
  - 截断：`Q = max(Q,\,0.1\bar{Q})`，保证非负

- **其它波形**（需改源码 `waveform` 参数）：  
  - `'simplified'`：收缩/舒张分段正弦  
  - 默认 `'sine'` 分支：单频正弦脉动  

参考文献已写在 `coronary_inlet_flow` 的 docstring（Womersley、Nichols & O'Rourke、Kim et al. 等）。

### 8.3 出口（\(x=L\)）

**算法（源码逻辑）**：

```python
U[0, -1] = U[0, 0] * 0.8
Q_out = coronary_outlet_flow_distribution(Q_in, U[0, [-1]])
U[1, -1] = Q_out
```

#### 截面积 \(A\)

- 出口参考截面积取为**入口截面积的 80%**：\(A_{L}=0.8\,A_{0}\)（此处 \(A_{0}\) 指边界点 `U[0,0]` 的当前值，非参考几何 `self.A0`）  
- 用于 1D 单出口管路的**几何缩放**，模拟出口段略细于入口

#### 流量 \(Q\)

调用 `coronary_outlet_flow_distribution(Q_total, outlet_radii, method='murray')`（默认 **Murray 定律** \(Q\propto r^3\)）：

1. 以当前入口流量 `Q_in` 为总流量 `Q_total`  
2. `outlet_radii` 传入为 `U[0, [-1]]`（出口处当前截面积标量；在**单出口 1D** 情形下仅一个“出口”，权重归一化后为 1）  
3. 返回标量 `Q_out` 赋给 `U[1, -1]`  

**1D 单管含义**：只有一个出口时，Murray 分配后 **`Q_out = Q_in`**（质量守恒与单分支一致）。若扩展为多出口 1D 或多分支网络，应传入各分支半径/面积数组，并接 `coronary_outlet_flow_with_impedance()` 等（见 `outlet_boundary_condition.py`，当前求解器未调用）。

#### `coronary_outlet_flow_distribution` 可选参数（改出口算法时用）

| 参数 | 说明 |
|------|------|
| `method='murray'` | \(w_i \propto r_i^3\)（默认） |
| `method='area'` | \(w_i \propto r_i^2\) |
| `method='murray_modified'` | \(w_i \propto r_i^{2.7}\) |
| `method='custom'` | 自定义 `flow_fractions` |
| `cardiac_phase` | `'systole'` / `'diastole'`，调整小血管权重 |

### 8.4 与旧版边界的区别

| 项目 | 旧版（文档曾描述） | **当前实现** |
|------|-------------------|--------------|
| 入口 \(Q\) | 正弦平方，固定 5 cm³/s 峰值 | `coronary_inlet_flow` 生理多谐波 + 心输出量标定 |
| 出口 \(Q\) | 非反射特征外推 | Murray 型流量分配（1D 单出口 ≈ \(Q_{\mathrm{in}}\)） |
| 出口 \(A\) | 内侧外推或特征配点 | **`0.8 × A(入口)`** |

### 8.5 物理约束（非边界，但每步执行）

`enforce_physical_bounds()` 将 \(A\) 限制为不低于 `0.1 * A0`（参考几何 `self.A0`），防止面积过小导致除零或负值。

### 8.6 自定义边界

1. **改心率 / 波形 / 心输出量**：编辑 `apply_boundary_conditions` 中 `coronary_inlet_flow(...)` 的参数，或修改类属性 `OneDBloodFlowTVD.heart_rate`  
2. **改出口面积关系**：修改 `U[0, -1] = U[0, 0] * 0.8` 中的系数或改为外推  
3. **改出口流量分配**：在 `apply_boundary_conditions` 中更换 `coronary_outlet_flow_distribution` 的 `method`、`cardiac_phase`，或改用同文件中的 `coronary_outlet_flow_with_impedance`  
4. **不改源码**：子类化 `OneDBloodFlowTVD` 并重写 `apply_boundary_conditions`

---

## 9. 求解与结果输出

### 9.1 控制台输出

```
TVD 求解: T=2.4s, nx=151, dx=0.2000 cm
完成: 21712 步, t=2.4000 s
```

### 9.2 自行读取场数据

```python
import matplotlib.pyplot as plt

# 最后一帧沿程分布（含压力）
plt.plot(solver.x, solver.A_history[-1], label="A")
plt.plot(solver.x, solver.p_history[-1], label="p")
plt.legend()
plt.show()

# 任意网格中心的时间序列
i = 50
plt.plot(solver.times, solver.Q_history[:, i], label="Q")
plt.plot(solver.times, solver.p_history[:, i], label="p")
plt.xlabel("t (s)")
plt.legend()
plt.title(f"x = {solver.x[i]:.1f} cm")
plt.show()
```

### 9.3 导出为 NumPy 文件

```python
np.savez(
    "result.npz",
    x=solver.x,
    times=solver.times,
    A=solver.A_history,
    Q=solver.Q_history,
    p=solver.p_history,
    A0=solver.A0,
    beta=solver.beta,
)
```

---

## 10. 后处理与 FFR

### 10.1 读取网格中心压力

**推荐**：直接使用 `solve()` 返回的 `p_history`（与保存时刻、中心点一一对应）。

```python
A, Q, p, times = solver.solve(T_total=2.0, save_every=20)

# 最后时刻、全部中心点
p_last = p[-1]           # 或 solver.p_history[-1]、solver.p
x = solver.x             # p_last[i] 对应 x[i]

# 某一时刻、沿程压力分布
import matplotlib.pyplot as plt
plt.plot(x, p[10])
plt.ylabel("p (dyne/cm²)")
plt.show()
```

若仅需由面积闭合计算（不依赖历史），可用：

```python
p = solver.pressure_field(solver.A_history[-1])
```

### 10.2 沿程压降

```python
# 使用时间平均压力（直接用 p_history）
p_mean = np.mean(solver.p_history[-50:], axis=0)
delta_p = p_mean[0] - p_mean[-1]
print(f"平均压降 ≈ {delta_p:.1f} dyne/cm²")
```

### 10.3 FFR

```python
ffr = solver.compute_ffr(n_tail=80)
print(f"FFR ≈ {ffr:.3f}")
```

解释建议：

- FFR 接近 1：远端与近端平均压差别小  
- FFR 明显低于 1：狭窄或阻力导致远端灌注压相对降低（在本简化模型意义下）

---

## 11. 参考求解器 `OneDBloodFlowLF`

一阶 Lax-Friedrichs，数值扩散大、界面模糊，**不推荐**用于狭窄/波传播精细分析。

```python
from navier_stokes import OneDBloodFlowLF

lf = OneDBloodFlowLF(length=30.0, nx=101, cfl=0.4)
lf.set_vessel_parameters(0.5 * np.ones(101), 1e5 * np.ones(101))
A, Q, p, t = lf.solve(T_total=1.0, save_every=50)
```

| 对比项 | `OneDBloodFlowTVD` | `OneDBloodFlowLF` |
|--------|--------------------|-------------------|
| 精度 | 二阶（空间/时间） | 一阶 |
| 间断分辨率 | 较好（TVD） | 较差 |
| 速度 | 中等（向量化 HLL） | 较慢（Python 循环） |
| 狭窄 | 支持 `stenosis` | 仅均匀 `A0`,`beta` |

---

## 12. 参数调优建议

| 目标 | 建议 |
|------|------|
| 提高精度 | 增大 `nx`；略减小 `cfl`（如 0.35） |
| 加快计算 | 减小 `nx` 或 `T_total`；增大 `save_every` |
| 不稳定 / 振荡 | 减小 `cfl`；检查 `beta` 是否过大；确认单位一致 |
| 结果光滑但钝 | 正常；若需更锐利的间断可尝试更小 `cfl` 或更细网格 |
| 多周期稳态 | `T_total` 取 2–4 个心动周期（默认 HR=75 → \(T=0.8\) s，建议 ≥1.6 s） |

**计算量粗估**：每步 2 次 `rhs`（SSP-RK2），每步 \(O(n_x)\)。`nx=151`, `T_total=2.4` s 时约数万步，普通 PC 上通常数十秒量级。

---

## 13. 限制与扩展方向

当前版本的已知简化：

1. **边界条件**在 `apply_boundary_conditions` 中写死，无独立 API；入口/出口算法见 §8，修改需改源码或子类化  
2. **出口**为 Murray 流量分配 + 面积比例缩放，**非** Windkessel / 非反射特征边界；1D 单出口时 \(Q_{\mathrm{out}}\approx Q_{\mathrm{in}}\)  
3. **摩擦项**为线性 Poiseuille 型，未含湍流或曲率损失  
4. **压力梯度**已并入守恒通量 \(F\)，非独立半隐式管律耦合  
5. **FFR** 为基于 1D 平均压的比值原型，非临床标准  
6. **分叉、泄漏、外周阻力网络** 未建模；`coronary_outlet_flow_with_impedance` 已提供但未接入主求解器  

常见扩展方式：

- 子类化 `OneDBloodFlowTVD`，重写 `apply_boundary_conditions`  
- 在 `solve` 循环外增加参数扫描或优化  
- 将 `stenosis` 参数与影像分割得到的半径轮廓对接  

---

## 14. 常见问题

**Q：`import` 或运行报错 `No module named numpy`**  
A：安装依赖：`pip install numpy matplotlib`。

**Q：`visualize()` 无窗口弹出**  
A：检查是否无 GUI 环境；使用 `MPLBACKEND=Agg` 并将 `fig.savefig("out.png")` 加到 `visualize()` 末尾。

**Q：FFR > 1 或明显不合理**  
A：可能尚未达到周期性稳态、近远端分区不合适，或狭窄极轻/极重导致数值失真；延长 `T_total`、检查 `A0/beta` 与单位。

**Q：与旧代码 `TVDBloodFlow` 不兼容？**  
A：使用别名 `from navier_stokes import TVDBloodFlow`，即 `OneDBloodFlowTVD`；旧版逐点循环 API 已移除，请按本文档 `solve()` 流程调用。

**Q：`solve()` 以前只返回 3 个值？**  
A：现返回 `(A_history, Q_history, p_history, times)` 四个数组；`p_history` 为各网格中心压力的历史记录。

**Q：如何修改入口流量波形或强度？**  
A：在 `apply_boundary_conditions` 中调整 `coronary_inlet_flow(...)` 的参数，例如 `cardiac_output`、`coronary_fraction`、`waveform`，或修改 `OneDBloodFlowTVD.heart_rate`（默认 75 bpm）。

**Q：如何修改出口边界？**  
A：修改 `U[0,-1]` 的面积公式（当前 `0.8*U[0,0]`），或在 `coronary_outlet_flow_distribution` 中更换 `method` / 传入正确的出口半径数组；多分支场景见 `outlet_boundary_condition.py`。

**Q：入口流量单位与求解器一致吗？**  
A：`coronary_inlet_flow` 返回 mL/s；CGS 中 1 mL = 1 cm³，数值上与 cm³/s 一致。

**Q：`save_every` 与存储内存**  
A：历史数组大小 \(\propto (n_{\mathrm{steps}}/\texttt{save\_every}) \times n_x\)。长时间模拟可增大 `save_every`。

---

## 附录：文件结构速查

```
navier_stokes.py
├── coronary_inlet_flow()  # 冠脉入口生理流量波形
├── minmod()                 # 限制器
├── OneDBloodFlowTVD         # 主求解器 ★（apply_boundary_conditions）
├── OneDBloodFlowLF          # LF 参考求解器
├── TVDBloodFlow             # 别名 → OneDBloodFlowTVD
├── OneDBloodFlow            # 别名 → OneDBloodFlowLF
└── if __name__ == "__main__"  # 内置示例

outlet_boundary_condition.py
├── coronary_outlet_flow_distribution()   # 出口 Murray/面积分配（已接入）
├── coronary_outlet_flow_with_impedance() # LAD/LCx/RCA 分支分配（未接入）
└── …                                     # 其它辅助接口
```

---

*文档版本与 `navier_stokes.py` 同步；若脚本 API 有变更，请以源码 docstring 为准。*
