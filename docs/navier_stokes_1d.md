# 一维冠脉血流求解器说明（`navier_stokes_1d.py`）

本文档说明 `navier_stokes_1d.py` 的物理模型、数值方法、输入输出、API 用法及与 `coronary_inlet.py`、`coronary_outlet.py` 的衔接方式。

---

## 目录

1. [概述](#1-概述)
2. [环境与依赖](#2-环境与依赖)
3. [物理模型与单位](#3-物理模型与单位)
4. [数值方法](#4-数值方法)
5. [边界条件](#5-边界条件)
6. [快速开始](#6-快速开始)
7. [输入数据：管腔面积剖面](#7-输入数据管腔面积剖面)
8. [参数类 `BloodFlowParameters`](#8-参数类-bloodflowparameters)
9. [求解器 `NavierStokes1D`](#9-求解器-navierstokes1d)
10. [后处理与 FFR](#10-后处理与-ffr)
11. [与离线 Windkessel 对比](#11-与离线-windkessel-对比)
12. [参数调优建议](#12-参数调优建议)
13. [限制与扩展方向](#13-限制与扩展方向)
14. [常见问题](#14-常见问题)
15. [文件关系速查](#15-文件关系速查)

---

## 1. 概述

`navier_stokes_1d.py` 实现**单支冠状动脉**在轴向 \(x \in [0, L]\) 上的一维可变形管血流传播。以截面积 \(A(x,t)\) 与体积流量 \(Q(x,t)\) 为求解变量，**管腔标量压** \(p(x,t)\) 由弹性 tube law 从 \(A\) 闭合，入口/出口边界分别由专用模块提供。

| 项目 | 说明 |
|------|------|
| 控制方程 | 1D 守恒型 Navier-Stokes（含 Poiseuille 摩擦源项） |
| 管壁 | \(p = P_{\mathrm{ref}} + \beta(\sqrt{A}-\sqrt{A_0})\) |
| 空间离散 | 有限体积 + MUSCL + minmod TVD + HLL |
| 时间离散 | SSP-RK2 |
| 入口 | `coronary_inlet.coronary_inlet_flow` |
| 出口 | 与 `coronary_outlet` 一致的 Windkessel ODE（在线耦合） |
| 主输入 | 沿程参考管腔面积 \(A_0(x)\) + `BloodFlowParameters` |

> 本模块**不依赖** `demo/navier_stokes.py`；`demo/` 目录下为早期原型，接口与边界实现不同，请勿混用。

---

## 2. 环境与依赖

| 依赖 | 用途 |
|------|------|
| Python 3.10+ | 类型注解、`float \| None` 等语法 |
| `numpy` | 数组与数值计算 |
| `matplotlib` | `plot_results()` 绘图（可选） |
| `coronary_inlet.py` | 入口脉动流量 |
| `coronary_outlet.py` | Windkessel 参数约定与离线参考压力 |
| `coronary_constants.py` | 共享公开全局常量（`P_INLET_REF_MMHG`、`MMHG_TO_DYNE_PER_CM2` 等） |

安装示例：

```bash
pip install numpy matplotlib
```

无图形界面时：

```bash
# Windows PowerShell
$env:MPLBACKEND = "Agg"
python navier_stokes_1d.py
```

---

## 3. 物理模型与单位

### 3.1 守恒方程

\[
\frac{\partial U}{\partial t} + \frac{\partial F}{\partial x} = S,
\quad
U = \begin{bmatrix} A \\ Q \end{bmatrix}
\]

\[
F = \begin{bmatrix} Q \\ \dfrac{\alpha Q^2}{A} + \dfrac{A}{\rho}p(A) \end{bmatrix},
\quad
S = \begin{bmatrix} 0 \\ -\dfrac{8\pi\mu Q}{A} \end{bmatrix}
\]

| 符号 | 含义 |
|------|------|
| \(A\) | 血管截面积 |
| \(Q\) | 体积流量（沿 \(x\) 正向为出口方向） |
| \(\rho\) | 血液密度 |
| \(\alpha\) | 动量修正系数（层流常取 1.1） |
| \(\mu\) | 运动粘度 |
| 摩擦项 | Poiseuille 型 \(-8\pi\mu Q/A\) |

通量中的 \(p(A)\) 在计算动量项时由 **mmHg 换算为 cgs（dyne/cm²）**，系数见 `coronary_constants.MMHG_TO_DYNE_PER_CM2`（1333.22）。

### 3.2 管壁弹性律（Tube Law）

\[
p(A, x) = P_{\mathrm{ref}} + \beta(x)\,\bigl(\sqrt{A} - \sqrt{A_0(x)}\bigr)
\]

- \(A_0(x)\)：用户给定的**参考管腔面积**（无应力或影像分割得到的基线面积）
- \(\beta(x)\)：管壁刚度（mmHg / cm）
- \(P_{\mathrm{ref}}\)：参考压力（默认 `coronary_constants.P_INLET_REF_MMHG`，80 mmHg）

**压力不单独求解散射方程**，每个时间步由当前 \(A\) 通过上式在网格中心得到 \(p_i\)。

#### 3.2.1 管腔压 \(p\) 的物理含义

\(p\) 是**各向同性的热力学标量压**（mmHg），**不是**轴向或径向的应力矢量分量，也**不应**与「垂直于截面（轴向）」或「管壁法向（径向）」混为一谈——后两者在几何上互相垂直。

| 用途 | 机制 |
|------|------|
| **tube law** | 将标量 \(p\) 与截面积 \(A\) 耦合，描述管腔**径向胀缩**（壁弹性） |
| **动量通量** \(\partial(pA/\rho^*)/\partial x\) | 沿 \(x\) 的**压力梯度**驱动轴向体积流量 \(Q\) |
| **`pressure[i]`** | 轴向位置 \(x_i\) 处的管腔压取值；`p(x)` 表示沿程分布，非方向性矢量 |

小扰动压力波速（与 §4.3 CFL 及 `_wave_speed` 一致）：

\[
c(A) = \sqrt{\frac{\beta\sqrt{A}}{2\rho^*}},
\qquad
\rho^* = \rho / 1333.22 \;\text{（mmHg 动量耦合有效密度）}
\]

### 3.3 默认单位制（CGS + mmHg 展示）

| 量 | 单位 |
|----|------|
| 轴向位置 \(x\) | cm |
| 截面积 \(A\) | cm² |
| 体积流量 \(Q\) | cm³/s |
| 入口模块输出 | mL/s（**1 mL = 1 cm³**，数值与 cm³/s 一致） |
| 密度 \(\rho\) | g/cm³（默认 1.05） |
| 粘度 \(\mu\) | cm²/s |
| 压力 \(p\)（`pressure`、`history_pressure`） | mmHg；沿 \(x\) 各点的**管腔标量压**，非方向分量 |
| 时间 \(t\) | s |

---

## 4. 数值方法

### 4.1 空间：有限体积 + MUSCL + TVD + HLL

1. 在单元中心存储 \(U_j^n\)，界面 \(j+\tfrac{1}{2}\) 用中心差分估计梯度，经 **minmod** 限制得斜率 \(\sigma_j\)。
2. MUSCL 重构：
   - \(U_L = U_j + \tfrac{1}{2}\sigma_j\)
   - \(U_R = U_{j+1} - \tfrac{1}{2}\sigma_{j+1}\)
3. 重构后的 \(A\) 不低于 \(0.1\,A_0\)（防止非物理负面积）。
4. 内部界面用 **HLL** 近似黎曼解得到数值通量 \(\hat{F}_{j+1/2}\)；边界界面用物理通量 \(F(U)\)。

### 4.2 时间：SSP-RK2

\[
\begin{aligned}
U^{(1)} &= \mathrm{BC}\bigl(U^n + \Delta t\,\mathcal{L}(U^n)\bigr) \\
U^{n+1} &= \mathrm{BC}\left(\tfrac{1}{2}U^n + \tfrac{1}{2}\bigl(U^{(1)} + \Delta t\,\mathcal{L}(U^{(1)})\bigr),\ \text{advance outlet}\right)
\end{aligned}
\]

- \(\mathcal{L}(U) = -\partial_x \hat{F} + S\)
- **边界条件**在每个子步施加；Windkessel 状态仅在**完整步末**推进（`advance_outlet=True`）。

### 4.3 CFL 条件

时间步长由 **CFL（Courant–Friedrichs–Lewy）条件**自适应选取，保证显式双曲部分（对流 + 弹性波）在数值上稳定。实现见 `NavierStokes1D._stable_timestep` 与 `_wave_speed`。

\[
\Delta t = \min\left(
  \mathrm{CFL}\cdot\frac{\Delta x}{\lambda_{\max}},\;
  \Delta t_{\max}
\right),
\qquad
\lambda_{\max} = \max_{x}\bigl(\,|u(x)| + c(x)\,\bigr)
\]

#### 公式中各符号

| 符号 | 含义 | 单位 | 代码/来源 |
|------|------|------|-----------|
| \(\Delta x\) | 轴向网格间距 \(L/(n_x-1)\) | cm | `solver.dx` |
| \(u(x)\) | **轴向流体速度**，\(u = Q/A\) | cm/s | `state[1] / state[0]` |
| \(c(x)\) | **小扰动弹性波速**（管壁顺应性引起的压力–面积波沿 \(x\) 传播的速度） | cm/s | `_wave_speed` |
| \(\lambda_{\max}\) | 全网格上特征速度上界 \(\|u\|+c\) 的最大值 | cm/s | `np.max(np.abs(u) + c)` |
| \(\mathrm{CFL}\) | 稳定性安全系数，\(0<\mathrm{CFL}\lesssim 1\) | — | `BloodFlowParameters.cfl`（默认 0.5） |
| \(\Delta t_{\max}\) | 时间步**硬上限**（避免单步过大、便于与 Windkessel 耦合） | s | `BloodFlowParameters.dt_max_s`（默认 \(5\times10^{-4}\)） |

#### \(u\)：对流速度

\[
u = \frac{Q}{A}
\]

- \(Q\)：该网格点的体积流量（cm³/s）
- \(A\)：该网格点的管腔截面积（cm²）
- \(u\) 描述**血液沿血管轴向 \(x\) 方向的输运**；\(|u|\) 越大，信息沿 \(x\) 传播越快，所需 \(\Delta t\) 越小

#### \(c\)：弹性波速

由 tube law 线性化得到的小扰动波速（与 HLL 求解器中 `_wave_speed` 一致）：

\[
c(A) = \sqrt{\frac{\beta\sqrt{A}}{2\rho^*}},
\qquad
\rho^* = \frac{\rho}{1333.22}
\]

| 量 | 说明 |
|----|------|
| \(\beta(x)\) | 管壁刚度（mmHg/√cm）；\(\beta\) 越大 → 管壁越硬 → \(c\) 越大 |
| \(A\) | 当前截面积（cm²） |
| \(\rho\) | 血液密度（g/cm³，默认 1.06） |
| \(\rho^*\) | 与 mmHg 制管腔压配对的**动量有效密度**（`momentum_rho()`） |

物理上，\(c\) 刻画**压力扰动沿管轴传播**的快慢：顺应性越小（\(\beta\) 越大）或管腔越大，波速越高，CFL 对 \(\Delta t\) 的限制越严。

#### 为何取 \(|u|+c\)

一维可变形管方程在双曲意义下有两族特征线，其有效传播速度约为 \(u \pm c\)。CFL 条件要求在一个时间步内，对流与弹性波**不应跨越超过约一个网格单元**：

\[
\Delta t \lesssim \frac{\Delta x}{|u| + c}
\]

取全网格 \(\max_x(|u|+c)\) 作为最严格（最小 \(\Delta t\)）的约束；再乘以 \(\mathrm{CFL}<1\) 留出数值安全裕度（MUSCL–HLL + SSP-RK2 仍建议 \(\mathrm{CFL}\approx 0.35\)–\(0.5\)）。

#### 默认参数与调参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cfl` | 0.5 | 减小可抑制振荡、提高稳定性，但步数增多 |
| `dt_max_s` | \(5\times10^{-4}\) s | 即使 \(\lambda_{\max}\) 很小也不超过此步长 |

**典型量级**（冠脉尺度）：\(u \sim 10\)–\(50\) cm/s，\(c \sim 10\)–\(30\) cm/s，\(\Delta x \sim 0.05\)–\(0.2\) cm → \(\Delta t\) 常为 \(10^{-4}\)–\(10^{-3}\) s 量级，常由 `dt_max_s` 截断。

**退化情况**：若 \(\lambda_{\max} < 10^{-14}\)（近乎静止），代码回退 \(\Delta t = 10^{-4}\) s，避免除零。

---

## 5. 边界条件

### 5.1 入口（\(x=0\)）

由 `coronary_inlet.coronary_inlet_flow` 提供 \(Q_{\mathrm{in}}(t)\)：

```text
U[1, 0] = Q_in(t)          # 定流量
U[0, 0] = U[0, 1]          # 截面积零梯度外推
```

入口参数通过 `BloodFlowParameters` 传入（心率、心输出量、冠脉分流比例、傅里叶系数等）。波形细节见 [`coronary_inlet_说明.md`](coronary_inlet_说明.md)。

平均冠脉流量：

\[
Q_{\mathrm{mean}} = \frac{\mathrm{CO}\times 1000 \times f_{\mathrm{cor}}}{60}
\quad\text{(mL/s)}
\]

### 5.2 出口（\(x=L\)）— Windkessel 耦合

与 `coronary_outlet.py` 文档中的微循环模型一致，求解器内维护状态 \(P_{\mathrm{wk}}\)：

**二元 Windkessel（`outlet_windkessel="2wk"`）**

\[
C\,\frac{\mathrm{d}P_{\mathrm{wk}}}{\mathrm{d}t}
= Q_{\mathrm{drive}} - \frac{P_{\mathrm{wk}}}{R_d},
\quad
P_{\mathrm{lumen}} = P_{\mathrm{wk}}
\]

**三元 Windkessel（`outlet_windkessel="3wk"`）**

\[
C\,\frac{\mathrm{d}P_{\mathrm{wk}}}{\mathrm{d}t}
= Q_{\mathrm{drive}} - \frac{P_{\mathrm{wk}}}{R_d},
\quad
P_{\mathrm{lumen}} = P_{\mathrm{wk}} + R_p\, Q_{\mathrm{drive}}
\]

其中：

\[
Q_{\mathrm{drive}} = Q(L)
\]

即 Windkessel ODE 由**出口管腔 PDE 流量**驱动，实现 0D–1D 质量闭合。

\[
Q_{\mathrm{venous}} = \frac{P_{\mathrm{wk}}}{R_d}
\]

仅在 \(dP_{\mathrm{wk}}/dt=0\) 时才有 \(Q(L) \approx Q_{\mathrm{venous}}\)；脉动过程中二者可分离。

出口边界施加（`_apply_boundaries`）：

```text
Q(L)   ← 保留 PDE 演化值（不强制为 P_wk/R_d）
A(L)   ← tube law 反解，使 p(A(L)) = P_lumen
P_wk   ← 步末 advance：C·dP_wk/dt = Q_drive − P_wk/R_d
```

**物理回路**：

```text
入口 Q_in(t) ──► 1D 管腔 PDE ──► Q(L) ──► Windkessel(C, P_wk) ──► Q_venous = P_wk/R_d ──► 静脉
                                      └──► P_lumen ──► A(L) via tube law
```

默认 \(R_d = P_{\mathrm{ref}} / Q_{\mathrm{mean}}\)，\(R_p = 0.12\, R_d\)（与 `coronary_outlet` 默认标定一致）。详见 [`coronary_outlet.md`](coronary_outlet.md)。

### 5.3 物理下限

每步将 \(A\) 限制为不低于 `0.8 * area_ref`（`_clip_area`）；MUSCL 重构阶段另有 `0.1 * area_ref` 下限，避免除零与非物理负面积。

### 5.4 边界变量 vs 内部自由度（示意图）

有限体积离散在单元中心存储 \(U_j=[A_j,Q_j]^\mathsf{T}\)。边界条件**不直接修改通量公式**，而是在每个 RK 子步先强制边界节点上的 \(U\)，再由 `_spatial_operator` 用边界上的物理通量 \(F(U)\) 作为 `f_face[0]`、`f_face[-1]`。

#### 5.4.1 沿程：prescribed / free / derived

```mermaid
flowchart LR
    subgraph IN["入口 x = 0"]
        direction TB
        IN_Q["Q₀ = Q_in(t)  ✓ prescribed"]
        IN_A["A₀ = A₁       ~ extrapolated"]
        IN_P["p₀ = tube_law(A₀)  derived"]
    end

    subgraph INT["内部 j = 1 … nx−2"]
        direction TB
        INT_A["A_j  ✓ free"]
        INT_Q["Q_j  ✓ free"]
        INT_P["p_j = tube_law(A_j)  derived"]
    end

    subgraph OUT["出口 x = L"]
        direction TB
        WK["P_wk  ✓ ODE state"]
        OUT_P["P_lumen = f(P_wk, Q_drive)  ✓ prescribed"]
        OUT_Q["Q(L)  ✓ free (PDE)"]
        OUT_QV["Q_venous = P_wk/R_d  ◇ diagnostic"]
        OUT_A["A(L) = inv_tube_law(P_lumen)  derived"]
    end

    IN -->|"HLL 通量传播"| INT
    INT -->|"HLL 通量传播"| OUT
```

**图例**

| 标记 | 含义 |
|------|------|
| prescribed（图中红色节点） | 边界**直接强制**的变量 |
| free（图中绿色节点） | **内部自由度**，由 PDE 时间推进 |
| derived（图中紫色节点） | **派生量**：每步由 tube law 从 \(A\) 得到标量 \(p\)，不单独求解 |
| extrapolated / ODE state（图中黄色节点） | \(A_0\) 外推、或 Windkessel 状态 \(P_{\mathrm{wk}}\) |

#### 5.4.2 1D 管段上的变量角色

```text
  入口 x=0              内部单元 j=1…nx−2              出口 x=L
  ─────────            ─────────────────────           ─────────
       │                        │                          │
  Q ───●─── prescribed          ○─── free                   ○─── free (PDE)
       │    Q_in(t)             │   evolve by              │   Q(L) 保留
       │                        │   −∂F/∂x+S               │   不强制 P_wk/R_d
  A ───○─── extrapolated        ○─── free                   ○─── derived
       │    A₀=A₁               │                          │   A(L) from P_lumen
       │                        │                          │
  p ───◇─── derived             ◇─── derived                ◇─── drives A(L)
       │    tube law            │   tube law               │   via inv tube law
       │                        │                          │
       │                        │                    P_wk ──●── ODE state
       │                        │                    Q_venous= P_wk/R_d (0D 出流)
       │                        │                          │   C·dP_wk/dt =
       │                        │                          │   Q_drive−Q_venous
       ▼                        ▼                          ▼
   F(U₀) 边界通量          F̂_{j+1/2} HLL 内通量         F(U_{n−1}) 边界通量
```

符号：`●` prescribed，`○` free，`◇` derived。

#### 5.4.3 守恒变量 **U = [A, Q]** 的约束一览

| 位置 | \(A\) | \(Q\) | \(p\) | 决定方式 |
|------|-------|-------|-------|----------|
| **入口** \(j=0\) | 外推 \(A_1\) | **\(Q_{\mathrm{in}}(t)\)** | tube law | `_apply_boundaries` |
| **内部** \(j=1…n{-}2\) | **自由** | **自由** | tube law | SSP-RK2 + HLL + 源项 |
| **出口** \(j=n{-}1\) | **反解** \(A(L)\)（配 \(P_{\mathrm{lumen}}\)） | **自由** \(Q(L)\)（PDE） | **\(P_{\mathrm{lumen}}\)** | Windkessel + tube law |

**内部自由度总数**：\(2(n_x - 2)\)（每个内部点的 \(A_j,\,Q_j\) 各 1 个）。入口主要约束 \(Q\)；出口经 Windkessel 约束 \(P_{\mathrm{lumen}}\) 与 \(A(L)\)，**不**强制 \(Q(L)=P_{\mathrm{wk}}/R_d\)。

#### 5.4.4 出口 Windkessel 与管腔的耦合

```mermaid
flowchart TB
    QLO["Q_lumen = U[1,-1] 出口管腔 PDE"]
    QDRV["Q_drive = Q(L)"]
    ODE["C · dP_wk/dt = Q_drive − P_wk/R_d"]
    PWK["P_wk"]
    PLUM["P_lumen = P_wk + Rp·Q_drive  (3wk)"]
    QVEN["Q_venous = P_wk / R_d  (微循环出流)"]
    AOUT["A(L) = area_from_lumen_pressure(P_lumen)"]

    QLO --> QDRV
    QDRV --> ODE
    ODE --> PWK
    PWK --> PLUM
    PWK --> QVEN
    PLUM --> AOUT
    AOUT --> UOUT["U[-1] = [A(L), Q(L)]"]
    QLO --> UOUT
    UOUT --> QLO
```

\(Q(L)\) 由 1D PDE 与入口 \(Q_{\mathrm{in}}(t)\) 经沿程传播决定，再经 \(Q_{\mathrm{drive}}\) 反馈到 Windkessel；\(P_{\mathrm{lumen}}\) 经 tube law 闭合 \(A(L)\)。**勿将 \(Q_{\mathrm{venous}}=P_{\mathrm{wk}}/R_d\) 写入 \(Q(L)\)**，否则破坏 0D–1D 质量闭合。

#### 5.4.5 一个 SSP-RK2 步内边界何时介入

与 §4.2 一致：Windkessel 状态 \(P_{\mathrm{wk}}\) **仅在完整步末**（`advance_outlet=True`）积分；RK 中间子步只 `apply_substep`，\(P_{\mathrm{wk}}\) 冻结。

```mermaid
sequenceDiagram
    participant S as state U^n
    participant BC as _apply_boundaries
    participant L as _spatial_operator
    participant WK as Windkessel

    Note over S,WK: 子步 1
    S->>BC: BC(state, advance=False)
    Note right of BC: 入口 Q_in<br/>出口只改 A(L)<br/>Q(L) 保留 PDE 值
    BC->>L: u0
    L->>L: k0 = −∂F/∂x + S
    L->>BC: u1 = clip(u0 + dt·k0)
    BC->>BC: BC(u1, advance=False)

    Note over S,WK: 子步 2（步末）
    L->>L: k1 from u1
    L->>BC: state* = 0.5·u0 + 0.5·(u1+dt·k1)
    BC->>WK: advance(dt, Q_lumen)
    Note right of WK: 更新 P_wk；返回 P_lumen
    WK->>BC: P_lumen
    BC->>S: state^{n+1} [A(L) updated, Q(L) unchanged]
```

#### 5.4.6 快速记忆

```text
        prescribed          free                 free + derived
           │                  │                         │
    Q_in ──┤                  │                    Q(L) PDE 演化
           │                  │                         │
           │            A_j, Q_j  evolve                 │
           │                  │                         │
    A←A₁ ──┤                  │         P_lumen ──→ A(L)
           │                  │              P_wk/R_d = Q_venous (0D only)
           └────── 双曲 PDE + 源项 传播 ───────────────┘
                         tube law → p_j  everywhere
```

**一句话**：入口为**定流量泵** \(Q_{\mathrm{in}}(t)\)；出口为 **Windkessel 定压（经 tube law 转为 \(A(L)\)）**，管腔流量 \(Q(L)\) 由 PDE 与入口经沿程传播耦合；微循环侧出流 \(Q_{\mathrm{venous}}=P_{\mathrm{wk}}/R_d\) 仅出现在 0D ODE 中。

### 5.5 初值与模块耦合

`set_lumen_area_profile` 在设置 \(A_0(x)\) 后**同步初始化** PDE 状态与出口 Windkessel，使各模块在 \(t=0\) 自洽：

| 量 | 初值 | 依据 |
|----|------|------|
| \(A(x,0)\) | \(A_0(x)\) | tube law → \(p \approx P_{\mathrm{ref}}\) |
| \(Q(x,0)\) | \(Q_{\mathrm{in}}(0)\) | 与入口定流量 BC 一致（**非** \(Q_{\mathrm{mean}}\)） |
| \(P_{\mathrm{wk}}(0)\) | \(Q_{\mathrm{mean}}\, R_d = P_{\mathrm{ref}}\) | Windkessel 稳态标定 |
| \(P_{\mathrm{lumen}}(0)\)（3wk） | \(P_{\mathrm{wk}} + R_p\, Q_{\mathrm{in}}(0)\) | `outlet.reset(q_drive=…)` |

**参数标定链**（默认自动，勿拆开改）：

```text
CO, coronary_fraction  →  Q_mean
P_ref (tube law)       →  R_d = P_ref / Q_mean
outlet_proximal_fraction  →  R_p = fraction · R_d
```

稳态对齐关系：当 \(A=A_0\)、\(Q=Q_{\mathrm{mean}}\) 时，tube law 给出 \(p=P_{\mathrm{ref}}\)，Windkessel 给出 \(P_{\mathrm{wk}}=P_{\mathrm{ref}}\)。

**`_OutletWindkesselState.reset(p_wk=None, q_drive=None)`**

| 参数 | 默认 | 说明 |
|------|------|------|
| `p_wk` | `Q_mean · R_d` | 顺应性节点压 |
| `q_drive` | `Q_in(0)` | 三元模型计算 \(P_{\mathrm{lumen}}=P_{\mathrm{wk}}+R_p Q_{\mathrm{drive}}\) |

建议 `duration_s` 取 ≥2–3 个心动周期再统计；若需进一步减小启动瞬态，可离线预积分 Windkessel 取周期末 \(P_{\mathrm{wk}}\) 传入 `reset(p_wk=…)`（见 §11）。

---

## 6. 快速开始

### 6.1 运行内置示例

```bash
python navier_stokes_1d.py
```

示例：30 cm 血管、渐细管腔、11–17 cm 狭窄段、模拟 2 s。

### 6.2 最小脚本

```python
import numpy as np
from navier_stokes_1d import NavierStokes1D, BloodFlowParameters

L, nx = 30.0, 151
x = np.linspace(0.0, L, 80)
area = np.pi * (0.40 - 0.08 * x / L) ** 2

par = BloodFlowParameters(
    heart_rate_bpm=75.0,
    coronary_flow_fraction=0.03,
    outlet_windkessel="2wk",
    cfl=0.45,
)

solver = NavierStokes1D(L, nx, par)
solver.set_lumen_area_profile(area, x=x)

A_hist, Q_hist, p_hist, t_hist = solver.run(duration_s=2.4, record_interval_steps=25)
print("FFR ≈", solver.ffr_ratio())
solver.plot_results()
```

### 6.3 推荐工作流

```text
BloodFlowParameters → NavierStokes1D → set_lumen_area_profile → run → 后处理 / plot_results
```

---

## 7. 输入数据：管腔面积剖面

### 7.1 `set_lumen_area_profile`

```python
solver.set_lumen_area_profile(
    area,           # 参考截面积 A₀，单位 cm²
    x=None,         # 采样位置 (cm)；None 则在 [0,L] 上均匀布点
    beta=None,      # β(x)，标量或数组；None 则用参数类默认值
    lesions=None,   # 可选狭窄 [(x0, x1, area_scale, beta_scale), ...]
)
```

| 参数 | 说明 |
|------|------|
| `area` | 至少 2 个点；定义 \(A_0(x)\) |
| `x` | 与 `area` 等长；若未覆盖 0 或 L，自动在端点补常值外推 |
| `beta` | 标量 → 全场常数；数组 → 插值到求解网格 |
| `lesions` | 在插值后的 \(A_0,\beta\) 上，对 \([x_0,x_1]\) 闭区间乘以 `area_scale`、`beta_scale` |

调用后会：

- 将 `state[0] = A₀(x)`，`state[1] = Q_in(0)`（与入口 BC 一致，非静止初值）
- 由 tube law 更新 `pressure`（\(A=A_0\) 时 \(p \approx P_{\mathrm{ref}}\)）
- 调用 `outlet.reset(p_wk=Q_mean·R_d, q_drive=Q_in(0))` 重置 Windkessel；三元模型含 \(R_p Q_{\mathrm{in}}(0)\) 项

### 7.2 由半径构造面积

```python
radius_cm = ...  # 沿程半径
area = np.pi * radius_cm**2
solver.set_lumen_area_profile(area, x=x_cm)
```

### 7.3 狭窄示例

```python
# 在 12–18 cm 处：参考面积 ×0.65，管壁刚度 ×4.5
lesions=[(12.0, 18.0, 0.65, 4.5)]
```

面积狭窄率（相对健康段 \(A_{0,\mathrm{ref}}\)）：

\[
\text{stenosis\%} \approx \left(1 - \frac{A_{0,\min}}{A_{0,\mathrm{ref}}}\right)\times 100\%
\]

### 7.4 参数数量级参考

| 参数 | 典型量级 |
|------|----------|
| \(A_0\) | 0.05–1.0 cm² |
| \(\beta\) | 500–1500 mmHg/√cm |
| \(L\) | 10–50 cm |
| \(Q_{\mathrm{mean}}\) | 约 2.5–4 mL/s（由 CO 与冠脉比例决定） |

---

## 8. 参数类 `BloodFlowParameters`

`@dataclass`，构造时可覆盖任意字段。

### 8.1 流体与数值

| 字段 | 默认 | 说明 |
|------|------|------|
| `rho` | 1.05 | 密度 (g/cm³) |
| `alpha` | 1.1 | 动量修正系数 |
| `mu` | 0.0035 | 运动粘度 (cm²/s) |
| `cfl` | 0.5 | CFL 数 |
| `dt_max_s` | 5e-4 | 时间步上限 (s) |

### 8.2 管壁

| 字段 | 默认 | 说明 |
|------|------|------|
| `p_ref_mmhg` | 80 | tube law 参考压 (mmHg) |
| `beta_mmhg_per_sqrt_cm` | 750 | 默认 \(\beta\) |

### 8.3 入口（→ `coronary_inlet_flow`）

| 字段 | 默认 | 说明 |
|------|------|------|
| `heart_rate_bpm` | 75 | 心率 |
| `cardiac_output_l_per_min` | 5.0 | 心输出量 (L/min) |
| `coronary_flow_fraction` | 0.03 | 冠脉占 CO 比例 |
| `inlet_min_flow_fraction` | 0.1 | 流量下限 = 该比例 × \(Q_{\mathrm{mean}}\) |
| `inlet_phase_offset_rad` | 0 | 波形相位偏移 |
| `inlet_fourier_coefficients` | None | 自定义谐波；None 用模块内置默认系数 |

方法：

- `mean_coronary_flow_ml_s()` → \(Q_{\mathrm{mean}}\)
- `inlet_flow(t)` → 入口流量数组
- `resolve_outlet_resistances()` → \((R_d, R_p)\)

### 8.4 出口 Windkessel

| 字段 | 默认 | 说明 |
|------|------|------|
| `outlet_windkessel` | `"2wk"` | `"2wk"` 或 `"3wk"` |
| `outlet_r_distal_mmhg_s_per_ml` | None | \(R_d\)；None 则 \(P_{\mathrm{ref}}/Q_{\mathrm{mean}}\) |
| `outlet_r_proximal_mmhg_s_per_ml` | None | \(R_p\)；None 则 `0.12 * R_d` |
| `outlet_compliance_ml_per_mmhg` | 0.08 | 顺应性 \(C\) |
| `outlet_proximal_fraction` | 0.12 | 三元模型近端阻力比例 |

Windkessel ODE 驱动流量固定为出口管腔流量 \(Q(L)\)，无额外配置项。

---

## 9. 求解器 `NavierStokes1D`

### 9.1 构造

```python
NavierStokes1D(
    vessel_length_cm: float,
    n_nodes: int,
    parameters: BloodFlowParameters | None = None,
)
```

| 属性 | 说明 |
|------|------|
| `length`, `nx`, `dx` | 长度、节点数、间距 |
| `x` | 节点坐标 (cm)，shape `(nx,)` |
| `area_ref`, `beta` | 参考几何，shape `(nx,)` |
| `state` | 守恒变量 `[A, Q]`，shape `(2, nx)` |
| `pressure` | 沿程管腔标量压 \(p(x)\) (mmHg)，shape `(nx,)`；由 tube law 从 \(A\) 派生 |
| `time` | 当前物理时间 (s) |

### 9.2 `run(duration_s, record_interval_steps=20)`

返回四元组：

| 输出 | shape | 说明 |
|------|--------|------|
| `history_area` | `(n_save, nx)` | \(A\) 历史 |
| `history_flow` | `(n_save, nx)` | \(Q\) 历史 |
| `history_pressure` | `(n_save, nx)` | 管腔标量压 \(p\) 历史 (mmHg) |
| `history_time` | `(n_save,)` | 保存时刻 (s) |

同时写入对象属性 `solver.history_*`。保存帧数约为 `1 + floor(n_steps / record_interval_steps)`。

### 9.3 `plot_results()`

2×2 子图：\(A(t,x)\) 云图、\(\langle p(x)\rangle_t\)、入口 \(Q,p\) 曲线、末时刻沿程 \(A,p,Q\)。需先 `run()`。

### 9.4 模块级工具函数

| 函数 | 说明 |
|------|------|
| `lumen_pressure_mmhg(A, A0, beta, p_ref)` | tube law：由 \(A\) 算管腔标量压 |
| `area_from_lumen_pressure_mmhg(p, A0, beta, p_ref)` | tube law 反解 \(A\) |

---

## 10. 后处理与 FFR

### 10.1 读取场数据

```python
A, Q, p, t = solver.run(2.0)

# 末时刻沿程压力
import matplotlib.pyplot as plt
plt.plot(solver.x, p[-1])
plt.ylabel("p (mmHg)")
plt.show()

# 入口流量时间序列
plt.plot(t, Q[:, 0])
plt.show()
```

### 10.2 导出 NumPy

```python
np.savez(
    "coronary_1d_result.npz",
    x=solver.x,
    times=solver.history_time,
    A=solver.history_area,
    Q=solver.history_flow,
    p=solver.history_pressure,
    area_ref=solver.area_ref,
    beta=solver.beta,
)
```

### 10.3 `ffr_ratio()`

当前实现为**简化指标**：

\[
\text{ratio} = \frac{\langle p \rangle_{x,\,t}}{p_{\mathrm{mean}}(x=0)}
\]

其中 \(\langle p \rangle_{x,\,t}\) 为各网格点在**全部保存时刻**上的时间平均管腔标量压，再与入口网格平均压之比（见源码：先 `mean(history_pressure, axis=0)`，再除以 `p_mean[0]`）。比较的是**沿 \(x\) 不同位置**的 \(p\)，非某一方向的应力分量。

> 与临床导管 FFR（远端/主动脉压比）定义不同，仅用于模型内相对比较；正式分析建议自定义近端/远端分区或取狭窄远心端网格。

---

## 11. 与离线 Windkessel 对比

`NavierStokes1D.reference_outlet_pressure(times, parameters, q_override=None)` 调用 `coronary_outlet` 的 `windkessel2_outlet_pressure` / `windkessel3_outlet_pressure`，在**给定流量序列**下离线积分得到参考 \(P_{\mathrm{out}}(t)\)，**不参与**耦合求解。

离线模型以 **`Q_in(t)`** 驱动 Windkessel（无 1D 管腔、无沿程延迟）；耦合求解以 **`Q(L)`** 驱动。二者驱动方式不同，耦合结果含沿程传播，与离线曲线会有相位/幅值差异，属预期行为。

```python
t = np.linspace(0, 2.0, 2000)
p_ref = NavierStokes1D.reference_outlet_pressure(t, par)
# 三元模型返回 (p_out, p_wk)
```

用途：检查 Windkessel 参数；与 `solver.history_pressure[:, -1]` 对比。耦合良好时相关系数通常 \> 0.99。

**质量守恒诊断**（`_demo` 中已有）：

```python
q_in  = np.mean(solver.history_flow[:, 0])
q_out = np.mean(solver.history_flow[:, -1])
print(f"Q_in={q_in:.3f}, Q_out={q_out:.3f}")  # 时间平均应接近
# 修复后 |Q(L)−Q[-2]| 应很小；勿将 Q(L) 与 outlet.p_wk/outlet.r_d 逐点对比
```

---

## 12. 参数调优建议

| 目标 | 建议 |
|------|------|
| 更高空间精度 | 增大 `n_nodes`；略减小 `cfl`（如 0.35–0.45） |
| 加快计算 | 减小 `n_nodes` 或 `duration_s`；增大 `record_interval_steps` |
| 不稳定 / 振荡 | 减小 `cfl`；检查 \(\beta\) 是否过大；确认单位一致 |
| 周期性稳态 | `duration_s` 取 2–4 个心动周期（HR=75 → \(T\approx 0.8\) s，建议 ≥1.6 s）；初值已用 \(Q_{\mathrm{in}}(0)\) 与稳态 \(P_{\mathrm{wk}}\)，启动瞬态较旧版 \(Q=0\) 更短 |
| 更重狭窄 | 适当增大狭窄段 `beta_scale`；必要时加密网格 |

计算量：每步 2 次空间算子（SSP-RK2），复杂度 \(O(n_x)\)。`nx=121`, `T=2` s 时通常为 \(10^4\) 量级步数，普通 PC 上数十秒量级。

---

## 13. 限制与扩展方向

1. **单支、单出口** 1D 模型，无分叉与侧支流量分配（`coronary_outlet.coronary_outlet_flow_distribution` 未接入）。
2. 出口为 **Windkessel–tube law 耦合**，非特征非反射边界；与纯 prescribed \(P(t)\) 或纯 \(Q\) 外推不同。
3. 摩擦为线性 Poiseuille，未含湍流、弯曲损失等。
4. **FFR** 为简化压比，非临床标准。
5. 初值为 \(A=A_0,\; Q=Q_{\mathrm{in}}(0)\)，\(P_{\mathrm{wk}}=P_{\mathrm{ref}}\)；脉动工况下仍建议 ≥2 个心动周期再取统计量。

扩展思路：子类化 `NavierStokes1D` 并重写 `_apply_boundaries`；将影像中心线半径 CSV 转为 `set_lumen_area_profile` 输入；多支血管可在外层循环或多段拼接。

---

## 14. 常见问题

**Q：`demo/navier_stokes.py` 与本模块有何区别？**  
A：`demo/` 为旧原型（不同边界类与 import 路径）。新项目请只用根目录 `navier_stokes_1d.py` + `coronary_inlet/outlet`。

**Q：入口 mL/s 与求解器流量单位？**  
A：1 mL = 1 cm³，数值可直接作为 cm³/s 使用。

**Q：`run()` 后 `ffr_ratio()` 大于 1？**  
A：可能尚未达到周期稳态，或指标定义与临床 FFR 不同；延长 `duration_s` 或自定义远端/近端分区。

**Q：如何改入口波形？**  
A：修改 `BloodFlowParameters` 中心输出量、冠脉比例、傅里叶系数等；细节见 `coronary_inlet_说明.md`。

**Q：如何改出口阻力/顺应性？**  
A：设置 `outlet_r_distal_mmhg_s_per_ml`、`outlet_compliance_ml_per_mmhg` 等；与 `coronary_outlet` 中参数含义相同。

**Q：无 GUI 如何出图？**  
A：设置 `MPLBACKEND=Agg`，在 `plot_results()` 后增加 `fig.savefig("result.png")`（需自行获取 `fig` 引用或改写绘图函数）。

**Q：`set_lumen_area_profile` 与 `lesions` 顺序？**  
A：先插值 `area`/`beta` 到网格，再对 `lesions` 区间做乘法修正。

**Q：为何初值用 \(Q_{\mathrm{in}}(0)\) 而非 \(Q_{\mathrm{mean}}\)？**  
A：入口 BC 为 \(Q(0,t)=Q_{\mathrm{in}}(t)\)；\(t=0\) 时若管内初值与 BC 不一致会产生启动跳变。\(P_{\mathrm{wk}}\) 仍用 \(Q_{\mathrm{mean}} R_d\) 稳态标定，与 tube law 的 \(P_{\mathrm{ref}}\) 对齐。

**Q：`pressure` 是沿 x 轴方向的压力吗？**  
A：否。\(p\) 是各向同性**标量**管腔压（mmHg），`pressure[i]` 只是位置 \(x_i\) 处的取值。tube law 将其与 \(A\) 耦合（径向胀缩）；动量方程中 \(\partial(pA)/\partial x\) 体现沿 \(x\) 的压力梯度对轴向流量 \(Q\) 的驱动。详见 [§3.2.1](#321-管腔压-p-的物理含义)。

**Q：出口 `Q(L)` 为何不等于 `P_wk/R_d`？**  
A：\(P_{\mathrm{wk}}/R_d\) 是 Windkessel **微循环侧出流** \(Q_{\mathrm{venous}}\)，不是管腔出口流量。耦合求解中 \(Q(L)\) 由 1D PDE 保留；仅当 \(dP_{\mathrm{wk}}/dt=0\) 时两者接近。详见 [§5.2](#52-出口xlwindkessel-耦合)。

---

## 15. 文件关系速查

```text
FFR_1D/
├── navier_stokes_1d.py      # 本求解器 ★
├── navier_stokes_1d.md      # 本文档
├── coronary_constants.py    # 共享公开全局常量
├── coronary_inlet.py        # 入口 Q(t)
├── coronary_inlet_说明.md
├── coronary_outlet.py       # Windkessel 出口模型与离线压力
├── coronary_outlet.md
└── demo/                    # 早期原型（勿与根目录求解器混用）
    └── navier_stokes.py
```

**数据流示意：**

```mermaid
flowchart LR
  subgraph input [输入]
    A0["A₀(x) 管腔面积"]
    PAR["BloodFlowParameters"]
  end
  subgraph solver [navier_stokes_1d]
    FV["FVM + MUSCL + HLL"]
    SSP["SSP-RK2"]
  end
  subgraph bc [边界]
    IN["coronary_inlet_flow"]
    WK["Windkessel ODE"]
  end
  A0 --> FV
  PAR --> FV
  PAR --> IN
  PAR --> WK
  IN --> FV
  WK --> FV
  FV --> SSP
  SSP --> OUT["A, Q, p 历史"]
```

---

*文档版本与 `navier_stokes_1d.py` 同步；若 API 变更，以源码 docstring 为准。*
