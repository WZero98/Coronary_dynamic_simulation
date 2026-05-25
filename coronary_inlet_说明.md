# 冠状动脉入口流量模块说明（`coronary_inlet.py`）

本文档说明 `coronary_inlet.py` 中主要函数、参数含义及调节方法，重点介绍**生理模板**如何塑造冠脉入口波形，以及模板参数与傅里叶拟合、最终流量输出之间的关系。

---

## 1. 模块做什么

本模块用两步生成冠脉入口脉动血流量 \(Q(t)\)（单位：**mL/s**）：

1. **生理模板** `coronary_flow_template`：在归一化心动相位 \(\tau \in [0,1)\) 上构造单周期参考波形（均值约为 1）。
2. **傅里叶拟合** `fit_fourier_coefficients`：将模板分解为有限项正弦谐波，得到系数 \((n, \text{amp\_ratio}, \text{phase})\)。
3. **入口流量** `coronary_inlet_flow`：按心率、心输出量等将归一化波形放大为物理流量，并施加非负约束。

在 1D 求解器（`navier_stokes.py`）中，通常通过 `inlet_boundary_condition.py` 导入 `coronary_inlet_flow` 作为入口边界。CGS 单位下 **1 mL = 1 cm³**，数值可与 cm³/s 直接对接。

### 1.1 数学形式

心动周期 \(T = 60/\text{HR}\)（秒），\(\omega = 2\pi/T\)，\(t_{\text{mod}} = t \bmod T\)：

\[
Q(t) = Q_{\text{mean}} + \sum_{n=1}^{N} Q_{\text{mean}} \cdot \text{amp\_ratio}_n \cdot \sin\bigl(n\omega t_{\text{mod}} + \phi_n + \phi_{\text{offset}}\bigr)
\]

其中：

\[
Q_{\text{mean}} = \frac{\text{cardiac\_output} \times 1000 \times \text{coronary\_fraction}}{60}
\]

模板拟合在归一化相位 \(\tau = t_{\text{mod}}/T\) 上进行；谐波形式为 \(\sin(2\pi n \tau + \phi_n)\)，与上式通过 \(\tau\) 与 \(t_{\text{mod}}\) 的对应关系一致。

### 1.2 生理背景（为何这样建模）

| 心动相 | 冠脉特点 | 模板中的体现 |
|--------|----------|----------------|
| 收缩期 | 心肌挤压，壁内血管可压闭，流量偏低 | 早期收缩峰低于舒张峰；晚期不跌入波谷 |
| 收缩末期 | 尚未到最低点即进入舒张灌注 | `junction_flow_frac` 控制的融合升支 |
| 舒张期 | 主动脉舒张压驱动，灌注占主导 | 舒张段升至主峰后回落 |

设计目标（可通过 `template_peak_metrics` / `coronary_inlet_flow_stats` 校验）：

- **脉动指数 PI** = \((Q_{\max}-Q_{\min})/Q_{\text{mean}}\) 约在 **0.9–1.3**
- **早期收缩峰 / 舒张峰** ≈ **70%**（由 `systolic_to_diastolic_peak_ratio` 设定）
- 收缩期末流量高于收缩晚期局部低点（融合，非“先探底再跳升”）

---

## 2. 单周期模板波形结构

归一化时间 \(\tau=0\) 为周期起点，\(\tau_{\text{sys}}=\) `systolic_duration_frac` 为收缩期结束时刻。

```
流量
  ^
  |     舒张峰 (diastolic_peak)
  |        /\
  |       /  \________ 回落 → baseline
  |      /    \
  |  融合升支 /      \
  |    ______/        \
  |   /  早期收缩峰     \
  |  /  (ratio×舒张峰)   \
  +--+---+---+---+---+---→ τ
     0  t_sp      τ_sys    1
        ↑早期峰    ↑收缩末
```

**三段构造（归一化幅值，拟合前）：**

| 区段 | \(\tau\) 范围 | 形状 |
|------|----------------|------|
| 收缩早期 | \([0,\ t_{sp}\cdot\tau_{sys})\) | 由 `baseline` 正弦升至 `s_peak` |
| 收缩晚期 | \([t_{sp}\cdot\tau_{sys},\ \tau_{sys})\) | 余弦平滑升支：\(s_{\text{peak}} \to q_{\text{join}}\) |
| 舒张期 | \([\tau_{sys},\ 1)\) | 先升至 `d_peak`，再回落至 `baseline` |

最后整段除以均值，使 \(\bar q = 1\)，供傅里叶拟合。

---

## 3. 主要函数一览

| 函数 | 作用 |
|------|------|
| `coronary_flow_template` | 生成单周期归一化生理模板 |
| `template_peak_metrics` | 计算模板峰值比、融合点等诊断指标 |
| `fit_fourier_coefficients` | 由模板拟合傅里叶谐波系数 |
| `coronary_inlet_flow` | **主接口**：给定时间 \(t\) 返回流量 (mL/s) |
| `diastolic_flow_fraction` | 单周期舒张期灌注量占比 |
| `coronary_inlet_flow_stats` | 均值、峰谷、PI、舒张占比等统计 |
| `plot_coronary_flow` | 可视化模板与傅里叶重建 |

内部函数：`_get_fourier_coefficients`、`_build_default_fourier_coefficients`；模块加载时生成 `_DEFAULT_FOURIER_COEFFICIENTS`。

---

## 4. `coronary_flow_template` — 生理模板（核心）

```python
coronary_flow_template(
    tau,
    systolic_duration_frac=0.33,
    diastolic_peak=1.30,
    systolic_to_diastolic_peak_ratio=0.50,  # 代码当前默认值，见下表
    baseline_level=0.20,
    systolic_early_peak_frac=0.18,
    junction_flow_frac=0.84,
    diastolic_peak_frac=0.28,
    systolic_level=None,
)
```

**输入**

| 参数 | 类型 | 说明 |
|------|------|------|
| `tau` | float 或 array | 归一化相位 \([0,1)\)，可大于 1（自动取模） |

**输出**

| 返回值 | 说明 |
|--------|------|
| 与 `tau` 同形的数组 | 归一化流量，**周期均值 ≈ 1** |

### 4.1 模板参数详解（调节重点）

以下参数在**归一化、除以均值之前**作用于幅值；改变后应重新调用 `fit_fourier_coefficients` 或重启 Python 以刷新 `_DEFAULT_FOURIER_COEFFICIENTS`（若依赖默认系数）。

#### `systolic_duration_frac`（默认 `0.33`）

- **含义**：收缩期占整个心动周期的比例 \(\tau_{\text{sys}}\)。
- **生理**：静息心率下收缩期约 0.33–0.37；心率升高时常缩短。
- **调节**：
  - **增大** → 收缩段变长，舒张段变短；舒张期灌注占比通常**下降**。
  - **减小** → 舒张段变长，利于提高舒张灌注占比。
- **注意**：`diastolic_flow_fraction` 默认用同一阈值划分收缩/舒张，应与模板一致。

#### `diastolic_peak`（默认 `1.30`）

- **含义**：舒张期目标峰值幅值（归一化前），近似“舒张早期最高灌注”的相对高度。
- **调节**：
  - **增大** → 舒张峰更高，周期幅值差变大，**PI 倾向升高**（经傅里叶重建后仍受此趋势影响）。
  - **减小** → 波形变平，PI 倾向降低。
- **与其他参数关系**：`baseline`、`s_peak`、`q_join` 均常以 `diastolic_peak` 为尺度（见下）。

#### `systolic_to_diastolic_peak_ratio`（默认 `0.50`，设计目标常为 `0.70`）

- **含义**：早期收缩峰 \(s_{\text{peak}} = \text{ratio} \times \text{diastolic\_peak}\)（当 `systolic_level is None`）。
- **生理**：收缩期灌注峰值一般明显低于舒张早期峰值；临床讨论中常用“收缩峰约为舒张峰的 70%”。
- **调节**：
  - **增大**（如 0.5 → 0.7）→ 早期收缩峰抬高，峰比更接近 70%。
  - **减小** → 收缩期更“扁”，与舒张峰对比更弱。
- **验证**：用 `template_peak_metrics()['peak_ratio']`（**早期收缩峰 / 舒张峰**，非整个收缩期最大值）。

#### `baseline_level`（默认 `0.20`）

- **含义**：周期最低流量水平，`baseline = baseline_level × diastolic_peak`。
- **生理**：舒张末期 / 收缩早期的基础灌注底线。
- **调节**：
  - **增大** → 全波形抬高，\(Q_{\min}\) 升高，**PI 通常下降**。
  - **减小** → 谷更深，**PI 通常升高**。
- **建议**：调 PI 时优先与 `diastolic_peak`、`junction_flow_frac` 联动小步调。

#### `systolic_early_peak_frac`（默认 `0.18`）

- **含义**：早期收缩峰在**收缩期内**的相对位置，\(t_{sp} \in (0.05, 0.45)\)（代码内裁剪）。
  - 绝对时刻 \(\approx t_{sp} \times \tau_{\text{sys}}\)（如 0.18×0.33 ≈ 周期的 6% 处）。
- **生理**：心肌挤压建立后，早期仍可有一小段流量，随后进入融合升支。
- **调节**：
  - **减小** → 峰更早出现，收缩晚期融合段更长。
  - **增大** → 峰后移，融合段更短，形态更“陡”。

#### `junction_flow_frac`（默认 `0.84`）

- **含义**：收缩期末融合流量 \(q_{\text{join}} = \max(\text{junction\_flow\_frac} \times d_{\text{peak}},\ s_{\text{peak}}+\varepsilon)\)。
- **生理**：实现“**尚未跌至波谷即与舒张升支衔接**”；期末流量应高于收缩晚期局部低点。
- **调节**：
  - **增大**（如 0.84 → 0.90）→ 收缩末更高，与舒张衔接更平滑，但早期峰/舒张峰比会变小（若用全收缩期 max 会误判；应用 `template_peak_metrics`）。
  - **减小** → 融合点降低；过低可能导致收缩晚期出现相对凹陷。
- **融合判据**：`flow_at_systole_end > pre_systole_end_local_min`。

#### `diastolic_peak_frac`（默认 `0.28`）

- **含义**：舒张期内主峰出现的归一化位置 \(t_{\text{dia,peak}} \in [0.08, 0.55]\)（相对舒张段长度）。
  - 越小 → 峰越早（舒张刚开局即达峰）。
  - 越大 → 峰后移，升支更长。
- **调节**：影响舒张早期形状及傅里叶高次谐波含量；对 **PI** 有次要影响，对 **舒张灌注时间分布** 影响更明显。

#### `systolic_level`（默认 `None`）

- **含义**：若给定浮点数，则**直接指定**早期收缩峰幅值，覆盖 `systolic_to_diastolic_peak_ratio` 的计算。
- **用途**：在固定 `diastolic_peak` 时做精细标定，或对接外部实测峰值。

### 4.2 模板参数调节流程（推荐）

1. **固定心率与均值相关参数**（在 `coronary_inlet_flow` 层）：`heart_rate`、`cardiac_output`、`coronary_fraction`。
2. **先定形态**：`systolic_to_diastolic_peak_ratio` → 0.7，`junction_flow_frac` → 0.82–0.88，确认 `template_peak_metrics` 融合成立。
3. **再定脉动**：用 `coronary_inlet_flow_stats` 看 **PI**，通过 `baseline_level`、`diastolic_peak` 微调至 0.9–1.3。
4. **重拟合谐波**：
   ```python
   from coronary_inlet import fit_fourier_coefficients, coronary_inlet_flow

   coeffs = tuple(fit_fourier_coefficients(
       n_harmonics=7,
       systolic_to_diastolic_peak_ratio=0.70,
       baseline_level=0.20,
       # ... 与模板一致的 kwargs
   ))
   Q = coronary_inlet_flow(t, fourier_coefficients=coeffs)
   ```
5. **避免只改模板不改系数**：默认 `coronary_inlet_flow()` 使用模块加载时的 `_DEFAULT_FOURIER_COEFFICIENTS`；模板参数变更后需显式传入新 `fourier_coefficients` 或重新导入模块。

### 4.3 参数耦合简表

| 目标 | 建议调节 |
|------|----------|
| 提高 PI | ↓ `baseline_level` 或 ↑ `diastolic_peak` |
| 降低 PI | ↑ `baseline_level` 或 ↓ `diastolic_peak` |
| 峰比 → 70% | ↑ `systolic_to_diastolic_peak_ratio`（或 `systolic_level`） |
| 强化收缩末融合 | ↑ `junction_flow_frac`，保证 `flow_at_systole_end > pre_systole_end_local_min` |
| 提高舒张灌注占比 | ↓ `systolic_duration_frac`，或略降低收缩段幅值 |
| 更尖的舒张峰 | ↓ `diastolic_peak_frac` |
| 更平滑的傅里叶逼近 | ↑ `n_harmonics`（在 `fit_fourier_coefficients` 中） |

---

## 5. `template_peak_metrics`

```python
template_peak_metrics(tau=None, systolic_duration_frac=0.33, **kwargs)
```

**作用**：在归一化模板上计算诊断量，**不经过傅里叶**，用于标定模板形态。

| 返回键 | 含义 |
|--------|------|
| `systolic_early_peak` | 收缩早期（\(\tau < t_{sp}\cdot\tau_{sys}\)）段最大值 |
| `diastolic_peak` | 舒张段（\(\tau \ge \tau_{sys}\)）最大值 |
| `peak_ratio` | `systolic_early_peak / diastolic_peak`（目标 ≈ 0.7） |
| `flow_at_systole_end` | \(\tau=\tau_{\text{sys}}^{-}\) 处流量（融合点） |
| `pre_systole_end_local_min` | 收缩晚期（约末 45% 收缩段）局部最小流量 |

**为何不用 `max(整个收缩期)`？** 收缩末融合点往往高于早期收缩峰，用全收缩期 max 会高估“收缩峰”，峰比失真。

---

## 6. `fit_fourier_coefficients`

```python
fit_fourier_coefficients(
    n_harmonics=7,
    n_samples=2048,
    **template_kwargs,  # 原样传给 coronary_flow_template
)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `n_harmonics` | 7 | 保留的谐波阶数 \(n=1,\ldots,N\)；增大可更贴模板，但计算略增 |
| `n_samples` | 2048 | 单周期采样点数；越高拟合越稳 |
| `**template_kwargs` | — | 与 `coronary_flow_template` 参数同名，**必须一致** |

**返回**：`list[(n, amp_ratio, phase_rad)]`

- `n`：谐波阶次（基频 \(n=1\) 对应心率频率）
- `amp_ratio`：该谐波振幅相对 **交流分量** 的比例（拟合时对模板去均值）
- `phase_rad`：谐波相位（弧度），对应 \(\sin(2\pi n\tau + \phi)\)

拟合方法：对 \(y(\tau)-1\) 做各阶正弦/余弦投影，\(\text{amp}=\sqrt{a_n^2+b_n^2}\)，\(\phi=\mathrm{atan2}(b_n,a_n)\)。

---

## 7. `coronary_inlet_flow`（求解器入口）

```python
coronary_inlet_flow(
    t,
    heart_rate=75.0,
    cardiac_output=5.0,
    coronary_fraction=0.045,
    fourier_coefficients=None,
    min_flow_fraction=0.1,
    phase_offset=0.0,
)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `t` | — | 时间 (s)，标量或数组 |
| `heart_rate` | 75 | 心率 (bpm)，\(T=60/\text{HR}\) |
| `cardiac_output` | 5.0 | 心输出量 (L/min) |
| `coronary_fraction` | 0.045 | 冠脉流量占心输出量比例（约 4–5%） |
| `fourier_coefficients` | `None` → 内置默认 | 自定义谐波列表；改模板后应传入新系数 |
| `min_flow_fraction` | 0.1 | 流量下限 \(= 0.1 \times Q_{\text{mean}}\)，避免负流量 |
| `phase_offset` | 0.0 | 整体相位 (rad)，用于与 ECG/主动脉压波形对齐 |

**输出**：与 `t` 同形的 **mL/s**，且 \(\ge \text{min\_flow\_fraction} \times Q_{\text{mean}}\)。

**与模板的关系**：模板只决定谐波形状；**绝对流量尺度**由 `cardiac_output` 与 `coronary_fraction` 决定，与模板归一化无关。

---

## 8. `diastolic_flow_fraction`

```python
diastolic_flow_fraction(
    t, heart_rate=75.0, cardiac_output=5.0,
    coronary_fraction=0.045,
    systolic_duration_frac=0.33,
    **kwargs,  # 可传 fourier_coefficients 等
)
```

- **含义**：一个周期内，\(\tau \ge \tau_{\text{sys}}\) 段流量积分占全周期积分比例。
- **注意**：基于**傅里叶重建**的 \(Q(t)\)，不是模板直接积分；划分收缩/舒张的阈值是 `systolic_duration_frac`，应与模板一致。
- **生理参考**：文献常述舒张期灌注约占 80–90%；当前模板偏“形态标定”，该比例可能低于 80%，若需提高应缩短 `systolic_duration_frac` 并配合统计复验。

---

## 9. `coronary_inlet_flow_stats`

```python
coronary_inlet_flow_stats(
    heart_rate=75.0,
    cardiac_output=5.0,
    coronary_fraction=0.045,
    duration_cycles=3.0,
    **kwargs,
)
```

| 返回键 | 含义 |
|--------|------|
| `Q_mean_mL_s` | 时间平均流量 |
| `Q_max_mL_s` / `Q_min_mL_s` | 峰、谷 |
| `pulsatility_index` | \((Q_{\max}-Q_{\min})/Q_{\text{mean}}\) |
| `diastolic_flow_fraction` | 舒张期灌注占比 |

默认模拟 `duration_cycles` 个心动周期（采样 2000×周期数点）。**PI 标定应以本函数的输出为准**（傅里叶重建后），不要仅用模板直观判断。

---

## 10. `plot_coronary_flow`

用于快速查看：

- 多周期 `coronary_inlet_flow` 曲线；
- 单周期：傅里叶重建 vs 生理模板（模板乘以 \(Q_{\text{mean}}\) 对比）。

默认 `coronary_fraction=0.025` 与主接口不同，仅影响绘图幅值，不影响默认系数。

运行：

```bash
python coronary_inlet.py
```

---

## 11. 与项目其他文件的衔接

| 文件 | 关系 |
|------|------|
| `inlet_boundary_condition.py` | `from coronary_inlet import coronary_inlet_flow`，兼容旧导入路径 |
| `navier_stokes.py` | 边界条件中调用 `coronary_inlet_flow(t, heart_rate=...)` |
| `navier_stokes_使用说明.md` | 求解器总说明；入口流量细节以本文档为准 |

修改模板或系数后，若使用默认谐波，请重新加载模块或显式传入 `fourier_coefficients`。

---

## 12. 常用代码示例

### 12.1 检查模板是否满足峰比与融合

```python
from coronary_inlet import template_peak_metrics, coronary_inlet_flow_stats

m = template_peak_metrics()
print(f"峰比={m['peak_ratio']:.2%}, 融合={m['flow_at_systole_end'] > m['pre_systole_end_local_min']}")

s = coronary_inlet_flow_stats()
print(f"PI={s['pulsatility_index']:.2f}, Q_mean={s['Q_mean_mL_s']:.2f} mL/s")
```

### 12.2 自定义模板并用于求解器

```python
import numpy as np
from coronary_inlet import fit_fourier_coefficients, coronary_inlet_flow

template_kw = dict(
    systolic_to_diastolic_peak_ratio=0.70,
    junction_flow_frac=0.84,
    baseline_level=0.20,
    diastolic_peak=1.30,
)
coeffs = tuple(fit_fourier_coefficients(n_harmonics=7, **template_kw))

t = np.linspace(0, 2, 500)
Q = coronary_inlet_flow(t, heart_rate=75, fourier_coefficients=coeffs, **{})
```

### 12.3 在 `navier_stokes` 中改入口（概念）

在 `apply_boundary_conditions` 或实例属性中调整 `heart_rate`、`cardiac_output`、`coronary_fraction`；若改模板形态，需在调用处传入与模板一致的 `fourier_coefficients`（或扩展求解器封装该参数）。

---

## 13. 默认常量速查（`coronary_inlet.py` 当前版本）

| 常量 | 值 | 对应模板参数 |
|------|-----|----------------|
| `_DEFAULT_N_HARMONICS` | 7 | `fit_fourier_coefficients` |
| `_DEFAULT_SYSTOLIC_DURATION_FRAC` | 0.33 | `systolic_duration_frac` |
| `_DEFAULT_DIASTOLIC_PEAK` | 1.30 | `diastolic_peak` |
| `_DEFAULT_SYSTOLIC_TO_DIASTOLIC_PEAK_RATIO` | 0.50 | `systolic_to_diastolic_peak_ratio` |
| `_DEFAULT_BASELINE_LEVEL` | 0.20 | `baseline_level` |
| `_DEFAULT_SYSTOLIC_EARLY_PEAK_FRAC` | 0.18 | `systolic_early_peak_frac` |
| `_DEFAULT_JUNCTION_FLOW_FRAC` | 0.84 | `junction_flow_frac` |
| `_DEFAULT_DIASTOLIC_PEAK_FRAC` | 0.28 | `diastolic_peak_frac` |

若需早期收缩峰/舒张峰 ≈ **70%**，请将 `systolic_to_diastolic_peak_ratio` 设为 **0.70** 并重新拟合傅里叶系数；文档字符串中的“70%”描述的是该参数的生理含义，与代码默认值可能不同步，以你项目中标定后的常量为准。

---

## 14. 参考文献（模块 docstring 延续）

- Womersley J.R. — 动脉血流理论  
- Nichols W.W., O'Rourke M.F. — *McDonald's Blood Flow in Arteries*  
- Kim H.J. et al. — 患者特异性冠脉血流建模  

---

*文档版本：与 `coronary_inlet.py` 分段模板 + 傅里叶拟合实现对应。若代码默认常量更新，请同步修改第 13 节表格。*
