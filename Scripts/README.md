# Scripts — 算例与可视化脚本

本目录存放**脚本层**内容：算例编排、交互式可视化、solver 本地缓存。
算法模块全部位于 `../SimuCoronary/`，其规范说明见
[`docs/navier_stokes_1d.md`](../docs/navier_stokes_1d.md)。

## 文件一览

| 文件 | 作用 |
|------|------|
| `experiment.py` | 影像数据完整算例：读 `data/stenoses_data` → 重采样 → 标定 → 绘图 → 质量校验 |
| `simudata_experiment.py` | 模拟数据批量仿真：7 种参数化几何 |
| `simudata_experiment_different_scale.py` | 面积缩放敏感性：7 种几何 × 5 个 `AREA_SCALER` |
| `simudata_experiment_different_stenosis_rate.py` | 狭窄率敏感性：4 种含狭窄几何 × 5 个 `STENOSIS_RATE` |

## 四个脚本速查

| 脚本 | 数据来源 | 算例数 | 输出目录 | 每例缓存 | 实测耗时 |
|------|----------|--------|----------|----------|----------|
| `experiment.py` | 影像 `areas.joblib` | 1 | `data/cache/` | ~18 MB | 首次 9 min / 命中 1.6 s |
| `simudata_experiment.py` | 参数化几何 | 7 | `results/模拟数据结果/` | 53–77 MB | ~24 min（4 进程） |
| `..._different_scale.py` | 参数化几何 | 5 × 7 = 35 | `results/模拟数据实验2（面积大小变化）/` | 53–77 MB | 96 min（4 进程） |
| `..._different_stenosis_rate.py` | 参数化几何 | 5 × 4 = 20 | `results/模拟数据实验3（狭窄率变化）/` | 53–77 MB | 约 30–45 min（预估） |

缓存体积差异的原因：`experiment.py` 的 `nx=200`，模拟数据系列 `nx=501`，
三个 `(n_saves, n_x)` 历史数组体积差约 3 倍。详见 [§6](#6-solver-本地缓存)。

---

## 1. `experiment.py` — 影像数据完整算例

### 1.1 运行

所有命令都在**仓库根目录**执行；脚本内部按 `__file__` 定位仓库根，
因此在任意工作目录下运行都能正确读写 `data/`。

```bash
python Scripts/experiment.py              # 有缓存则复用，否则跑标定并写入缓存
python Scripts/experiment.py --rerun      # 忽略缓存，强制重跑标定
python Scripts/experiment.py --plot-only  # 只读缓存重画；缓存缺失则报错
```

依赖：

```bash
pip install numpy scipy plotly joblib
```

Plotly 只写 HTML 文件，无图形界面时不需要设置 `MPLBACKEND`。

### 1.2 `example()` 流程

`example()` 是 `main()` 调用的完整算例：

1. 从 `data/stenoses_data/` 读 `areas.joblib`、`branches.joblib`、
   `branch_diameters.joblib`，把像素单位换成 cm / cm²（`mm_per_pixel = 9.7/756`）。
2. `resample_axial_profile(...)` 沿程降采样到 `target_nx`，物理长度不变。
3. 查 solver 缓存；未命中则调
   `SimuCoronary.calculators.calibrate_q_scale_to_inlet_peak(...)`
   把入口峰值压收进 `(95, 125) mmHg`（`max_iters=10`，`record_interval_steps=30`）。
4. `plot_results()` 出五图交互页，写到 `data/cache/final_<name>_nx<nx>_qs<q>.html`。
5. 用 `integral_mean` 做质量守恒校验：`Q_in ≈ Q_out + Q_side`。
6. 返回 `solver.ffr_ratio()`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `target_nx` | `200` | 目标节点数；`None` 表示不降采样 |
| `frame_spacing_cm` | `0.02` | 原始帧间距 (cm)，用于推血管长度 |
| `duration_s` | `3.0` | 每次 `run()` 的物理时长 |
| `name` | `"stenoses"` | 算例名，用于缓存文件与输出命名 |
| `use_virtual_stent` | `False` | 是否调 `apply_virtual_stent()`（虚拟 FFR） |
| `force_rerun` | `False` | 忽略已有缓存 |
| `plot_only` | `False` | 只读缓存，缺缓存则报错 |

已知现象：`use_virtual_stent=False` 时 `min ffr` 约 0.28（重度狭窄）；
开启支架后显著回升，且同样流量下入口峰值压更低（血管整体变宽、阻力下降）。
数值对 `MINIMUM_AREA_RATIO`（`SimuCoronary/coronary_constants.py`）很敏感：
默认 0.2 时钳位较松，若调回 0.8 则 `min ffr` 会明显回升。

---

## 2. `simudata_experiment.py` — 模拟数据批量仿真

用 `SimuCoronary.simudata.SimulatedVessel` 生成 7 种典型几何，逐个跑 1D 求解器
（含 q_scale 外环标定），**进程池并行**。

```bash
python Scripts/simudata_experiment.py                # 默认 4 进程
python Scripts/simudata_experiment.py --workers 2    # 指定并行度
python Scripts/simudata_experiment.py --dry-run      # 只打印算例配置
python Scripts/simudata_experiment.py --only stenosis # 只跑匹配子串的算例
```

### 2.1 7 个算例

| key | 描述 | 近端→远端面积 (cm²) |
|-----|------|---------------------|
| `uniform` | ① 均匀直管 | 0.15 → 0.15 |
| `taper` | ② 锥形（近粗远细） | 0.15 → 0.09 |
| `stenosis_center` | ③ 中间狭窄，等截面 | 0.15 → 0.15，MLA 0.075 |
| `taper_stenosis` | ④ 锥形 + 中间狭窄 | 0.15 → 0.09，MLA 0.06 |
| `taper_reverse` | ⑤ 反向锥形（近细远粗） | 0.09 → 0.15 |
| `stenosis_proximal` | ⑥ 狭窄位于近端（0.25L） | 0.15 → 0.15 |
| `stenosis_distal` | ⑦ 狭窄位于远端（0.75L） | 0.15 → 0.15 |

统一设置：`L=10 cm`、`n_cells=501`（`dx=0.02 cm`）、`duration_s=3.0`、
标定窗口 `(95, 125) mmHg`、`max_iters=10`、`virtual_stent=False`、
3 条侧支（`x = 2.5 / 5.0 / 7.5 cm`，直径 0.15 cm）。

几何定义由 `build_cases()` 统一生成，另两个敏感性脚本通过参数复用它：

```python
build_cases(area_scaler=1.0, stenosis_rate=0.5, only_stenosis=False)
```

### 2.2 输出

（`results/模拟数据结果/`）

```text
summary.json                全部指标 + 诊断说明（notes 字段）
compare_ffr.png             FFR(x) 七例对比
compare_area.png            面积剖面对比
compare_pressure.png        时间平均压力对比
compare_summary.png         关键指标柱状图
<case>/geometry.png         面积剖面（原始 + 平滑 + 检测到的狭窄段）
<case>/final_<case>.html    Plotly 五图交互页
<case>/solver.pkl           solver 缓存（53–77 MB/例，可复现、可重画）
```

> **缓存状态**：现有 7 例结果是在加入缓存功能**之前**跑出的，因此目录里暂时
> 没有 `solver.pkl`。缓存会在下次运行新算例时自动生成，或指定
> `--only <case>` 单独重跑补齐。

### 2.3 并行度建议

每进程约 400–600 MB 峰值内存。16 核 / 32 GB 机器用 4 进程约 24 分钟；
串行则需约 3 小时（带狭窄算例每例 19 分钟）。

### 2.4 已知数值特征

（详见 `summary.json` 的 `notes`）

- `mass_err` 在等截面无狭窄算例上约 0，锥形约 1%，带狭窄 + 3 侧支时约 5–9%；
  实测 `dvdt ≈ 0` 说明这来自数值耗散而非瞬态。
- FFR 曲线在侧支开口处的台阶是真实的局部压降。
- `q_scale` 在狭窄算例上升到 1.25，因为阻力偏大使入口峰值压低于目标窗下沿。

---

## 3. `simudata_experiment_different_scale.py` — 面积缩放敏感性

复用上节的 7 种几何定义，只改变 `AREA_SCALER`：

```python
prox_area = 0.15 * AREA_SCALER
dist_area = 0.15 * AREA_SCALER      # 锥形例再 × 0.6
```

扫描取值 `0.3 / 0.6 / 0.9 / 1.2 / 1.5`，共 **5 × 7 = 35 个算例**。
血管长度、狭窄率、狭窄位置/占比全部不变，因此相当于把整条血管等比缩放
（面积 × scaler，直径 × √scaler）。

```bash
python Scripts/simudata_experiment_different_scale.py              # 4 进程
python Scripts/simudata_experiment_different_scale.py --workers 6
python Scripts/simudata_experiment_different_scale.py --scalers 0.3 1.5
python Scripts/simudata_experiment_different_scale.py --dry-run
```

### 3.1 输出

（`results/模拟数据实验2（面积大小变化）/`）

```text
summary.json                        全部 35 例指标 + scan 设置
compare_ffr_by_scale.png            每个 scaler 一个子图的 FFR(x)
compare_ffr_uniform.png             同一几何（uniform）跨 scaler 对比
compare_ffr_min_vs_scale.png        FFR_min 随 scaler 变化（7 条曲线）
compare_pressure_drop_vs_scale.png  时间平均压降随 scaler 变化
compare_q_scale_vs_scale.png        标定 q_scale 随 scaler 变化
compare_summary.png                 关键指标柱状图（按 case 分组）
scale_<s>/<case>/geometry.png       面积剖面
scale_<s>/<case>/final_<case>.html  Plotly 五图交互页
scale_<s>/<case>/solver.pkl         solver 缓存（53–77 MB/例，35 例合计 2.26 GB）
```

### 3.2 读取缓存

```bash
python Scripts/simudata_experiment_different_scale.py --list-cached
python Scripts/simudata_experiment_different_scale.py --replot
python Scripts/simudata_experiment_different_scale.py --replot scale_0.90/uniform
```

```python
import sys; sys.path.insert(0, "Scripts")
from simudata_experiment_different_scale import (
    load_scale_case_solver,   # (scaler, case) → {"solver", "cal", "meta", "path"}
    list_cached_cases,        # {"scale_0.90/uniform": pkl 路径, ...}
    replot_from_cache,
)

bundle = load_scale_case_solver(1.2, "stenosis_center")
solver = bundle["solver"]                 # 已完成 run()
print(bundle["meta"]["area_scaler"])      # 1.2
ffr = solver.ffr_ratio()
```

`scaler` 既可以是数值（`1.2`），也可以是目录名（`"scale_1.20"`）。

### 3.3 实测结论（35/35 成功，96 分钟）

**最小 FFR 随 AREA_SCALER 变化**

| case | s=0.3 | s=0.6 | s=0.9 | s=1.2 | s=1.5 |
|------|-------|-------|-------|-------|-------|
| stenosis_distal | **0.774** | 0.932 | 0.963 | 0.975 | 0.980 |
| taper_stenosis | **0.800** | 0.933 | 0.967 | 0.980 | 0.988 |
| stenosis_center | 0.865 | 0.953 | 0.977 | 0.987 | 0.991 |
| stenosis_proximal | 0.864 | 0.958 | 0.984 | 0.994 | 1.000 |
| taper | 0.905 | 0.971 | 0.984 | 0.989 | 0.992 |
| taper_reverse | 0.929 | 0.973 | 0.986 | 0.991 | 0.994 |
| uniform | 0.902 | 0.973 | 0.988 | 0.993 | 0.996 |

**时间平均压降（mmHg）**

| case | s=0.3 | s=0.6 | s=0.9 | s=1.2 | s=1.5 |
|------|-------|-------|-------|-------|-------|
| stenosis_distal | 21.16 | 5.94 | 3.09 | 2.48 | 1.97 |
| taper_stenosis | 19.47 | 4.96 | 2.09 | 1.27 | 0.69 |
| stenosis_center | 10.98 | 3.70 | 1.86 | 1.36 | 0.91 |
| stenosis_proximal | 11.59 | 3.26 | 1.58 | 0.56 | 0.01 |
| uniform | 8.94 | 2.28 | 1.02 | 0.58 | 0.37 |

两个要点：

- **强非线性**：s 从 0.3→0.6 时压降降为 1/3~1/4，但 1.2→1.5 只降不到 20%，
  符合泊肃叶律（阻力 ∝ 1/A² = 1/r⁴）。
- **只有 s=0.3 时两个远端狭窄算例跌破 0.80 临床阈值**；`stenosis_distal`
  危害最大，因为远端本身面积已最小。

### 3.4 一个设计取舍

侧支直径固定为 `0.15 cm`，**不随** `area_scaler` 缩放。因此小 scaler 时
侧支相对主干更粗、分流比更大，会**放大**小 scaler 组的压降。
若需要「纯几何缩放」，侧支直径应改为 `0.15 * sqrt(scaler)`。

---

## 4. `simudata_experiment_different_stenosis_rate.py` — 狭窄率敏感性

只取 **4 种含狭窄几何**（`uniform` / `taper` / `taper_reverse` 不含狭窄，
与本扫描无关），把 `STENOSIS_RATE` 取一系列值：

```text
stenosis_center     等截面 + 中间狭窄
taper_stenosis      锥形 + 中间狭窄
stenosis_proximal   等截面 + 近端狭窄（x = 0.25L）
stenosis_distal     等截面 + 远端狭窄（x = 0.75L）
```

扫描取值 `0.9 / 0.8 / 0.7 / 0.6 / 0.5`（由重到轻），共 **5 × 4 = 20 个算例**。
其余参数与主实验一致（`AREA_SCALER=1.0`、3 条侧支、`duration_s=3.0`）。

> **状态**：脚本已就绪，**尚未运行**（20 例预计 30–45 分钟）。

### 4.1 狭窄率的含义

`MLA = 0.5·(prox + dist)·(1 − STENOSIS_RATE)`

| rate | 等截面例 MLA | 含义 |
|------|-------------|------|
| 0.9 | 0.0150 cm² | MLA 只剩参考面积的 **10%**（重度） |
| 0.8 | 0.0300 cm² | 20% |
| 0.7 | 0.0450 cm² | 30% |
| 0.6 | 0.0600 cm² | 40% |
| 0.5 | 0.0750 cm² | 50%（中度） |

**锥形例的特殊性**：`taper_stenosis` 的近/远端面积不等（0.15 / 0.09 cm²），
按上式 `MLA = 0.12·(1−rate)`，因此同等 rate 下其 MLA 比等截面例小 20%，
绝对管腔更细。跨几何比较时需注意这一点。

### 4.2 运行

```bash
python Scripts/simudata_experiment_different_stenosis_rate.py              # 4 进程
python Scripts/simudata_experiment_different_stenosis_rate.py --workers 6
python Scripts/simudata_experiment_different_stenosis_rate.py --rates 0.9 0.5
python Scripts/simudata_experiment_different_stenosis_rate.py --dry-run
```

### 4.3 输出

（`results/模拟数据实验3（狭窄率变化）/`）

```text
summary.json                              全部 20 例指标 + scan 设置
compare_ffr_by_rate.png                   每个狭窄率一个子图的 FFR(x)
compare_ffr_stenosis_center.png           同一几何跨狭窄率对比
compare_ffr_min_vs_rate.png               最小 FFR 随狭窄率变化（4 条曲线）
compare_pressure_drop_vs_rate.png         压降随狭窄率变化
compare_q_scale_vs_rate.png               标定 q_scale 随狭窄率变化
compare_mla_vs_rate.png                   最小 FFR 与 MLA 的关系（标注狭窄率）
compare_summary.png                       关键指标柱状图
rate_<r>/<case>/geometry.png              面积剖面
rate_<r>/<case>/final_<case>.html         Plotly 五图交互页
rate_<r>/<case>/solver.pkl                solver 缓存（53–77 MB/例）
```

### 4.4 读取缓存

```bash
python Scripts/simudata_experiment_different_stenosis_rate.py --list-cached
python Scripts/simudata_experiment_different_stenosis_rate.py --replot
python Scripts/simudata_experiment_different_stenosis_rate.py --replot rate_0.90/stenosis_center
```

```python
import sys; sys.path.insert(0, "Scripts")
from simudata_experiment_different_stenosis_rate import (
    load_rate_case_solver,    # (rate, case) → {"solver", "cal", "meta", "path"}
    list_cached_cases,        # {"rate_0.90/stenosis_center": pkl 路径, ...}
    replot_from_cache,
)

bundle = load_rate_case_solver(0.9, "stenosis_center")
solver = bundle["solver"]                 # 已完成 run()
print(bundle["meta"]["stenosis_rate"])    # 0.9
print(bundle["meta"]["mla_area_cm2"])     # 0.015
```

`rate` 既可以是数值（`0.9`），也可以是目录名（`"rate_0.90"`）。

---

## 5. `plot_results()` — 五图交互版

实现位于 **`SimuCoronary/utils.py`**（包内通用可视化工具），三个批处理脚本都调用它。

```python
from SimuCoronary.utils import plot_results

plot_results(
    solver,                  # 已 run() 的 NavierStokes1D
    show=False,              # 是否打开浏览器
    save_path="out.html",    # .html 保留交互；其它后缀走 write_image
)
```

其余可选参数：`max_slider_steps=60`（拖动条步数）、`max_time_points=1500`
（时程曲线抽稀上限，仅影响显示密度）、`t_index` / `x_index`（初始位置）。

同模块另导出：

| 函数 | 用途 |
|------|------|
| `even_indices(n, max_count)` | 拖动条均匀取点 |
| `decimate_indices(n, max_points)` | 时程曲线显示抽稀 |
| `rescale_y_domains(fig, lo, hi)` | 压缩子图 y 域，腾出标题/图例/拖动条空间 |
| `save_solver_bundle` / `load_solver_bundle` | solver 缓存读写（见 §6） |
| `slim_calibration_report` | 剔除标定 report 中的二维场数据 |

### 5.1 版面

按「受哪个拖动条控制」分列（3 行 × 2 列）：

```text
┌───────────────────────┬───────────────────────┐
│ ① A(x)       [图例]   │ ③ Q(t)        [图例]  │
│  ─── 时间 t ───       │                       │
├───────────────────────┼───────────────────────┤
│ ② FFR(x)     [图例]   │ ④ p(t)        [图例]  │
├───────────────────────┼───────────────────────┤
│                       │ ⑤ A(t)        [图例]  │
│                       │  ─── 空间 x ───       │
└───────────────────────┴───────────────────────┘
```

左列是只随空间变化的沿程分布图（`A(x)`、`FFR(x)`），时间拖动条紧贴
`A(x)` 下方；右列三幅时程图全部受空间坐标控制，纵向对齐便于比较，
空间拖动条位于三幅图的最下方。FFR 已在时间上积分，故无拖动条、放在左列末位。

| # | 图 | 拖动条 | 数据源 |
|---|----|--------|--------|
| ① | `A(x)` 沿程分布 | **时间 t** | `history_area[t]`，另叠 `area_ref` 与狭窄段标注 |
| ② | `FFR(x)` | 无 | `ffr_ratio()`，附 `FFR=0.80` 参考线 |
| ③ | `Q(t)` 时程 | **空间 x** | `history_flow[:, x]` |
| ④ | `p(t)` 时程 | **空间 x** | `history_pressure[:, x]` |
| ⑤ | `A(t)` 时程 | **空间 x** | `history_area[:, x]` |

初始位置默认取**末时刻**与**最狭窄处**（`lesions[0].mla_idx`），
可用 `t_index` / `x_index` 覆盖。图例标题与 hover 文本都会随拖动条实时更新。

### 5.2 三个实现要点

1. **每图独立图例**。Plotly 只有一个全局 `legend`，五个子图共用一个会把
   不同含义的条目混在一起。这里分配 `legend` / `legend2` … `legend5`，
   各自放在对应子图**外部上方右侧**（与子图标题同一行），不遮挡曲线区。
2. **拖动条用 `restyle` 而非 `frames` + `animate`**。两组拖动条若共用
   frames，Plotly 会把它们排进同一动画队列（`_frameQueue`），导致拖动条
   自行连续推进（实测 `active` 会一直自增）且互相干扰。`restyle` 直接改
   trace 数据，无队列、无耦合。另需注意 `steps[].value` 必须用**步序号**
   而非数据下标，否则 `slider.active` 匹配不上、初始位置会漂移。
3. **压缩 y 域**（`rescale_y_domains`）。`make_subplots` 生成的 y 域总是
   铺满 `[0, 1]`，画布上下没有空间放子图标题、图例和拖动条；整体重映射到
   `[0.17, 0.95]` 后才放得下，拖动条位置再按各子图域下沿计算。

另需注意：初始位置用 `_nearest_step()` 取**最近的滑块步**，而不是直接查表。
当 `n_x > max_slider_steps` 时（如 `n_x=501`、滑块 60 步），MLA 下标可能不在
滑块候选集里，直接查字典会 `KeyError`。

---

## 6. solver 本地缓存

三个批处理脚本与 `experiment.py` 都会把 `solver` 整体缓存，重画图表或做二次
分析都不必重跑仿真（单例标定 5–20 分钟，读缓存只需 1–2 秒）。

| 脚本 | 缓存路径 | 单例体积 |
|------|----------|----------|
| `experiment.py` | `data/cache/solver/<name>_nx<nx>_T<时长>[_stent].pkl` | ~18 MB |
| `simudata_experiment.py` | `results/模拟数据结果/<case>/solver.pkl` | 53–77 MB |
| `..._different_scale.py` | `results/模拟数据实验2…/scale_<s>/<case>/solver.pkl` | 53–77 MB |
| `..._different_stenosis_rate.py` | `results/模拟数据实验3…/rate_<r>/<case>/solver.pkl` | 53–77 MB |

格式统一为 pickle protocol 5，bundle 结构 `{"solver", "cal", "meta"}`。
`meta` 记录 `case_key` / `case_dir` / `vessel_kwargs` / `settings`（以及
`area_scaler` 或 `stenosis_rate`），足以完整复现。

### 6.1 体积说明

体积差异全部来自 `solver.history_*`：

```
history_area     (n_saves, n_x)   float64
history_flow     (n_saves, n_x)
history_pressure (n_saves, n_x)
```

- `experiment.py`：`n_x=200`、`n_saves≈3700` → 三个数组各 ~5.7 MB → 合计 ~18 MB
- 模拟数据系列：`n_x=501`、`n_saves≈3700–4700` → 各 15–18 MB → 合计 53–77 MB
  （帧数与算例有关，面积越小、狭窄越重则步长越小、帧数越多）

`cal["history"]` 中每轮迭代的二维场已由 `slim_calibration_report()` 剔除
（否则再加 68 MB）。若需进一步压缩，可只保留后半段稳态历史
（`solver.history_* = solver.history_*[n//2:]`），体积约减半。

### 6.2 通用接口

```python
from SimuCoronary.utils import save_solver_bundle, load_solver_bundle

save_solver_bundle({"solver": solver, "cal": cal, "meta": meta}, "out.pkl")
bundle = load_solver_bundle("out.pkl")
```

各脚本另有针对目录结构的便捷封装：

| 脚本 | 函数 |
|------|------|
| `simudata_experiment.py` | `load_case_solver(case_key)` / `list_cached_cases()` / `replot_from_cache()` |
| `..._different_scale.py` | `load_scale_case_solver(scaler, case)` / 同上 |
| `..._different_stenosis_rate.py` | `load_rate_case_solver(rate, case)` / 同上 |

行为约定（三者一致）：

- 缓存不存在 → `FileNotFoundError`，提示可用算例或先跑批处理
- `case_key=None` 且缓存中只有一个算例 → 自动选它；多个则 `ValueError`
- 返回的 `solver.history_*` 与 `ffr_ratio()` 与原对象逐位一致（已验证）

`results/*` 与 `data/cache/*` 均在 `.gitignore` 中，缓存不会进版本库。

---

## 7. 与包的关系

- 脚本只**调用**包内 API，不含算法实现；算法改动请同步 `docs/navier_stokes_1d.md`
  （见 `.cursor/rules/algorithm-doc-sync.mdc`）。
- `SimuCoronary/simudata.py` 提供 `SimulatedVessel`（参数化几何）；
  三个批处理脚本的所有几何都由它生成。
- `SimuCoronary/calculators.py` 提供 `calibrate_q_scale_to_inlet_peak(...)`；
  本节各脚本都用它做入口峰值压标定。
- `SimuCoronary/utils.py` 提供 `plot_results()` 与 solver 缓存读写工具；
  脚本只负责数据准备与编排。
- `SimuCoronary/navier_stokes_1d.py` 是纯库，**没有** `__main__`，
  不要尝试 `python -m SimuCoronary.navier_stokes_1d`。
