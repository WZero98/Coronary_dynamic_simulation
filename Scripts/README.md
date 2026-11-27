# Scripts — 实验脚本使用说明

本目录存放**脚本层**内容：算例编排、批量仿真、交互式可视化、solver 缓存。
算法模块全部位于 `../SimuCoronary/`，其规范说明见
[`docs/navier_stokes_1d.md`](../docs/navier_stokes_1d.md)。

## 文件一览

| 文件 | 作用 |
|------|------|
| `experiment.py` | 影像数据完整算例：读 `data/stenoses_data` → 重采样 → 标定 → 绘图 → 质量校验 |
| `simudata_experiment.py` | 模拟数据批量仿真：7 种参数化几何 |
| `simudata_experiment_different_scale.py` | 面积缩放敏感性：7 种几何 × 多个 `AREA_SCALER` |
| `simudata_experiment_different_stenosis_rate.py` | 狭窄率敏感性：4 种含狭窄几何 × 多个 `STENOSIS_RATE` |

后三个脚本共用同一套几何定义（`simudata_experiment.build_cases`）与可视化
（`SimuCoronary.utils.plot_results`），只是在扫描维度上不同。

---

## 1. `experiment.py` — 影像数据完整算例

### 1.1 运行

```bash
python Scripts/experiment.py              # 有缓存则复用，否则跑标定并写入缓存
python Scripts/experiment.py --rerun      # 忽略缓存，强制重跑标定
python Scripts/experiment.py --plot-only  # 只读缓存重画；缓存缺失则报错
```

### 1.2 接口

**`example(target_nx=200, frame_spacing_cm=0.02, duration_s=3.0, name="stenoses", use_virtual_stent=False, force_rerun=False, plot_only=False)`**

完整算例。流程：读 `data/stenoses_data/` 的 `areas.joblib` / `branches.joblib` /
`branch_diameters.joblib` → 像素单位换成 cm / cm² → `resample_axial_profile`
降采样 → 查缓存或调 `calibrate_q_scale_to_inlet_peak` 标定到 `(95, 125) mmHg`
→ `plot_results` 出交互页 → 质量守恒校验 `Q_in ≈ Q_out + Q_side` → 返回
`solver.ffr_ratio()`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `target_nx` | `200` | 目标节点数；`None` 表示不降采样 |
| `frame_spacing_cm` | `0.02` | 原始帧间距 (cm)，用于推血管长度 |
| `duration_s` | `3.0` | 每次 `run()` 的物理时长 |
| `name` | `"stenoses"` | 算例名，用于缓存与输出命名 |
| `use_virtual_stent` | `False` | 是否调 `apply_virtual_stent()`（虚拟 FFR） |
| `force_rerun` | `False` | 忽略已有缓存 |
| `plot_only` | `False` | 只读缓存，缺缓存则报错 |

**缓存接口**

| 函数 | 用途 |
|------|------|
| `solver_cache_path(name, nx, duration_s, use_virtual_stent)` | 由算例参数拼出缓存文件路径 |
| `save_solver_bundle(bundle, path)` | 落盘 `{"solver", "cal", "meta"}`（转发 `SimuCoronary.utils`） |
| `load_solver_bundle(path)` | 读回 bundle |

**`main(argv=None)`** 解析 `--rerun` / `--plot-only` 后调用 `example()`。

---

## 2. `simudata_experiment.py` — 模拟数据批量仿真

用 `SimuCoronary.simudata.SimulatedVessel` 生成参数化几何，逐个跑 1D 求解器
（含 q_scale 标定），**进程池并行**。

### 2.1 运行

```bash
python Scripts/simudata_experiment.py                 # 默认 4 进程
python Scripts/simudata_experiment.py --workers 6     # 指定并行度
python Scripts/simudata_experiment.py --only stenosis # 只跑 key/label 含该子串的算例
python Scripts/simudata_experiment.py --dry-run       # 只打印算例配置
python Scripts/simudata_experiment.py --out DIR       # 换输出根目录
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--workers` | `min(4, cpu_count)` | 并行进程数 |
| `--only` | `None` | 按 key 或 label 子串过滤算例 |
| `--dry-run` | — | 只打印清单，不跑仿真 |
| `--out` | `results/模拟数据实验/` | 输出根目录 |
| `--list-cached` | — | 列出已缓存算例后退出 |
| `--replot [CASE]` | — | 从缓存重画：给 CASE 只重画该例，不带值则全部 |

### 2.2 接口

| 函数 | 用途 |
|------|------|
| `build_cases(area_scaler=1.0, stenosis_rate=0.5, only_stenosis=False)` | 生成 7 个 `CaseSpec`（`only_stenosis=True` 时只返回 4 个含狭窄的）。另两个敏感性脚本复用它 |
| `run_one_case(spec, out_root, case_dir_name=None, extra_meta=None)` | 单例完整流程：构造几何 → 重采样 → 标定 → 绘图 → 缓存 → 返回指标字典。**模块级函数**，供 `ProcessPoolExecutor` pickle |
| `save_comparison_figures(results, out_root)` | 生成汇总对比图，返回文件路径列表 |
| `main(argv=None)` | CLI 入口 |

**缓存读取**

| 函数 | 用途 |
|------|------|
| `case_solver_path(case_key, out_root)` | 某算例的 `solver.pkl` 路径 |
| `list_cached_cases(out_root)` | `{case_key: pkl 路径}`，只列出已有缓存的算例 |
| `load_case_solver(case_key=None, out_root)` | 读缓存，返回 `{"solver", "cal", "meta", "path"}`。`case_key=None` 且只有一例时自动选它；多例则 `ValueError`，不存在则 `FileNotFoundError` |
| `replot_from_cache(case_key=None, out_root, save_path=None, show=False)` | 读缓存并重新出五图交互页 |

命令行对应：`--list-cached` 列出缓存、`--replot [CASE]` 重画（不带值为全部）。

```python
import sys; sys.path.insert(0, "Scripts")
from simudata_experiment import load_case_solver

bundle = load_case_solver("stenosis_center")
solver = bundle["solver"]                    # 已完成 run()
ffr = solver.ffr_ratio()
print(bundle["meta"]["vessel_kwargs"])       # 复现用几何参数
```

### 2.3 7 个算例

| key | 描述 | 近端→远端面积 (cm²) |
|-----|------|---------------------|
| `uniform` | ① 均匀直管 | 0.15 → 0.15 |
| `taper` | ② 锥形（近粗远细） | 0.15 → 0.09 |
| `stenosis_center` | ③ 中间狭窄，等截面 | 0.15 → 0.15，MLA 0.075 |
| `taper_stenosis` | ④ 锥形 + 中间狭窄 | 0.15 → 0.09，MLA 0.06 |
| `taper_reverse` | ⑤ 反向锥形（近细远粗） | 0.09 → 0.15 |
| `stenosis_proximal` | ⑥ 狭窄位于近端（0.25L） | 0.15 → 0.15 |
| `stenosis_distal` | ⑦ 狭窄位于远端（0.75L） | 0.15 → 0.15 |

### 2.4 关键常量

均定义在 `simudata_experiment.py` 顶部，供本脚本与两个派生脚本共用：

| 常量 | 默认 | 说明 |
|------|------|------|
| `LENGTH_CM` | `10.0` | 血管长度 (cm) |
| `DX_TARGET_CM` / `N_CELLS` | `0.02` / `501` | **生成**网格间距与节点数 |
| `RESAMPLE_NX` | `200` | 求解前重采样到的节点数。`dt ∝ dx`，降采样可显著减少时间步数 |
| `AREA_SCALER` | `1.0` | 面积缩放，`prox_area = 0.15 × AREA_SCALER` |
| `STENOSIS_RATE` / `_POSITION` / `_FRACTION` | `0.5` / `0.5` / `0.2` | 狭窄率、中心位置、长度占比 |
| `BRANCH_POSITIONS` / `BRANCH_DIAMETERS` | `[0.25, 0.5, 0.75]` / 0.15 cm | 3 条侧支 |
| `DURATION_S` / `RECORD_INTERVAL_STEPS` | `3.0` / `30` | 仿真时长与存帧间隔 |
| `PEAK_LO_MMHG` / `PEAK_HI_MMHG` / `MAX_ITERS` | `95` / `125` / `10` | 标定目标窗与最大迭代 |

### 2.5 输出目录结构

```text
results/模拟数据实验/
├── summary.json                全部指标 + settings + notes 诊断说明
├── compare_*.png               汇总对比图（FFR / 面积 / 压力 / 指标柱状图）
└── <case>/
    ├── geometry.png            面积剖面（原始 + 平滑 + 检测到的狭窄段）
    ├── final_<case>.html       Plotly 五图交互页
    └── solver.pkl              solver 缓存
```

---

## 3. `simudata_experiment_different_scale.py` — 面积缩放敏感性

复用 `build_cases()` 的 7 种几何，只改变 `AREA_SCALER`：

```python
prox_area = 0.15 * AREA_SCALER
dist_area = 0.15 * AREA_SCALER      # 锥形例再 × 0.6
```

血管长度、狭窄率、狭窄位置/占比全部不变，因此相当于把整条血管等比缩放
（面积 × scaler，直径 × √scaler）。

### 3.1 运行

```bash
python Scripts/simudata_experiment_different_scale.py              # 4 进程
python Scripts/simudata_experiment_different_scale.py --workers 6
python Scripts/simudata_experiment_different_scale.py --scalers 0.3 1.5
python Scripts/simudata_experiment_different_scale.py --dry-run
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--workers` | `min(4, cpu_count)` | 并行进程数 |
| `--scalers` | `0.3 0.6 0.9 1.2 1.5` | 要扫描的 `AREA_SCALER` 列表（`SCALERS` 常量） |
| `--only` | `None` | 按几何 key/label 过滤 |
| `--dry-run` | — | 只打印清单 |
| `--out` | `results/模拟数据实验2（面积大小变化）/` | 输出根目录 |
| `--list-cached` | — | 列出已缓存算例后退出 |
| `--replot [SCALE_DIR/CASE]` | — | 从缓存重画，形如 `scale_0.90/uniform`；不带值则全部 |

### 3.2 接口

| 函数 | 用途 |
|------|------|
| `_scaler_dir(scaler)` | scaler → 子目录名（`scale_0.90`，两位小数避免歧义） |
| `save_comparison_figures(results, out_root)` | 按 scaler 分组的对比图，返回路径列表 |
| `scale_case_dir(scaler, case_key)` | 某组合的目录绝对路径 |
| `list_cached_cases(out_root)` | `{"scale_0.90/uniform": pkl 路径}` |
| `load_scale_case_solver(scaler, case_key, out_root)` | 读缓存。`scaler` 可传数值 `1.2` 或目录名 `"scale_1.20"` |
| `replot_from_cache(scaler, case_key, out_root, save_path=None, show=False)` | 读缓存重画 |
| `_run_scale_job(job)` | 进程池任务包装（模块级，`job = (spec, dir_name, extra, out_root)`） |
| `main(argv=None)` | CLI 入口 |

```python
from simudata_experiment_different_scale import load_scale_case_solver

bundle = load_scale_case_solver(1.2, "stenosis_center")
solver = bundle["solver"]
print(bundle["meta"]["area_scaler"])     # 1.2
```

### 3.3 输出目录结构

```text
results/模拟数据实验2（面积大小变化）/
├── summary.json                        指标 + scan 设置
├── compare_ffr_by_scale.png            每个 scaler 一个子图的 FFR(x)
├── compare_ffr_<ref>.png               同一几何跨 scaler 对比
├── compare_ffr_min_vs_scale.png        FFR_min 随 scaler 变化
├── compare_pressure_drop_vs_scale.png  时间平均压降随 scaler 变化
├── compare_q_scale_vs_scale.png        标定 q_scale 随 scaler 变化
├── compare_summary.png                 关键指标柱状图
└── scale_<s>/<case>/
    ├── geometry.png / final_<case>.html / solver.pkl
```

### 3.4 设计取舍

侧支直径固定为 `0.15 cm`，**不随** `area_scaler` 缩放。因此小 scaler 时
侧支相对主干更粗、分流比更大，会放大该组的压降。若需要「纯几何缩放」，
把侧支直径改为 `0.15 * sqrt(scaler)` 即可。

---

## 4. `simudata_experiment_different_stenosis_rate.py` — 狭窄率敏感性

只取 **4 种含狭窄几何**（`uniform` / `taper` / `taper_reverse` 不含狭窄，
与本扫描无关），把 `STENOSIS_RATE` 取一系列值。

### 4.1 运行

```bash
python Scripts/simudata_experiment_different_stenosis_rate.py              # 4 进程
python Scripts/simudata_experiment_different_stenosis_rate.py --workers 6
python Scripts/simudata_experiment_different_stenosis_rate.py --rates 0.9 0.5
python Scripts/simudata_experiment_different_stenosis_rate.py --dry-run
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--workers` | `min(4, cpu_count)` | 并行进程数 |
| `--rates` | `0.9 0.8 0.7 0.6 0.5` | 要扫描的 `STENOSIS_RATE` 列表（`STENOSIS_RATES` 常量，由重到轻） |
| `--only` | `None` | 按几何 key/label 过滤 |
| `--dry-run` | — | 只打印清单 |
| `--out` | `results/模拟数据实验3（狭窄率变化）/` | 输出根目录 |
| `--list-cached` | — | 列出已缓存算例后退出 |
| `--replot [RATE_DIR/CASE]` | — | 从缓存重画，形如 `rate_0.90/stenosis_center`；不带值则全部 |

### 4.2 接口

| 函数 | 用途 |
|------|------|
| `_rate_dir(rate)` / `_rate_label(rate)` | rate → 子目录名（`rate_0.90`）/ 人类可读标签 |
| `save_comparison_figures(results, out_root)` | 按狭窄率分组的对比图，返回路径列表 |
| `rate_case_dir(rate, case_key)` | 某组合的目录绝对路径 |
| `list_cached_cases(out_root)` | `{"rate_0.90/stenosis_center": pkl 路径}` |
| `load_rate_case_solver(rate, case_key, out_root)` | 读缓存。`rate` 可传数值 `0.9` 或目录名 `"rate_0.90"` |
| `replot_from_cache(rate, case_key, out_root, save_path=None, show=False)` | 读缓存重画 |
| `_run_rate_job(job)` | 进程池任务包装（模块级） |
| `main(argv=None)` | CLI 入口 |

```python
from simudata_experiment_different_stenosis_rate import load_rate_case_solver

bundle = load_rate_case_solver(0.9, "stenosis_center")
solver = bundle["solver"]
print(bundle["meta"]["stenosis_rate"])   # 0.9
print(bundle["meta"]["mla_area_cm2"])    # 对应 MLA
```

### 4.3 狭窄率的含义

`MLA = 0.5·(prox + dist)·(1 − STENOSIS_RATE)`，取值越接近 1 狭窄越重。
等截面例（`A=0.15`）对应：

| rate | MLA |
|------|-----|
| 0.9 | 0.0150 cm² |
| 0.8 | 0.0300 cm² |
| 0.7 | 0.0450 cm² |
| 0.6 | 0.0600 cm² |
| 0.5 | 0.0750 cm² |

**锥形例的特殊性**：`taper_stenosis` 近/远端面积不等（0.15 / 0.09 cm²），
按上式 `MLA = 0.12·(1−rate)`，同等 rate 下 MLA 比等截面例小 20%。
跨几何比较时需注意。

### 4.4 输出目录结构

```text
results/模拟数据实验3（狭窄率变化）/
├── summary.json                        指标 + scan 设置
├── compare_ffr_by_rate.png             每个狭窄率一个子图的 FFR(x)
├── compare_ffr_<ref>.png               同一几何跨狭窄率对比
├── compare_ffr_min_vs_rate.png         最小 FFR 随狭窄率变化
├── compare_pressure_drop_vs_rate.png   压降随狭窄率变化
├── compare_q_scale_vs_rate.png         标定 q_scale 随狭窄率变化
├── compare_mla_vs_rate.png             FFR_min 与 MLA 的关系（标注狭窄率）
├── compare_summary.png                 关键指标柱状图
└── rate_<r>/<case>/
    ├── geometry.png / final_<case>.html / solver.pkl
```

---

## 5. `SimuCoronary.utils` 可视化与缓存工具

四个脚本共用的工具都在包里，不放在脚本目录。

### 5.1 绘图

```python
from SimuCoronary.utils import plot_results

plot_results(
    solver,                  # 已 run() 的 NavierStokes1D
    show=False,              # 是否打开浏览器
    title="",                # 图标题后缀
    renderer=None,           # 传给 fig.show(renderer=...)
    save_path="out.html",    # .html 保留交互；其它后缀走 write_image
    max_slider_steps=60,     # 拖动条最大离散步数
    max_time_points=1500,    # 时程曲线抽稀上限（仅影响显示密度）
    t_index=None,            # 初始时刻下标（默认末时刻）
    x_index=None,            # 初始空间下标（默认最狭窄处）
)
```

版面按「受哪个拖动条控制」分列（3 行 × 2 列）：

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

| # | 图 | 拖动条 | 数据源 |
|---|----|--------|--------|
| ① | `A(x)` 沿程分布 | **时间 t** | `history_area[t]` + `area_ref` + 狭窄段标注 |
| ② | `FFR(x)` | 无 | `ffr_ratio()`，附 `FFR=0.80` 参考线 |
| ③ | `Q(t)` 时程 | **空间 x** | `history_flow[:, x]` |
| ④ | `p(t)` 时程 | **空间 x** | `history_pressure[:, x]` |
| ⑤ | `A(t)` 时程 | **空间 x** | `history_area[:, x]` |

三处实现要点（修改时勿回退）：

1. **每图独立图例**。Plotly 只有一个全局 `legend`，五图共用会把不同含义的条目
   混在一起。这里分配 `legend` … `legend5`，各放在对应子图**外部上方右侧**。
2. **拖动条用 `restyle` 而非 `frames` + `animate`**。两组拖动条若共用 frames，
   Plotly 会排进同一动画队列（`_frameQueue`），导致自行连续推进且互相干扰。
   另需 `steps[].value` 用**步序号**而非数据下标，否则 `slider.active` 匹配不上。
3. **压缩 y 域**（`rescale_y_domains`）。`make_subplots` 的 y 域总铺满 `[0,1]`，
   画布上下没有空间放标题/图例/拖动条，需整体重映射到 `[0.17, 0.95]`。

初始位置用 `_nearest_step()` 取**最近的滑块步**而非直接查表：当
`n_x > max_slider_steps` 时 MLA 下标可能不在候选集里，直接查字典会 `KeyError`。

### 5.2 辅助函数

| 函数 | 用途 |
|------|------|
| `even_indices(n, max_count)` | 在 `[0, n-1]` 上均匀取不超过 `max_count` 个下标（含首尾） |
| `decimate_indices(n, max_points)` | 等步长抽稀，用于压缩时程曲线体积 |
| `rescale_y_domains(fig, lo, hi)` | 把所有子图 y 域线性重映射到 `[lo, hi]` |
| `save_solver_bundle(bundle, path)` | 落盘 `{"solver", "cal", "meta"}`（pickle protocol 5），落盘前精简 `cal` |
| `load_solver_bundle(path)` | 读回 bundle；文件不存在抛 `FileNotFoundError` |
| `slim_calibration_report(cal)` | 剔除标定 report 中每轮迭代的二维场数据（`area`/`flow`/`pressure`） |

---

## 6. solver 本地缓存

四个脚本都把求解结果缓存下来，重画图表或做二次分析不必重跑仿真。

| 脚本 | 缓存路径 |
|------|----------|
| `experiment.py` | `data/cache/solver/<name>_nx<nx>_T<时长>[_stent].pkl` |
| `simudata_experiment.py` | `results/模拟数据实验/<case>/solver.pkl` |
| `..._different_scale.py` | `results/模拟数据实验2…/scale_<s>/<case>/solver.pkl` |
| `..._different_stenosis_rate.py` | `results/模拟数据实验3…/rate_<r>/<case>/solver.pkl` |

bundle 统一为 `{"solver", "cal", "meta"}`：

- `solver` —— 已完成 `run()` 的 `NavierStokes1D`
- `cal` —— 标定 report（已由 `slim_calibration_report` 精简）
- `meta` —— `case_key` / `case_dir` / `vessel_kwargs` / `settings`，
  敏感性脚本另有 `area_scaler` 或 `stenosis_rate`，足以完整复现

**读取约定**（三个批处理脚本一致）：

- 缓存不存在 → `FileNotFoundError`，提示可用算例或先跑批处理
- `case_key=None` 且缓存中只有一例 → 自动选它；多例则 `ValueError`
- 返回的 `solver.history_*` 与 `ffr_ratio()` 与原对象逐位一致

**体积主要来自 `solver.history_*`**（三个 `(n_saves, n_x)` float64 数组）。
`cal["history"]` 里每轮迭代的二维场已在落盘时剔除，否则体积会翻倍。
如需进一步压缩，可只保留后半段稳态历史：

```python
solver.history_area = solver.history_area[n // 2:]
```

`results/*` 与 `data/cache/*` 均在 `.gitignore` 中，缓存不进版本库。

---

## 7. 与包的关系

- 脚本只**调用**包内 API，不含算法实现；算法改动请同步
  `docs/navier_stokes_1d.md`（见 `.cursor/rules/algorithm-doc-sync.mdc`）。
- `SimuCoronary/simudata.py` —— `SimulatedVessel`，三个批处理脚本的几何来源。
- `SimuCoronary/calculators.py` —— `calibrate_q_scale_to_inlet_peak(...)`，
  四个脚本都用它做入口峰值压标定。
- `SimuCoronary/utils.py` —— `plot_results()` 与 solver 缓存读写工具。
- `SimuCoronary/navier_stokes_1d.py` —— 纯库，**没有** `__main__`，
  不要尝试 `python -m SimuCoronary.navier_stokes_1d`。

### 依赖

```bash
pip install numpy scipy plotly joblib
```

Plotly 只写 HTML 文件，无图形界面时不需要设置 `MPLBACKEND`。
所有命令都在**仓库根目录**执行；脚本内部按 `__file__` 定位仓库根，
因此在任意工作目录下运行都能正确读写 `data/` 与 `results/`。
