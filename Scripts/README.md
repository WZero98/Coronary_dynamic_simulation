# Scripts — 算例与可视化脚本

本目录存放**脚本层**内容：算例编排、交互式可视化、solver 本地缓存。
算法模块全部位于 `../SimuCoronary/`，其规范说明见
[`docs/navier_stokes_1d.md`](../docs/navier_stokes_1d.md)。

| 文件 | 作用 |
|------|------|
| `experiment.py` | 可直跑的完整算例：读影像面积 → 重采样 → 标定 q_scale → 绘图 → 质量校验；同时提供 `plot_results()` 与 solver 缓存工具 |

---

## 1. 运行

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

---

## 2. `example()` 流程

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

已知现象：`use_virtual_stent=False` 时 `min ffr` 约 0.51（重度狭窄）；
开启支架后显著回升，且同样流量下入口峰值压更低（血管整体变宽、阻力下降）。

---

## 3. `plot_results()` — 五图交互版

```python
import sys; sys.path.insert(0, "Scripts")
from experiment import plot_results

plot_results(
    solver,                  # 已 run() 的 NavierStokes1D
    show=False,              # 是否打开浏览器
    save_path="out.html",    # .html 保留交互；其它后缀走 write_image
)
```

其余可选参数：`max_slider_steps=60`（拖动条步数）、`max_time_points=1500`
（时程曲线抽稀上限，仅影响显示密度）、`t_index` / `x_index`（初始位置）。

### 3.1 版面

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

### 3.2 三个实现要点

1. **每图独立图例**。Plotly 只有一个全局 `legend`，五个子图共用一个会把
   不同含义的条目混在一起。这里分配 `legend` / `legend2` … `legend5`，
   各自放在对应子图**外部上方右侧**（与子图标题同一行），不遮挡曲线区。
2. **拖动条用 `restyle` 而非 `frames` + `animate`**。两组拖动条若共用
   frames，Plotly 会把它们排进同一动画队列（`_frameQueue`），导致拖动条
   自行连续推进（实测 `active` 会一直自增）且互相干扰。`restyle` 直接改
   trace 数据，无队列、无耦合。另需注意 `steps[].value` 必须用**步序号**
   而非数据下标，否则 `slider.active` 匹配不上、初始位置会漂移。
3. **压缩 y 域**（`_rescale_y_domains`）。`make_subplots` 生成的 y 域总是
   铺满 `[0, 1]`，画布上下没有空间放子图标题、图例和拖动条；整体重映射到
   `[0.17, 0.95]` 后才放得下，拖动条位置再按各子图域下沿计算。

---

## 4. solver 本地缓存

`experiment.py` 把标定得到的 `solver` 整体缓存，反复调图不必重跑数分钟的标定。
实测同一算例：首次约 9 分钟（4 轮标定），缓存命中后 **1.6 秒**。

| 项 | 说明 |
|----|------|
| 路径 | `data/cache/solver/<name>_nx<nx>_T<时长>[_stent].pkl` |
| 格式 | pickle protocol 5 |
| 体积 | 约 18 MB（见下方体积说明） |
| 命名 | 含 `name` / `nx` / `duration_s` / 是否启用虚拟支架，避免不同算例互相污染 |
| 忽略 | `data/cache/*` 已在 `.gitignore` 中 |

可用接口：`solver_cache_path(...)`、`save_solver_bundle(bundle, path)`、
`load_solver_bundle(path)`。bundle 结构为 `{"solver", "cal", "meta"}`。

**体积说明**：初版缓存 85.7 MB，原因是 `cal["history"]` 每轮迭代都存了完整的
`(n_t, n_x)` 场数据（`area` / `flow` / `pressure`），4 轮共约 68 MB，而绘图
只用 `solver` 自带的 `history_*`。落盘前剔除这些二维数组后降到 **18.2 MB**
（-79%），标量与各探针（`p_prox` / `q_sten` / `ffr` 等）全部保留。
逐轮完整场数据仍在 `data/cache/q_scale_cal_*/iter_XX.npz`，需要时可从那里读。

反序列化后 `history_time` / `history_area` / `history_flow` / `history_pressure`
与 `ffr_ratio()` 均与原对象逐位一致，可直接用于 `plot_results()`。

---

## 5. 与包的关系

- 脚本只**调用**包内 API，不含算法实现；算法改动请同步 `docs/navier_stokes_1d.md`
  （见 `.cursor/rules/algorithm-doc-sync.mdc`）。
- `SimuCoronary/calculators.py` 提供可复用的外环标定函数
  `calibrate_q_scale_to_inlet_peak(...)`；本目录的 `example()` 是它的一个完整用例。
- `SimuCoronary/navier_stokes_1d.py` 是纯库，**没有** `__main__`，
  不要尝试 `python -m SimuCoronary.navier_stokes_1d`。
