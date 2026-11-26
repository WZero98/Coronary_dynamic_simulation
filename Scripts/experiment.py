"""
experiment.py — 可直接运行的算例脚本

读 ``data/stenoses_data``，调用 ``SimuCoronary.calculators.calibrate_q_scale_to_inlet_peak``
把入口峰值压收到目标窗，再绘图、做质量校验并返回 FFR。

``solver`` 会缓存到 ``data/cache/solver/``，调图时无需重跑标定：

    python Scripts/experiment.py              # 有缓存则复用，否则跑标定
    python Scripts/experiment.py --rerun      # 强制重跑标定
    python Scripts/experiment.py --plot-only  # 只用缓存重画（无缓存则报错）

所有数据读写路径都按**仓库根目录**解析，与当前工作目录无关。
"""

from __future__ import annotations

import os
import pickle
import sys

# 本脚本位于 <root>/Scripts/，仓库根 = 上一级
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from SimuCoronary.calculators import calibrate_q_scale_to_inlet_peak  # noqa: E402
from SimuCoronary.geometry import resample_axial_profile  # noqa: E402
from SimuCoronary.navier_stokes_1d import (  # noqa: E402
    BloodFlowParameters,
    NavierStokes1D,
    integral_mean,
)

_SOLVER_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "cache", "solver")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _even_indices(n: int, max_count: int) -> list[int]:
    """在 ``[0, n-1]`` 上均匀取至多 ``max_count`` 个整数下标（含首尾，去重保序）。"""
    if n <= 0:
        return [0]
    if max_count < 1:
        max_count = 1
    if n <= max_count:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, max_count)})


def _decimate(n: int, max_points: int) -> np.ndarray:
    """等步长抽稀到至多 ``max_points`` 个样本；仅影响显示密度。"""
    if n <= 0:
        return np.zeros(0, dtype=int)
    if max_points < 2 or n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _label(text: str, k: int, n: int, every: int) -> str:
    """拖动条刻度标签：稀疏显示，避免密集重叠。"""
    if n <= 1:
        return "" if k else text
    return text if (k % max(every, 1) == 0 or k == n - 1) else ""


def _rescale_y_domains(fig, lo: float, hi: float) -> None:
    """把所有子图的 y 域线性重映射到 ``[lo, hi]``。

    ``make_subplots`` 生成的域总是铺满 ``[0, 1]``，画布顶部/底部之外没有
    空间放子图标题、图例和拖动条。整体压缩后这些元素才能落在画布内。
    """
    span = hi - lo
    for key in list(fig.layout):
        if not key.startswith("yaxis"):
            continue
        axis = getattr(fig.layout, key)
        dom = axis.domain
        if dom is None:
            continue
        axis.domain = (lo + dom[0] * span, lo + dom[1] * span)


# ---------------------------------------------------------------------------
# solver 本地缓存：调图时不必重跑标定
# ---------------------------------------------------------------------------
def solver_cache_path(
    name: str,
    nx: int,
    duration_s: float,
    use_virtual_stent: bool,
) -> str:
    """缓存文件路径。命名含决定结果的关键参数，避免误用别的算例。"""
    suffix = "_stent" if use_virtual_stent else ""
    return os.path.join(
        _SOLVER_CACHE_DIR, f"{name}_nx{nx}_T{duration_s:.2f}{suffix}.pkl"
    )


def save_solver_bundle(bundle: dict, path: str) -> str:
    """把 ``{"solver", "cal", "meta"}`` 落盘（pickle protocol 5）。

    ``cal["history"]`` 里每轮迭代都带完整 ``(n_t, n_x)`` 场数据（``area`` /
    ``flow`` / ``pressure``），实测 4 轮约 68 MB，而绘图只用 ``solver`` 自带的
    ``history_*``。因此落盘前把 ``history`` 精简为标量与近端/狭窄/远端探针，
    体积可降到 ~1/5。完整的逐轮场数据仍在
    ``data/cache/q_scale_cal_.../iter_XX.npz``，需要时可从那里读。
    """
    slim = dict(bundle)
    cal = slim.get("cal")
    if isinstance(cal, dict) and isinstance(cal.get("history"), list):
        cal = dict(cal)
        cal["history"] = [
            {
                k: v
                for k, v in snap.items()
                if not (
                    isinstance(v, np.ndarray) and v.ndim == 2
                )
            }
            for snap in cal["history"]
        ]
        slim["cal"] = cal

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(slim, f, protocol=5)
    return path


def load_solver_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_results(
    solver: NavierStokes1D,
    show: bool = True,
    title: str = "",
    renderer: str | None = None,
    save_path: str | None = None,
    *,
    max_slider_steps: int = 60,
    max_time_points: int = 1500,
    t_index: int | None = None,
    x_index: int | None = None,
):
    """五幅图的可交互 Plotly 版本，按「受哪个拖动条控制」分列排布。

    版面（3 行 × 2 列）::

        ┌───────────────────────┬───────────────────────┐
        │ ① A(x)       [图例]   │ ③ Q(t)        [图例]  │
        │  ─── 时间 t ───       │                       │
        ├───────────────────────┼───────────────────────┤
        │ ② FFR(x)     [图例]   │ ④ p(t)        [图例]  │
        ├───────────────────────┼───────────────────────┤
        │                       │ ⑤ A(t)        [图例]  │
        │                       │  ─── 空间 x ───       │
        └───────────────────────┴───────────────────────┘

    左列是只随空间变化的沿程分布图（A(x)、FFR(x)），时间拖动条紧贴
    A(x) 下方；右列三幅时程图全部受空间坐标控制，纵向对齐便于比较，
    空间拖动条位于三幅图的最下方。

    ==============  ==================  =====================
    子图            拖动条              数据
    ==============  ==================  =====================
    ① ``A(x)``      **时间 t**          ``history_area[t]`` + ``area_ref``
    ② ``FFR(x)``    无（时间已积分）    ``ffr_ratio()``
    ③ ``Q(t)``      **空间 x**          ``history_flow[:, x]``
    ④ ``p(t)``      **空间 x**          ``history_pressure[:, x]``
    ⑤ ``A(t)``      **空间 x**          ``history_area[:, x]``
    ==============  ==================  =====================

    Parameters
    ----------
    solver :
        已完成 ``run()`` 的 ``NavierStokes1D`` 实例
    show :
        是否调用 ``fig.show()``
    title :
        图标题后缀
    renderer :
        传给 ``fig.show(renderer=...)``，如 "browser"
    save_path :
        保存路径；``.html`` 写交互页（拖动条随文件保留），其它后缀走
        ``write_image``（静态图，需 kaleido）
    max_slider_steps :
        每条拖动条的最大离散步数；越大越平滑，HTML 也越大
    max_time_points :
        时程曲线抽稀后的最大点数，仅影响显示密度
    t_index, x_index :
        初始时刻/初始空间下标；默认取末时刻与最狭窄处（无狭窄取中点）

    Notes
    -----
    - **每图独立图例**：Plotly 只有一个全局 ``legend``，五个子图共用会把
      不同含义的条目混在一起。这里分配 ``legend``…``legend5``，放在各子图
      **外部上方右侧**（与子图标题同一行），不遮挡曲线。
    - **拖动条用 ``restyle`` 而非 ``frames`` + ``animate``**：两组拖动条若
      共用 frames，Plotly 会把它们排进同一动画队列（``_frameQueue``），
      导致拖动条自行连续推进且互相干扰。
    - **整体压缩 y 域**（见 ``_rescale_y_domains``）：画布上下留边距，
      子图标题、图例、拖动条才有位置可放。
    """
    if solver.history_time is None:
        raise RuntimeError("请先调用 run()")
    if solver.history_area is None or solver.history_flow is None:
        raise RuntimeError("请先调用 run()")
    if solver.history_pressure is None:
        raise RuntimeError("请先调用 run()")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tt = np.asarray(solver.history_time, dtype=float).ravel()
    xx = np.asarray(solver.x, dtype=float).ravel()
    area = np.asarray(solver.history_area, dtype=float)
    flow = np.asarray(solver.history_flow, dtype=float)
    pres = np.asarray(solver.history_pressure, dtype=float)

    n_t, n_x = tt.size, xx.size
    if area.shape != (n_t, n_x) or flow.shape != (n_t, n_x) or pres.shape != (n_t, n_x):
        raise ValueError(
            "history 数组形状与 (n_t, n_x) 不一致："
            f"area{area.shape}, flow{flow.shape}, pressure{pres.shape}, "
            f"n_t={n_t}, n_x={n_x}"
        )

    ffr = np.asarray(solver.ffr_ratio(), dtype=float).ravel()

    # ---- 初始位置 ----
    i_t = n_t - 1 if t_index is None else int(np.clip(t_index, 0, n_t - 1))

    if x_index is not None:
        i_x = int(np.clip(x_index, 0, n_x - 1))
    else:
        i_x = n_x // 2
        lesions = getattr(solver, "lesions", None)
        if lesions:
            try:
                i_x = int(lesions[0].mla_idx)
            except (AttributeError, IndexError, TypeError):
                pass
    i_x = int(np.clip(i_x, 0, n_x - 1))

    # ---- 拖动条离散下标 ----
    t_idx = _even_indices(n_t, max_slider_steps)
    x_idx = _even_indices(n_x, max_slider_steps)
    t_pos = {v: k for k, v in enumerate(t_idx)}
    x_pos = {v: k for k, v in enumerate(x_idx)}

    # ---- 时程曲线抽稀（仅显示） ----
    t_sel = _decimate(n_t, max_time_points)
    t_disp = tt[t_sel]

    def t_name(i: int) -> str:
        return f"A(x)，t = {tt[i]:.3f} s"

    def x_name(var: str, i: int) -> str:
        return f"{var}(t)，x = {xx[i]:.3f} cm"

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [None, {}]],
        vertical_spacing=0.22,
        horizontal_spacing=0.10,
    )

    # ---- trace 0/1: A(x) @ t（时间拖动条控制），(1,1) ----
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=area[i_t],
            mode="lines",
            name=t_name(i_t),
            legend="legend",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="x=%{x:.3f} cm<br>A=%{y:.5f} cm²<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=solver.area_ref,
            mode="lines",
            name="A₀ 参考",
            legend="legend",
            line=dict(color="rgba(0,0,0,0.55)", width=1.6, dash="dot"),
            hovertemplate="x=%{x:.3f} cm<br>A₀=%{y:.5f} cm²<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # ---- trace 2: FFR(x)（无拖动条），(2,1) ----
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=ffr,
            mode="lines",
            name="FFR（含侧支）" if solver.side_branches else "FFR",
            legend="legend2",
            line=dict(color="darkred", width=2.2),
            hovertemplate="x=%{x:.3f} cm<br>FFR=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=0.80, line_dash="dot", line_color="rgba(0,0,0,0.7)", row=2, col=1
    )

    # ---- trace 3/4/5: Q(t)/p(t)/A(t) @ x（空间拖动条控制），右列 ----
    fig.add_trace(
        go.Scatter(
            x=t_disp,
            y=flow[t_sel, i_x],
            mode="lines",
            name=x_name("Q", i_x),
            legend="legend3",
            line=dict(color="#2ca02c", width=1.8),
            hovertemplate="t=%{x:.4f} s<br>Q=%{y:.4f} cm³/s<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t_disp,
            y=pres[t_sel, i_x],
            mode="lines",
            name=x_name("p", i_x),
            legend="legend4",
            line=dict(color="#d62728", width=1.8),
            hovertemplate="t=%{x:.4f} s<br>p=%{y:.3f} mmHg<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t_disp,
            y=area[t_sel, i_x],
            mode="lines",
            name=x_name("A", i_x),
            legend="legend5",
            line=dict(color="#9467bd", width=1.8),
            hovertemplate="t=%{x:.4f} s<br>A=%{y:.5f} cm²<extra></extra>",
        ),
        row=3,
        col=2,
    )

    # ---- 狭窄段标注（沿程分布图） ----
    lesions = getattr(solver, "lesions", None)
    if lesions:
        try:
            seg = lesions[0]
            x_prox = float(xx[int(seg.prox_ref_idx)])
            x_dist = float(xx[int(seg.dist_ref_idx)])
            x_mla = float(xx[int(seg.mla_idx)])
        except (AttributeError, IndexError, TypeError):
            x_prox = None
        if x_prox is not None:
            for r in (1, 2):
                fig.add_vrect(
                    x0=x_prox,
                    x1=x_dist,
                    fillcolor="rgba(255,165,0,0.13)",
                    line_width=0,
                    layer="below",
                    row=r,
                    col=1,
                )
                fig.add_vline(
                    x=x_mla,
                    line_dash="dash",
                    line_color="rgba(255,140,0,0.9)",
                    row=r,
                    col=1,
                )

    # ---- 轴标题 ----
    fig.update_xaxes(title_text="x (cm)", row=1, col=1)
    fig.update_yaxes(title_text="A (cm²)", row=1, col=1)
    fig.update_xaxes(title_text="x (cm)", row=2, col=1)
    fig.update_yaxes(title_text="FFR", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=1, col=2)
    fig.update_yaxes(title_text="Q (cm³/s)", row=1, col=2)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_yaxes(title_text="p (mmHg)", row=2, col=2)
    fig.update_xaxes(title_text="t (s)", row=3, col=2)
    fig.update_yaxes(title_text="A (cm²)", row=3, col=2)

    # ---- 拖动条动作：restyle（不要用 frames/animate） ----
    def _restyle_steps(idx: list[int], labels: list[str], every: int, payload_of):
        steps = []
        for k, i in enumerate(idx):
            upd = payload_of(i)
            steps.append(
                dict(
                    method="restyle",
                    args=[upd["data"], upd["traces"]],
                    label=_label(labels[k], k, len(idx), every),
                    # value 用步序号 k，与 slider.active 语义一致
                    value=k,
                )
            )
        return steps

    def _t_payload(i: int) -> dict:
        return {
            "traces": [0],
            "data": {
                "y": [area[i].tolist()],
                "name": [t_name(i)],
                "hovertemplate": [
                    f"x=%{{x:.3f}} cm<br>A @ t={tt[i]:.3f} s"
                    " = %{y:.5f} cm²<extra></extra>"
                ],
            },
        }

    def _x_payload(i: int) -> dict:
        return {
            "traces": [3, 4, 5],
            "data": {
                "y": [
                    flow[t_sel, i].tolist(),
                    pres[t_sel, i].tolist(),
                    area[t_sel, i].tolist(),
                ],
                "name": [x_name("Q", i), x_name("p", i), x_name("A", i)],
                "hovertemplate": [
                    f"t=%{{x:.4f}} s<br>Q @ x={xx[i]:.3f} cm"
                    " = %{y:.4f} cm³/s<extra></extra>",
                    f"t=%{{x:.4f}} s<br>p @ x={xx[i]:.3f} cm"
                    " = %{y:.3f} mmHg<extra></extra>",
                    f"t=%{{x:.4f}} s<br>A @ x={xx[i]:.3f} cm"
                    " = %{y:.5f} cm²<extra></extra>",
                ],
            },
        }

    n_lab = 6
    t_labels = [f"{tt[i]:.2f}" for i in t_idx]
    x_labels = [f"{xx[i]:.2f}" for i in x_idx]

    # ---- 压缩 y 域，为标题/图例/拖动条腾出画布内空间 ----
    Y_LO, Y_HI = 0.17, 0.95
    _rescale_y_domains(fig, Y_LO, Y_HI)

    def _dom(row: int, col: int) -> tuple[tuple, tuple]:
        sp = fig.get_subplot(row, col)
        xd = tuple(getattr(fig.layout, sp.xaxis.plotly_name).domain)
        yd = tuple(getattr(fig.layout, sp.yaxis.plotly_name).domain)
        return xd, yd

    xd11, yd11 = _dom(1, 1)  # ① A(x)
    xd21, yd21 = _dom(2, 1)  # ② FFR
    xd12, yd12 = _dom(1, 2)  # ③ Q(t)
    xd22, yd22 = _dom(2, 2)  # ④ p(t)
    xd32, yd32 = _dom(3, 2)  # ⑤ A(t)

    row_h = yd11[1] - yd11[0]
    # 拖动条到所属子图下沿的距离：留出轴标题+刻度，约半行高
    slider_gap = row_h * 0.62

    fig.update_layout(
        height=1320,
        width=1340,
        title_text=(
            f"Navier-Stokes 1D 结果（{title}）" if title else "Navier-Stokes 1D 结果"
        ),
        template="plotly_white",
        showlegend=True,
        margin=dict(l=140, r=50, t=150, b=90),
        sliders=[
            dict(
                name="时间 t",
                active=t_pos[i_t],
                steps=_restyle_steps(
                    t_idx, t_labels, max(1, len(t_idx) // n_lab), _t_payload
                ),
                x=xd11[0],
                len=xd11[1] - xd11[0],
                y=yd11[0] - slider_gap,
                pad=dict(t=6, b=0),
                ticklen=0,
                minorticklen=0,
                bgcolor="rgba(31,119,180,0.10)",
                bordercolor="rgba(31,119,180,0.45)",
                font=dict(size=10),
            ),
            dict(
                name="空间 x",
                active=x_pos[i_x],
                steps=_restyle_steps(
                    x_idx, x_labels, max(1, len(x_idx) // n_lab), _x_payload
                ),
                x=xd32[0],
                len=xd32[1] - xd32[0],
                y=yd32[0] - slider_gap,
                pad=dict(t=6, b=0),
                ticklen=0,
                minorticklen=0,
                bgcolor="rgba(214,39,40,0.10)",
                bordercolor="rgba(214,39,40,0.45)",
                font=dict(size=10),
            ),
        ],
    )

    # ---- 子图标题：子图上方左对齐 ----
    _titles = (
        ("① 横截面积 A 沿程分布", 1, 1),
        ("② FFR 沿程分布（时间已积分）", 2, 1),
        ("③ 流量 Q 时程", 1, 2),
        ("④ 压力 p 时程", 2, 2),
        ("⑤ 横截面积 A 时程", 3, 2),
    )
    for text, row, col in _titles:
        xd, yd = _dom(row, col)
        fig.add_annotation(
            text=text,
            xref="paper",
            yref="paper",
            x=xd[0] + 0.001,
            y=yd[1] + 0.007,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=13),
        )

    # ---- 每图独立图例：子图外部上方、右对齐（与标题同一行） ----
    _legend_slots = (
        ("legend", xd11, yd11),   # ① A(x)
        ("legend2", xd21, yd21),  # ② FFR(x)
        ("legend3", xd12, yd12),  # ③ Q(t)
        ("legend4", xd22, yd22),  # ④ p(t)
        ("legend5", xd32, yd32),  # ⑤ A(t)
    )
    for lname, xd, yd in _legend_slots:
        fig.layout[lname] = dict(
            x=xd[1],
            y=yd[1] + 0.007,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=10),
            tracegroupgap=0,
            itemsizing="constant",
        )

    # ---- 拖动条名称：轨道左端外侧，不额外占竖直空间 ----
    fig.add_annotation(
        text="<b>时间 t (s)</b>",
        xref="paper",
        yref="paper",
        x=xd11[0] - 0.006,
        y=yd11[0] - slider_gap,
        xanchor="right",
        yanchor="middle",
        showarrow=False,
        font=dict(size=10, color="#1f77b4"),
    )
    fig.add_annotation(
        text="<b>空间 x (cm)</b>",
        xref="paper",
        yref="paper",
        x=xd32[0] - 0.006,
        y=yd32[0] - slider_gap,
        xanchor="right",
        yanchor="middle",
        showarrow=False,
        font=dict(size=10, color="#d62728"),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        if str(save_path).lower().endswith(".html"):
            fig.write_html(save_path, include_plotlyjs="cdn")
        else:
            fig.write_image(save_path, scale=2)
        print(f"saved figure: {save_path}")

    if show:
        show_kw = {"renderer": renderer} if renderer is not None else {}
        fig.show(**show_kw)
    return fig


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def example(
    target_nx: int | None = 200,
    frame_spacing_cm: float = 0.02,
    duration_s: float = 3.0,
    name: str = "stenoses",
    use_virtual_stent: bool = False,
    force_rerun: bool = False,
    plot_only: bool = False,
):
    """下游 joblib 数据示例；``target_nx`` 控制轴向降采样（None=不降采样）。

    ``solver`` 会缓存到 ``data/cache/solver/<name>_nx<nx>_T<时长>[_stent].pkl``。
    缓存存在时默认直接复用，跳过标定（省数分钟）。

    Parameters
    ----------
    force_rerun :
        忽略已有缓存，重跑标定并覆盖缓存
    plot_only :
        只允许读缓存；缓存缺失则报错，不启动标定
    """
    import joblib

    np.random.seed(2034)
    mm_per_pixel = 9.7 / 756

    dirname = os.path.join(_REPO_ROOT, "data", "stenoses_data")
    # 原始 npy：面积为 pixel²，直径为 pixel，读入后一律换成 cm / cm²
    area_file = joblib.load(os.path.join(dirname, "areas.joblib"))
    area = np.array(area_file)[::-1] * (mm_per_pixel**2) / 100.0  # mm² → cm²
    nx = area.shape[0]
    print(f"raw nx={nx}")

    # 分支信息（与 area 同向：近端→远端）
    branch_groups = joblib.load(os.path.join(dirname, "branches.joblib"))
    branch_d = np.array(joblib.load(os.path.join(dirname, "branch_diameters.joblib")))
    branch_d = branch_d[::-1] * mm_per_pixel / 10.0  # mm → cm
    branch_idx = [nx - idxs[0] for idxs in branch_groups][::-1]

    # 轴向重采样：降低 nx 以加速；物理长度保持 frame_spacing_cm * nx_raw
    area, length, nx, branch_idx, branch_d = resample_axial_profile(
        area,
        target_nx=target_nx,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
        frame_spacing_cm=frame_spacing_cm,
    )
    print(nx, length, f"n_branch={len(branch_idx)}")

    # ---- 缓存命中则直接复用 solver ----
    cache_path = solver_cache_path(name, nx, duration_s, use_virtual_stent)
    if os.path.exists(cache_path) and not force_rerun:
        print(f"reuse cached solver: {cache_path}")
        bundle = load_solver_bundle(cache_path)
        solver = bundle["solver"]
        cal = bundle["cal"]
    else:
        if plot_only:
            raise FileNotFoundError(
                f"--plot-only 需要已有缓存，但未找到：{cache_path}\n"
                "请先不带 --plot-only 运行一次以生成缓存。"
            )
        parameters = BloodFlowParameters()
        cal_cache_dir = os.path.join(
            _REPO_ROOT,
            "data",
            "cache",
            f"q_scale_cal_{name}_nx{nx}_T{duration_s:.2f}"
            + ("_with_virtual_stent" if use_virtual_stent else ""),
        )
        solver, cal = calibrate_q_scale_to_inlet_peak(
            area,
            length,
            parameters,
            branch_frame_indices=branch_idx,
            branch_diameters_cm=branch_d,
            duration_s=duration_s,
            record_interval_steps=30,
            peak_lo_mmhg=95.0,
            peak_hi_mmhg=125.0,
            max_iters=10,
            verbose=True,
            cache_dir=cal_cache_dir,
            virtual_stent=use_virtual_stent,
        )
        bundle = {
            "solver": solver,
            "cal": cal,
            "meta": {
                "name": name,
                "nx": nx,
                "length_cm": length,
                "duration_s": duration_s,
                "use_virtual_stent": use_virtual_stent,
                "frame_spacing_cm": frame_spacing_cm,
                "target_nx": target_nx,
                "branch_idx": branch_idx,
                "branch_d": branch_d,
            },
        }
        saved = save_solver_bundle(bundle, cache_path)
        print(f"cached solver → {saved}")

    print(
        f"calibrated: converged={cal['converged']}, "
        f"q_scale={cal['q_scale']:.4f}, "
        f"p(0)_peak={cal['p_in_peak']:.2f} mmHg, Q_mean={cal['q_mean']:.3f} mL/s"
    )

    html_path = os.path.join(
        _REPO_ROOT,
        "data",
        "cache",
        f"final_{name}_nx{nx}_qs{cal['q_scale']:.3f}.html",
    )
    plot_results(
        solver,
        title=f"{name}_nx{nx}_qs{cal['q_scale']:.3f}",
        show=False,
        save_path=html_path,
    )
    print(f"saved final plotly: {html_path}")

    assert solver.history_time is not None
    assert solver.history_flow is not None
    assert solver.history_pressure is not None

    t_hist = solver.history_time
    q_in = integral_mean(solver.history_flow[:, 0], t_hist)
    q_out = integral_mean(solver.history_flow[:, -1], t_hist)
    q_side = (
        float(integral_mean(np.sum(solver.history_branch_flow, axis=1), t_hist))
        if solver.history_branch_flow is not None and solver.history_branch_flow.size
        else 0.0
    )
    print(f"Q_in={q_in:.3f}, Q_out={q_out:.3f}, Q_side_sum={q_side:.3f}")
    print(f"mass check: Q_in ≈ Q_out+Q_side → {q_in:.3f} vs {q_out + q_side:.3f}")

    ffr = solver.ffr_ratio()
    return ffr


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    force_rerun = "--rerun" in argv
    plot_only = "--plot-only" in argv

    ffr = example(force_rerun=force_rerun, plot_only=plot_only)
    print(f"max ffr: {np.max(ffr)}\n min ffr: {np.min(ffr)}")


if __name__ == "__main__":
    main()
