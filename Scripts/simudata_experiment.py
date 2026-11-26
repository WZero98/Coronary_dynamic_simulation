"""
simudata_experiment.py — 模拟数据批量仿真

用 :class:`SimuCoronary.simudata.SimulatedVessel` 生成 7 种典型几何，
逐个跑 1D 求解器（含 q_scale 外环标定），结果写到
``results/模拟数据结果/``。

并行：进程池（默认 4 进程）。每例独立进程，互不影响。

运行::

    python Scripts/simudata_experiment.py                # 默认 4 进程
    python Scripts/simudata_experiment.py --workers 2    # 指定并行度
    python Scripts/simudata_experiment.py --dry-run      # 只打印算例配置
    python Scripts/simudata_experiment.py --only uniform # 只跑名字含该子串的算例

复用缓存（不必重跑仿真）::

    python Scripts/simudata_experiment.py --list-cached          # 列出已缓存算例
    python Scripts/simudata_experiment.py --replot               # 全部重画
    python Scripts/simudata_experiment.py --replot stenosis_center  # 只重画一例

输出（``results/模拟数据结果/``）::

    summary.json                    全部算例的关键指标
    <case>/geometry.png             面积剖面（原始 + 平滑后）
    <case>/final_<case>.html        Plotly 五图交互页
    <case>/solver.pkl               该例的 solver 缓存（可复现、可重画）
    compare_ffr.png                 FFR(x) 七例对比
    compare_area.png                面积剖面七例对比
    compare_pressure.png            时间平均压力七例对比
    compare_summary.png             关键指标柱状图

读取缓存做二次分析::

    from simudata_experiment import load_case_solver

    bundle = load_case_solver("stenosis_center")
    solver = bundle["solver"]      # 已 run() 的 NavierStokes1D
    ffr = solver.ffr_ratio()
    print(bundle["meta"]["vessel_kwargs"])   # 复现用的几何参数
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# 本脚本位于 <root>/Scripts/，仓库根 = 上一级
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from SimuCoronary.calculators import calibrate_q_scale_to_inlet_peak  # noqa: E402
from SimuCoronary.geometry import build_stenosis_segment, smooth_axial_area, resample_axial_profile  # noqa: E402
from SimuCoronary.navier_stokes_1d import (  # noqa: E402
    BloodFlowParameters,
    NavierStokes1D,
    integral_mean,
)
from SimuCoronary.simudata import SimulatedVessel  # noqa: E402
from SimuCoronary.utils import (  # noqa: E402
    load_solver_bundle,
    plot_results,
    save_solver_bundle,
)

_OUT_ROOT = os.path.join(_REPO_ROOT, "results", "模拟数据实验")

# 每个算例目录下的缓存文件名；调用方可用 load_case_solver 读取
SOLVER_FILENAME = "solver.pkl"

# ---------------------------------------------------------------------------
# 公共算例参数（与脚本原有的 7 例一致）
# ---------------------------------------------------------------------------
LENGTH_CM = 10.0
DX_TARGET_CM = 0.02
N_CELLS = int(round(LENGTH_CM / DX_TARGET_CM)) + 1  # 501，dx ≈ 0.02 cm

AREA_SCALER = 1.0
PROX_AREA = 0.15 * AREA_SCALER
DIST_AREA = 0.15 * AREA_SCALER
STENOSIS_RATE = 0.5
STENOSIS_POSITION = 0.5
STENOSIS_FRACTION = 0.2
BRANCH_POSITIONS = [0.25, 0.5, 0.75]
BRANCH_DIAMETERS = [0.15 * np.sqrt(AREA_SCALER)] * 3

DURATION_S = 3.0
RECORD_INTERVAL_STEPS = 30
PEAK_LO_MMHG = 95.0
PEAK_HI_MMHG = 125.0
MAX_ITERS = 10


@dataclass(frozen=True)
class CaseSpec:
    """一个算例的定义：名称 + SimulatedVessel 构造参数。"""

    key: str
    label: str
    vessel_kwargs: dict[str, Any] = field(default_factory=dict)


def build_cases(
    area_scaler: float = AREA_SCALER,
    stenosis_rate: float = STENOSIS_RATE,
    only_stenosis: bool = False,
) -> list[CaseSpec]:
    """7 个典型几何（``only_stenosis=True`` 时只返回 4 个含狭窄的）。

    Parameters
    ----------
    area_scaler :
        面积缩放系数。``prox_area = 0.15 * area_scaler``，
        ``dist_area = 0.15 * area_scaler``。默认 1.0（本脚本的主算例集）。
        其他脚本（如面积敏感性研究）可传入不同值复用同一套几何定义。
    stenosis_rate :
        狭窄率，仅影响含狭窄的 4 个算例；对 ``uniform`` / ``taper`` /
        ``taper_reverse`` 无影响。取值须 ``[0, 1)``。
    only_stenosis :
        只返回含狭窄的 4 个算例（``stenosis_center`` / ``taper_stenosis`` /
        ``stenosis_proximal`` / ``stenosis_distal``），供狭窄率敏感性研究使用。
    """
    prox_area = 0.15 * float(area_scaler)
    dist_area = 0.15 * float(area_scaler)
    rate = float(stenosis_rate)
    common_branch = dict(
        branch_positions=BRANCH_POSITIONS,
        branch_diameters=BRANCH_DIAMETERS,
    )
    base = dict(
        vessel_length_cm=LENGTH_CM,
        n_cells=N_CELLS,
        prox_position=0.0,
        dist_position=1.0,
        **common_branch,
    )
    non_stenosis = [
        CaseSpec(
            key="uniform",
            label="① 均匀直管",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area,
            ),
        ),
        CaseSpec(
            key="taper",
            label="② 锥形血管（近粗远细）",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area * 0.6,
            ),
        ),
        CaseSpec(
            key="taper_reverse",
            label="⑤ 反向锥形（近细远粗）",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area * 0.6,
                dist_area_cm2=dist_area,
            ),
        ),
    ]
    stenosis = [
        CaseSpec(
            key="stenosis_center",
            label="③ 中间狭窄，等截面",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area,
                stenosis_ratio=rate,
                stenosis_position=STENOSIS_POSITION,
                stenosis_fraction=STENOSIS_FRACTION,
            ),
        ),
        CaseSpec(
            key="taper_stenosis",
            label="④ 锥形 + 中间狭窄",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area * 0.6,
                stenosis_ratio=rate,
                stenosis_position=STENOSIS_POSITION,
                stenosis_fraction=STENOSIS_FRACTION,
            ),
        ),
        CaseSpec(
            key="stenosis_proximal",
            label="⑥ 狭窄位于近端",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area,
                stenosis_ratio=rate,
                stenosis_position=0.25,
                stenosis_fraction=STENOSIS_FRACTION,
            ),
        ),
        CaseSpec(
            key="stenosis_distal",
            label="⑦ 狭窄位于远端",
            vessel_kwargs=dict(
                **base,
                prox_area_cm2=prox_area,
                dist_area_cm2=dist_area,
                stenosis_ratio=rate,
                stenosis_position=0.75,
                stenosis_fraction=STENOSIS_FRACTION,
            ),
        ),
    ]
    if only_stenosis:
        return stenosis
    # 保持原有顺序：uniform / taper / stenosis_center / taper_stenosis /
    # taper_reverse / stenosis_proximal / stenosis_distal
    return [
        non_stenosis[0],
        non_stenosis[1],
        stenosis[0],
        stenosis[1],
        non_stenosis[2],
        stenosis[2],
        stenosis[3],
    ]


# ---------------------------------------------------------------------------
# 单例几何图（matplotlib，Agg 后端）
# ---------------------------------------------------------------------------
def _save_geometry_figure(
    vessel: SimulatedVessel,
    path: str,
    label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    x = vessel.x_cm
    raw = vessel.generate_area()
    sten = build_stenosis_segment(
        raw, frame_spacing_cm=vessel.dx_cm, max_ref_distance_cm=LENGTH_CM
    )
    smoothed = smooth_axial_area(raw, sten)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, raw, "C0-", lw=2, label="原始（参数化）")
    ax.plot(x, smoothed, "C1--", lw=1.8, label="平滑后（求解器内部使用）")
    ax.axvspan(
        x[sten.prox_ref_idx],
        x[sten.dist_ref_idx],
        color="orange",
        alpha=0.12,
        label="检测到的狭窄段",
    )
    ax.axvline(
        x[sten.mla_idx], color="darkorange", ls=":", lw=1.5, label="MLA"
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("A (cm²)")
    ax.set_title(f"{label} — 面积剖面")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 单个算例的完整流程（在子进程中执行）
# ---------------------------------------------------------------------------
def run_one_case(
    spec: CaseSpec,
    out_root: str,
    case_dir_name: str | None = None,
    extra_meta: dict | None = None,
) -> dict:
    """构造几何 → 标定 → 出图 → 缓存 → 返回指标。

    异常时返回 ``ok=False`` 而不抛出，避免单例失败拖垮整批。

    Parameters
    ----------
    spec :
        算例定义
    out_root :
        结果根目录
    case_dir_name :
        算子目录名。``None`` 则用 ``spec.key``；参数扫描时可传
        ``f"{scaler}/{spec.key}"`` 之类，从而按参数分组存放。
    extra_meta :
        额外写入 bundle 的 ``meta``（如面积缩放系数），便于日后追溯。
    """
    t_start = time.perf_counter()
    dir_name = case_dir_name if case_dir_name is not None else spec.key
    case_dir = os.path.join(out_root, dir_name)
    os.makedirs(case_dir, exist_ok=True)
    tag = dir_name.replace(os.sep, "/")

    try:
        vessel = SimulatedVessel(**spec.vessel_kwargs)
        area = vessel.generate_area()
        branch_idx, branch_dia = vessel.generate_branch()

        # 此处添加重采样：resample_axial_profile，将节点数重采样为统一的200个节点
        area, dx_cm, n_cells, branch_idx, branch_dia = resample_axial_profile(
            area,
            target_nx=200,
            branch_frame_indices=branch_idx,
            branch_diameters_cm=branch_dia,
            frame_spacing_cm=vessel.dx_cm,
        )

        # 几何图（原始 + 平滑）
        _save_geometry_figure(
            vessel, os.path.join(case_dir, "geometry.png"), spec.label
        )

        # 标定：virtual_stent=False，保留参数化几何本身，不被支架改写
        solver, cal = calibrate_q_scale_to_inlet_peak(
            area,
            vessel.vessel_length_cm,
            BloodFlowParameters(),
            branch_frame_indices=branch_idx,
            branch_diameters_cm=branch_dia,
            duration_s=DURATION_S,
            record_interval_steps=RECORD_INTERVAL_STEPS,
            peak_lo_mmhg=PEAK_LO_MMHG,
            peak_hi_mmhg=PEAK_HI_MMHG,
            max_iters=MAX_ITERS,
            verbose=False,
            cache_dir=None,       # 不落逐轮 npz（每轮 16 MB）
            virtual_stent=False,  # 关键：不用支架覆盖狭窄
        )

        # 五图交互页
        html_path = os.path.join(case_dir, f"final_{spec.key}.html")
        plot_results(
            solver,
            title=f"{spec.key} (q_scale={cal['q_scale']:.3f})",
            show=False,
            save_path=html_path,
        )

        # ---- 缓存 solver，便于日后复现 / 重新出图 ----
        solver_path = save_solver_bundle(
            {
                "solver": solver,
                "cal": cal,
                "meta": {
                    "case_key": spec.key,
                    "case_label": spec.label,
                    "case_dir": tag,
                    "vessel_kwargs": dict(spec.vessel_kwargs),
                    "settings": {
                        "duration_s": DURATION_S,
                        "record_interval_steps": RECORD_INTERVAL_STEPS,
                        "peak_lo_mmhg": PEAK_LO_MMHG,
                        "peak_hi_mmhg": PEAK_HI_MMHG,
                        "max_iters": MAX_ITERS,
                        "virtual_stent": False,
                        "branch_positions": BRANCH_POSITIONS,
                        "branch_diameters_cm": BRANCH_DIAMETERS,
                    },
                    **(extra_meta or {}),
                },
            },
            os.path.join(case_dir, SOLVER_FILENAME),
        )

        # ---- 指标 ----
        ffr = np.asarray(solver.ffr_ratio(), dtype=float)
        tt = np.asarray(solver.history_time, dtype=float)
        i0 = int(len(tt) * 0.5)
        p_mean = np.asarray(
            [
                integral_mean(solver.history_pressure[i0:, i], tt[i0:])
                for i in range(solver.nx)
            ]
        )
        pe = solver.pressure_extrema_mmhg()
        mb = solver.mass_balance()

        result = {
            "ok": True,
            "key": spec.key,
            "label": spec.label,
            "case_dir": tag,
            "n_cells": int(vessel.n_cells),
            "length_cm": float(vessel.vessel_length_cm),
            "dx_cm": float(vessel.dx_cm),
            "n_branch": int(branch_idx.size),
            "has_stenosis": bool(vessel.has_stenosis),
            "mla_area_cm2": float(vessel.mla_area_cm2),
            "area_min_cm2": float(area.min()),
            "area_max_cm2": float(area.max()),
            "q_scale": float(cal["q_scale"]),
            "converged": bool(cal["converged"]),
            "n_iters": int(cal["n_iters"]),
            "p_in_peak_mmhg": float(cal["p_in_peak"]),
            "q_mean_ml_s": float(cal["q_mean"]),
            "p_in_systolic": float(pe["p_in_systolic"]),
            "p_in_diastolic": float(pe["p_in_diastolic"]),
            "p_out_systolic": float(pe["p_out_systolic"]),
            "p_out_diastolic": float(pe["p_out_diastolic"]),
            "ffr_min": float(ffr.min()),
            "ffr_distal": float(ffr[-1]),
            "mass_err": float(mb["mass_err"]),
            "mass_residual": float(mb["residual"]),
            "mass_q_in": float(mb["q_in"]),
            "mass_q_out": float(mb["q_out"]),
            "mass_q_side": float(mb["q_side"]),
            "mass_dvdt": float(mb["dvdt"]),
            "n_saves": int(len(tt)),
            "html": os.path.relpath(html_path, out_root),
            "solver_cache": os.path.relpath(solver_path, out_root),
            "wall_s": float(time.perf_counter() - t_start),
            # 供汇总图使用（不写入 JSON 的部分会被过滤）
            "_x_cm": vessel.x_cm.tolist(),
            "_ffr": ffr.tolist(),
            "_area": area.tolist(),
            "_p_mean": p_mean.tolist(),
        }
        return result

    except Exception as exc:  # noqa: BLE001 - 单例失败不应拖垮整批
        return {
            "ok": False,
            "key": spec.key,
            "label": spec.label,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - t_start),
        }


# ---------------------------------------------------------------------------
# 读取缓存：复现 / 重新出图 / 二次分析
# ---------------------------------------------------------------------------
def case_solver_path(case_key: str, out_root: str = _OUT_ROOT) -> str:
    """某算例的 solver 缓存路径。"""
    return os.path.join(out_root, case_key, SOLVER_FILENAME)


def list_cached_cases(out_root: str = _OUT_ROOT) -> dict[str, str]:
    """列出所有已缓存的算例。

    Returns
    -------
    dict
        ``{case_key: solver.pkl 绝对路径}``，按 key 排序；无缓存时为空字典
    """
    if not os.path.isdir(out_root):
        return {}
    found: dict[str, str] = {}
    for name in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, name, SOLVER_FILENAME)
        if os.path.isfile(p):
            found[name] = os.path.abspath(p)
    return found


def load_case_solver(
    case_key: str | None = None,
    out_root: str = _OUT_ROOT,
) -> dict:
    """读取某个算例的缓存，返回可直接用于后续分析的全部对象。

    Parameters
    ----------
    case_key :
        算例名（如 ``"stenosis_center"``）。传 ``None`` 则要求缓存中恰好
        只有一个算例，否则报错并列出可用 key。
    out_root :
        结果根目录，默认 ``results/模拟数据结果``。

    Returns
    -------
    dict
        含 ``solver``（已 ``run()`` 的 ``NavierStokes1D``）、``cal``（标定
        report）、``meta``（算例参数与仿真设置）、``path``（缓存文件路径）。

    Examples
    --------
    载入后重新出图（不用重跑标定）::

        from simudata_experiment import load_case_solver
        from SimuCoronary.utils import plot_results

        bundle = load_case_solver("stenosis_center")
        solver = bundle["solver"]
        plot_results(solver, show=False, save_path="replot.html")

    二次分析::

        ffr = solver.ffr_ratio()
        p_p50 = solver.history_pressure[len(solver.history_time) // 2:, :]

    Raises
    ------
    FileNotFoundError
        缓存不存在（提示先跑批处理）
    ValueError
        ``case_key=None`` 但缓存中有多个算例
    """
    available = list_cached_cases(out_root)

    if case_key is None:
        if not available:
            raise FileNotFoundError(
                f"{out_root} 下没有任何 solver 缓存。"
                "请先运行 `python Scripts/simudata_experiment.py`。"
            )
        if len(available) > 1:
            raise ValueError(
                "缓存中存在多个算例，请显式指定 case_key。可用：\n  "
                + "\n  ".join(available)
            )
        case_key = next(iter(available))

    if case_key not in available:
        hint = (
            "可用算例：\n  " + "\n  ".join(available)
            if available
            else f"{out_root} 下没有任何缓存，请先运行批处理脚本。"
        )
        raise FileNotFoundError(f"算例 {case_key!r} 没有缓存。{hint}")

    bundle = load_solver_bundle(available[case_key])
    bundle["path"] = available[case_key]
    return bundle


def replot_from_cache(
    case_key: str | None = None,
    out_root: str = _OUT_ROOT,
    save_path: str | None = None,
    show: bool = False,
):
    """从缓存重新出五图交互页，不重跑仿真。

    等价于 ``plot_results(load_case_solver(...)["solver"], ...)``，
    默认写回该算例目录下的 ``replot_<key>.html``。
    """
    bundle = load_case_solver(case_key, out_root)
    solver = bundle["solver"]
    key = bundle["meta"].get("case_key", case_key or "case")

    if save_path is None:
        save_path = os.path.join(out_root, key, f"replot_{key}.html")

    q = bundle.get("cal", {}).get("q_scale")
    title = f"{key}" + (f" (q_scale={q:.3f})" if isinstance(q, float) else "")
    return plot_results(solver, title=title, show=show, save_path=save_path)


# ---------------------------------------------------------------------------
# 汇总对比图
# ---------------------------------------------------------------------------
def save_comparison_figures(results: Sequence[dict], out_root: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    ok = [r for r in results if r.get("ok")]
    if not ok:
        return []

    ok.sort(key=lambda r: r["key"])
    written: list[str] = []
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def _new() -> tuple:
        return plt.subplots(figsize=(9.5, 5.2))

    # ---- 1. FFR(x) ----
    fig, ax = _new()
    for i, r in enumerate(ok):
        ax.plot(r["_x_cm"], r["_ffr"], lw=1.9, color=colors[i % 10],
                label=f'{r["key"]}  (min={r["ffr_min"]:.3f})')
    ax.axhline(0.80, color="k", ls=":", lw=1.2, label="FFR = 0.80 阈值")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("FFR")
    ax.set_title("FFR 沿程分布（时间平均压比）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_ffr.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 2. 面积剖面 ----
    fig, ax = _new()
    for i, r in enumerate(ok):
        ax.plot(r["_x_cm"], r["_area"], lw=1.9, color=colors[i % 10],
                label=r["key"])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("A₀ (cm²)")
    ax.set_title("面积剖面（参数化原始几何）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_area.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 3. 时间平均压力 ----
    fig, ax = _new()
    for i, r in enumerate(ok):
        ax.plot(r["_x_cm"], r["_p_mean"], lw=1.9, color=colors[i % 10],
                label=r["key"])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("p̄ (mmHg)")
    ax.set_title("时间平均管腔压沿程分布")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_pressure.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 4. 关键指标柱状图 ----
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    keys = [r["key"] for r in ok]
    pos = np.arange(len(ok))

    panels = (
        ("ffr_min", "最小 FFR", axes[0, 0]),
        ("p_in_peak_mmhg", "入口峰值压 (mmHg)", axes[0, 1]),
        ("q_scale", "标定得到的 q_scale", axes[1, 0]),
        ("wall_s", "墙钟耗时 (s)", axes[1, 1]),
    )
    for field_name, title, ax in panels:
        vals = [r[field_name] for r in ok]
        bars = ax.bar(pos, vals, color=[colors[i % 10] for i in range(len(ok))])
        ax.set_xticks(pos)
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    if "ffr_min" in ok[0]:
        axes[0, 0].axhline(0.80, color="k", ls=":", lw=1.2)
    fig.suptitle("模拟数据算例关键指标", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_summary.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    return written


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _json_safe(result: dict) -> dict:
    """去掉以下划线开头的私有绘图数据，只留指标。"""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="模拟数据批量 1D 仿真（并行）"
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="并行进程数（默认 min(4, cpu_count)）",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="只跑 key 或 label 含该子串的算例",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印算例配置，不跑仿真",
    )
    parser.add_argument(
        "--out", type=str, default=_OUT_ROOT, help="输出根目录",
    )
    parser.add_argument(
        "--list-cached", action="store_true",
        help="列出已缓存的算例后退出（不跑仿真）",
    )
    parser.add_argument(
        "--replot", type=str, nargs="?", const="__ALL__", default=None,
        metavar="CASE",
        help="从缓存重新出图后退出：给 CASE 只重画该例；不带参数则重画全部",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ---- 子命令：列出 / 重画 ----
    if args.list_cached:
        cached = list_cached_cases(args.out)
        if not cached:
            print(f"{args.out} 下没有 solver 缓存。")
            print("请先运行（去掉 --list-cached）：")
            print("  python Scripts/simudata_experiment.py")
            return
        print(f"已缓存 {len(cached)} 个算例（{args.out}）：")
        for key, path in cached.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {key:22} {size_mb:7.1f} MB  {path}")
        return

    if args.replot is not None:
        cached = list_cached_cases(args.out)
        if not cached:
            raise SystemExit(
                f"{args.out} 下没有 solver 缓存，无法重画。"
                "请先运行 `python Scripts/simudata_experiment.py`。"
            )
        targets = (
            sorted(cached) if args.replot == "__ALL__" else [args.replot]
        )
        print(f"从缓存重画 {len(targets)} 个算例…")
        for key in targets:
            try:
                fig = replot_from_cache(key, args.out)
                print(f"  [OK] {key}  ({len(fig.data)} traces)")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {key}: {type(exc).__name__}: {exc}")
        return

    cases = build_cases()
    if args.only:
        needle = args.only.lower()
        cases = [
            c for c in cases
            if needle in c.key.lower() or needle in c.label.lower()
        ]
        if not cases:
            raise SystemExit(f"--only {args.only!r} 没有匹配的算例")

    print(f"共 {len(cases)} 个算例；输出目录：{args.out}")
    print(f"网格：L={LENGTH_CM} cm, n_cells={N_CELLS}, "
          f"dx={LENGTH_CM / (N_CELLS - 1):.5f} cm")
    print(f"标定：duration={DURATION_S}s, peak∈({PEAK_LO_MMHG}, "
          f"{PEAK_HI_MMHG}) mmHg, max_iters={MAX_ITERS}, virtual_stent=False")
    print(f"并行：{args.workers} 进程")
    print()
    for c in cases:
        print(f"  {c.key:22} {c.label}")
    print()

    if args.dry_run:
        print("--dry-run：未执行仿真")
        return

    os.makedirs(args.out, exist_ok=True)
    t_all = time.perf_counter()
    results: list[dict] = []

    if args.workers <= 1:
        for c in cases:
            print(f"[start] {c.key}", flush=True)
            r = run_one_case(c, args.out)
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{status}] {c.key}  {r['wall_s']:.0f}s"
                  + ("" if r["ok"] else f"  {r['error']}"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_case, c, args.out): c for c in cases
            }
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001 - 进程级失败
                    r = {
                        "ok": False,
                        "key": c.key,
                        "label": c.label,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "wall_s": float("nan"),
                    }
                results.append(r)
                status = "OK" if r["ok"] else "FAIL"
                print(f"[{status}] {r['key']}  {r['wall_s']:.0f}s"
                      + ("" if r["ok"] else f"  {r['error']}"), flush=True)

    elapsed = time.perf_counter() - t_all

    # ---- 汇总 ----
    results.sort(key=lambda r: r["key"])
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    summary = {
        "n_cases": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "wall_s_total": round(elapsed, 1),
        "workers": args.workers,
        "settings": {
            "length_cm": LENGTH_CM,
            "n_cells": N_CELLS,
            "dx_cm": LENGTH_CM / (N_CELLS - 1),
            "duration_s": DURATION_S,
            "record_interval_steps": RECORD_INTERVAL_STEPS,
            "peak_lo_mmhg": PEAK_LO_MMHG,
            "peak_hi_mmhg": PEAK_HI_MMHG,
            "max_iters": MAX_ITERS,
            "virtual_stent": False,
            "branch_positions": BRANCH_POSITIONS,
            "branch_diameters_cm": BRANCH_DIAMETERS,
        },
        "cases": [_json_safe(r) for r in results],
    }

    fig_paths = save_comparison_figures(results, args.out) if ok else []
    summary["comparison_figures"] = [
        os.path.relpath(p, args.out) for p in fig_paths
    ]

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print(f"完成：{len(ok)}/{len(results)} 成功，总耗时 {elapsed:.0f}s")
    if ok:
        print(f"{'case':22} {'q_scale':>8} {'p_in_peak':>10} "
              f"{'ffr_min':>9} {'n_iter':>7} {'wall(s)':>9}")
        for r in ok:
            print(f"{r['key']:22} {r['q_scale']:8.4f} "
                  f"{r['p_in_peak_mmhg']:10.2f} {r['ffr_min']:9.4f} "
                  f"{r['n_iters']:7d} {r['wall_s']:9.0f}")
    if failed:
        print()
        print("失败算例：")
        for r in failed:
            print(f"  {r['key']}: {r['error']}")
    print()
    print(f"summary: {summary_path}")
    for p in fig_paths:
        print(f"figure : {p}")


if __name__ == "__main__":
    main()
