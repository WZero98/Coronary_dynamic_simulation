"""
calculators.py — 面向使用者的高级接口（外环标定）

本模块存放"跨模块编排"级别的高层函数，不参与 ``NavierStokes1D`` 的数值核心。
``navier_stokes_1d.py`` 保持为纯库（求解器 + 物性参数 + 数值工具），本模块在其
之上组合 ``calibration_cache`` 完成端到端流程。

包含
----
- ``calibrate_q_scale_to_inlet_peak``  外环二分标定 ``q_scale``，使入口峰值压落入目标窗

具体算例（读 ``data/stenoses_data``、绘图、算 FFR）属于脚本层，
放在仓库根的 ``Scripts/experiment.py``，不在本包内。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from .calibration_cache import iteration_snapshot, save_calibration_cache
from .navier_stokes_1d import BloodFlowParameters, NavierStokes1D


def calibrate_q_scale_to_inlet_peak(
    area_cm2: np.ndarray | list,
    vessel_length_cm: float,
    parameters: BloodFlowParameters | None = None,
    *,
    branch_frame_indices: Optional[Sequence[int]] = None,
    branch_diameters_cm: Optional[Sequence[float]] = None,
    apply_murray_scale: bool = True,
    duration_s: float = 3.0,
    record_interval_steps: int = 20,
    peak_lo_mmhg: float = 95.0,
    peak_hi_mmhg: float = 125.0,
    q_scale_min: float = 0.05,
    q_scale_max: float = 3.0,
    max_iters: int = 10,
    verbose: bool = True,
    cache_dir: str | None = None,
    virtual_stent: bool = True,
) -> tuple["NavierStokes1D", dict]:
    """外环二分标定 ``q_scale``，使入口压力峰值落入 ``(peak_lo, peak_hi)``。

    不改 ``coronary_flow_fraction``：生成入口时用 ``fraction * q_scale``。
    假定定流量入口下峰值压随流量近似单调：峰值过高则减小 ``q_scale``，过低则增大。

    Parameters
    ----------
    virtual_stent :
        每次迭代构造求解器后是否调用 ``solver.apply_virtual_stent()``
        （按近/远端参考重构支架段管腔面积，用于虚拟 FFR）。默认开启。
    cache_dir :
        若给定，每次迭代写入 ``iter_XX.npz`` + ``summary.json``；事后读图用
        ``python -m SimuCoronary.calibration_cache <cache_dir>``。

    Returns
    -------
    solver, report
        ``report`` 含 ``q_scale``、``p_in_peak``、``converged``、``history`` 等。
    """
    if peak_hi_mmhg <= peak_lo_mmhg:
        raise ValueError("peak_hi_mmhg 须大于 peak_lo_mmhg")
    if q_scale_max <= q_scale_min:
        raise ValueError("q_scale_max 须大于 q_scale_min")
    if max_iters < 1:
        raise ValueError("max_iters 至少为 1")

    base = parameters if parameters is not None else BloodFlowParameters()
    lo = float(q_scale_min)
    hi = float(q_scale_max)
    q = float(np.clip(base.q_scale, lo, hi))
    target = 0.5 * (float(peak_lo_mmhg) + float(peak_hi_mmhg))
    history: list[dict] = []
    best_solver: NavierStokes1D | None = None
    best_loss = float("inf")
    best_rec: dict | None = None

    if verbose:
        print(
            f"calibrate q_scale: target p(0) peak in ({peak_lo_mmhg:.1f}, "
            f"{peak_hi_mmhg:.1f}) mmHg, search [{lo:.3f}, {hi:.3f}], "
            f"max_iters={max_iters}, virtual_stent={virtual_stent}"
        )

    def _finish(solver: NavierStokes1D, converged: bool) -> tuple[NavierStokes1D, dict]:
        assert best_rec is not None
        last = history[-1]
        report = {
            "converged": converged,
            "q_scale": float(last["q_scale"] if converged else best_rec["q_scale"]),
            "p_in_peak": float(last["p_in_peak"] if converged else best_rec["p_in_peak"]),
            "q_mean": float(last["q_mean"] if converged else best_rec["q_mean"]),
            "n_iters": len(history),
            "peak_lo_mmhg": float(peak_lo_mmhg),
            "peak_hi_mmhg": float(peak_hi_mmhg),
            "history": history,
        }
        if cache_dir is not None:
            path = save_calibration_cache(report, cache_dir)
            if verbose:
                print(f"  cached iterations → {path}")
        return solver, report

    for it in range(1, int(max_iters) + 1):
        paras = replace(base, q_scale=q)
        solver = NavierStokes1D(
            area_cm2,
            vessel_length_cm,
            paras,
            branch_frame_indices=branch_frame_indices,
            branch_diameters_cm=branch_diameters_cm,
            apply_murray_scale=apply_murray_scale,
        )
        if virtual_stent:
            solver.apply_virtual_stent()
        solver.run(
            duration_s=duration_s,
            record_interval_steps=record_interval_steps,
            verbose=True,
        )
        peak = float(solver.pressure_extrema_mmhg()["p_in_systolic"])
        in_range = bool(peak_lo_mmhg < peak < peak_hi_mmhg)
        # loss：距目标窗的偏差（越小越好）；窗内为 0，另加中心偏置作并列打破
        if in_range:
            loss = 0.0
        elif peak >= peak_hi_mmhg:
            loss = peak - peak_hi_mmhg
        else:
            loss = peak_lo_mmhg - peak
        loss = float(loss) + 1e-3 * abs(peak - target)

        snap = iteration_snapshot(solver, it, q, peak, in_range)
        history.append(snap)
        if verbose:
            print(
                f"  iter {it}/{max_iters}: q_scale={q:.4f}, "
                f"p(0)_peak={peak:.2f} mmHg, Q_mean={snap['q_mean']:.3f} mL/s"
                + ("  [OK]" if in_range else "")
            )

        if loss < best_loss:
            best_loss = loss
            best_solver = solver
            best_rec = snap

        if in_range:
            return _finish(solver, True)

        if peak >= peak_hi_mmhg:
            hi = q
        else:
            lo = q
        if abs(hi - lo) < 1e-9:
            break
        q = 0.5 * (lo + hi)

    assert best_solver is not None and best_rec is not None
    if verbose:
        print(
            f"  not converged in {max_iters} iters; "
            f"best q_scale={float(best_rec['q_scale']):.4f}, "
            f"p(0)_peak={float(best_rec['p_in_peak']):.2f} mmHg"
        )
    return _finish(best_solver, False)

