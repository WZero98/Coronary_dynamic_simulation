"""
experiment.py — 可直接运行的算例脚本

在仓库根目录把 ``SimuCoronary`` 加入 ``sys.path``，读 ``data/stenoses_data``，
调用 ``SimuCoronary.calculators.calibrate_q_scale_to_inlet_peak`` 把入口峰值压
收到目标窗，再绘图、做质量校验并返回 FFR。

运行::

    python Scripts/experiment.py

也支持::

    cd Scripts && python experiment.py

所有数据读写路径都按**仓库根目录**解析，与当前工作目录无关。
"""

from __future__ import annotations

import os
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
    integral_mean,
)


def example(
    target_nx: int | None = 100,
    frame_spacing_cm: float = 0.02,
    duration_s: float = 3.0,
    name: str = 'Null',
    use_virtual_stent: bool = True,
):
    """下游 joblib 数据示例；``target_nx`` 控制轴向降采样（None=不降采样）。

    先用 ``calibrate_q_scale_to_inlet_peak`` 把入口峰值压收到目标窗，
    再在该 ``q_scale`` 下绘图并返回 FFR。

    ``data/stenoses_data`` 与 ``data/cache`` 均按**仓库根目录**解析，
    因此在任意工作目录下调用均可。
    """
    import joblib

    np.random.seed(2034)
    mm_per_pixel = (9.7 / 756)

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

    # 只选择目标区域
    # roi_idx = list(range(10, 230))
    # area_smooth, branch_idx, branch_d = roi_process(roi_idx, area_smooth, branch_idx, branch_d)

    # 轴向重采样：降低 nx 以加速；物理长度保持 frame_spacing_cm * nx_raw
    area, length, nx, branch_idx, branch_d = resample_axial_profile(
        area,
        target_nx=target_nx,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
        frame_spacing_cm=frame_spacing_cm,
    )
    print(nx, length, f"n_branch={len(branch_idx)}")

    parameters = BloodFlowParameters()
    suffix = "_with_virtual_stent" if use_virtual_stent else ""
    cache_dir = os.path.join(
        _REPO_ROOT,
        "data",
        "cache",
        f"q_scale_cal_{name}_nx{nx}_T{duration_s:.2f}{suffix}",
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
        cache_dir=cache_dir,
        virtual_stent=use_virtual_stent,
    )
    print(
        f"calibrated: converged={cal['converged']}, "
        f"q_scale={cal['q_scale']:.4f}, "
        f"p(0)_peak={cal['p_in_peak']:.2f} mmHg, Q_mean={cal['q_mean']:.3f} mL/s"
    )
    print(f"cache_dir={cal.get('cache_dir', cache_dir)}")

    html_path = os.path.join(
        cal.get("cache_dir", cache_dir),
        f"final_nx{nx}_qs{cal['q_scale']:.3f}.html",
    )
    solver.plot_results_1(
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


def main() -> None:
    # target_nx=200 降采样；改为 None 可跑原始分辨率对比准确率
    ffr = example(target_nx=200, name="stenoses")
    print(f"max ffr: {np.max(ffr)}\n min ffr: {np.min(ffr)}")


if __name__ == "__main__":
    main()
