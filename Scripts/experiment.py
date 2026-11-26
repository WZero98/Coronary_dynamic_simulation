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
from SimuCoronary.utils import plot_results  # noqa: E402
from SimuCoronary.utils import (  # noqa: E402
    load_solver_bundle as _load_solver_bundle,
    save_solver_bundle as _save_solver_bundle,
)

_SOLVER_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "cache", "solver")


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
    """兼容旧调用；实际实现见 :func:`SimuCoronary.utils.save_solver_bundle`。"""
    return _save_solver_bundle(bundle, path)


def load_solver_bundle(path: str) -> dict:
    """兼容旧调用；实际实现见 :func:`SimuCoronary.utils.load_solver_bundle`。"""
    return _load_solver_bundle(path)


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
        f"q_scale_cal_{name}_nx{nx}_T{duration_s:.2f}"
        + ("_with_virtual_stent" if use_virtual_stent else ""),
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
