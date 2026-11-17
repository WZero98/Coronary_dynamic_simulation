"""
q_scale 标定迭代的缓存读写与事后绘图。

与 ``navier_stokes_1d.calibrate_q_scale_to_inlet_peak(..., cache_dir=...)`` 配合：
标定过程写入 ``iter_XX.npz`` + ``summary.json``；本模块负责落盘、读回与出图。

用法
----
::

    python calibration_cache.py data/cache/q_scale_cal_stenoses_nx200_T3.00

或::

    from calibration_cache import load_calibration_cache, plot_calibration_iterations
    cal = load_calibration_cache(r"data/cache/...")
    plot_calibration_iterations(cal)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def stenosis_probe_index(solver: Any) -> int:
    """狭窄探针：主狭窄 MLA；若无狭窄则取近远端之间面积最小点。"""
    if solver.lesions:
        return int(solver.lesions[0].mla_idx)
    i0 = int(solver.prox_idx) + 1
    i1 = int(solver.dist_idx)
    if i1 <= i0:
        return int(np.argmin(solver.area_ref))
    return i0 + int(np.argmin(solver.area_ref[i0:i1]))


def iteration_snapshot(
    solver: Any,
    it: int,
    q_scale: float,
    peak: float,
    in_range: bool,
) -> dict:
    """单次标定迭代的可缓存快照（含完整时空场，便于事后分析）。"""
    assert solver.history_time is not None
    assert solver.history_area is not None
    assert solver.history_flow is not None
    assert solver.history_pressure is not None
    prox = int(solver.prox_idx)
    dist = int(solver.dist_idx)
    sten = stenosis_probe_index(solver)
    ffr = np.asarray(solver.ffr_ratio(), dtype=float)
    return {
        "iter": int(it),
        "q_scale": float(q_scale),
        "q_mean": float(solver.paras.mean_coronary_flow_ml_s()),
        "p_in_peak": float(peak),
        "in_range": bool(in_range),
        "prox_idx": prox,
        "sten_idx": sten,
        "dist_idx": dist,
        "x_prox_cm": float(solver.x[prox]),
        "x_sten_cm": float(solver.x[sten]),
        "x_dist_cm": float(solver.x[dist]),
        "time": np.asarray(solver.history_time, dtype=float).copy(),
        "x": np.asarray(solver.x, dtype=float).copy(),
        "area": np.asarray(solver.history_area, dtype=float).copy(),
        "flow": np.asarray(solver.history_flow, dtype=float).copy(),
        "pressure": np.asarray(solver.history_pressure, dtype=float).copy(),
        "ffr": ffr.copy(),
        "p_prox": np.asarray(solver.history_pressure[:, prox], dtype=float).copy(),
        "p_sten": np.asarray(solver.history_pressure[:, sten], dtype=float).copy(),
        "p_dist": np.asarray(solver.history_pressure[:, dist], dtype=float).copy(),
        "q_prox": np.asarray(solver.history_flow[:, prox], dtype=float).copy(),
        "q_sten": np.asarray(solver.history_flow[:, sten], dtype=float).copy(),
        "q_dist": np.asarray(solver.history_flow[:, dist], dtype=float).copy(),
        "a_prox": np.asarray(solver.history_area[:, prox], dtype=float).copy(),
        "a_sten": np.asarray(solver.history_area[:, sten], dtype=float).copy(),
        "a_dist": np.asarray(solver.history_area[:, dist], dtype=float).copy(),
    }


def save_calibration_cache(
    report: dict,
    cache_dir: str,
    *,
    tag: str = "q_scale_cal",
) -> str:
    """把标定 ``report['history']`` 写入 ``cache_dir``，返回目录路径。"""
    os.makedirs(cache_dir, exist_ok=True)
    history = list(report.get("history", []))
    summary = {
        "tag": tag,
        "converged": bool(report.get("converged", False)),
        "q_scale": float(report.get("q_scale", float("nan"))),
        "p_in_peak": float(report.get("p_in_peak", float("nan"))),
        "q_mean": float(report.get("q_mean", float("nan"))),
        "n_iters": int(report.get("n_iters", len(history))),
        "peak_lo_mmhg": float(report.get("peak_lo_mmhg", float("nan"))),
        "peak_hi_mmhg": float(report.get("peak_hi_mmhg", float("nan"))),
        "cache_dir": os.path.abspath(cache_dir),
        "iterations": [],
    }
    for snap in history:
        it = int(snap["iter"])
        npz_name = f"iter_{it:02d}.npz"
        np.savez_compressed(
            os.path.join(cache_dir, npz_name),
            time=snap["time"],
            x=snap["x"],
            area=snap["area"],
            flow=snap["flow"],
            pressure=snap["pressure"],
            ffr=snap["ffr"],
            p_prox=snap["p_prox"],
            p_sten=snap["p_sten"],
            p_dist=snap["p_dist"],
            q_prox=snap["q_prox"],
            q_sten=snap["q_sten"],
            q_dist=snap["q_dist"],
            a_prox=snap["a_prox"],
            a_sten=snap["a_sten"],
            a_dist=snap["a_dist"],
            q_scale=np.array(snap["q_scale"]),
            q_mean=np.array(snap["q_mean"]),
            p_in_peak=np.array(snap["p_in_peak"]),
            in_range=np.array(bool(snap["in_range"])),
            prox_idx=np.array(snap["prox_idx"]),
            sten_idx=np.array(snap["sten_idx"]),
            dist_idx=np.array(snap["dist_idx"]),
            x_prox_cm=np.array(snap["x_prox_cm"]),
            x_sten_cm=np.array(snap["x_sten_cm"]),
            x_dist_cm=np.array(snap["x_dist_cm"]),
        )
        summary["iterations"].append(
            {
                "iter": it,
                "file": npz_name,
                "q_scale": float(snap["q_scale"]),
                "q_mean": float(snap["q_mean"]),
                "p_in_peak": float(snap["p_in_peak"]),
                "in_range": bool(snap["in_range"]),
                "prox_idx": int(snap["prox_idx"]),
                "sten_idx": int(snap["sten_idx"]),
                "dist_idx": int(snap["dist_idx"]),
                "x_prox_cm": float(snap["x_prox_cm"]),
                "x_sten_cm": float(snap["x_sten_cm"]),
                "x_dist_cm": float(snap["x_dist_cm"]),
            }
        )
    with open(os.path.join(cache_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report["cache_dir"] = os.path.abspath(cache_dir)
    return os.path.abspath(cache_dir)


def load_calibration_cache(cache_dir: str) -> dict:
    """读取 ``save_calibration_cache`` 写出的目录，返回与标定类似的 report。"""
    with open(os.path.join(cache_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    history = []
    for item in summary["iterations"]:
        data = np.load(os.path.join(cache_dir, item["file"]))
        snap = {k: data[k] for k in data.files}
        snap["iter"] = int(item["iter"])
        snap["in_range"] = bool(np.asarray(snap["in_range"]).reshape(-1)[0])
        for key in (
            "q_scale",
            "q_mean",
            "p_in_peak",
            "x_prox_cm",
            "x_sten_cm",
            "x_dist_cm",
        ):
            snap[key] = float(np.asarray(snap[key]).reshape(-1)[0])
        for key in ("prox_idx", "sten_idx", "dist_idx"):
            snap[key] = int(np.asarray(snap[key]).reshape(-1)[0])
        history.append(snap)
    return {
        "converged": bool(summary.get("converged", False)),
        "q_scale": float(summary.get("q_scale", float("nan"))),
        "p_in_peak": float(summary.get("p_in_peak", float("nan"))),
        "q_mean": float(summary.get("q_mean", float("nan"))),
        "n_iters": int(summary.get("n_iters", len(history))),
        "peak_lo_mmhg": float(summary.get("peak_lo_mmhg", float("nan"))),
        "peak_hi_mmhg": float(summary.get("peak_hi_mmhg", float("nan"))),
        "cache_dir": os.path.abspath(cache_dir),
        "history": history,
    }


def plot_calibration_iterations(
    report: dict,
    *,
    out_dir: str | None = None,
    show: bool = False,
) -> list[str]:
    """绘制每次迭代：近端/狭窄/远端的 P、Q、A(t)，以及 FFR(x)。"""
    import matplotlib.pyplot as plt

    hist = list(report.get("history", []))
    if hist and "time" not in hist[0]:
        cache_dir = report.get("cache_dir")
        if not cache_dir:
            raise ValueError("report.history 无场数据且缺少 cache_dir，无法绘图")
        report = load_calibration_cache(str(cache_dir))
        hist = list(report["history"])
    if not hist:
        raise ValueError("没有可绘制的迭代历史")

    if out_dir is None:
        out_dir = report.get("cache_dir") or "."
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )

    saved: list[str] = []
    sites = (
        ("prox", "近端参考"),
        ("sten", "狭窄处"),
        ("dist", "远端参考"),
    )
    quantities = (
        ("p", "压力 (mmHg)", "p"),
        ("q", "流量 (mL/s)", "q"),
        ("a", "截面积 (cm²)", "a"),
    )

    for snap in hist:
        it = int(snap["iter"])
        t = np.asarray(snap["time"], dtype=float)
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        fig.suptitle(
            f"迭代 {it}: q_scale={float(snap['q_scale']):.4f}, "
            f"p(0)_peak={float(snap['p_in_peak']):.1f} mmHg, "
            f"Q_mean={float(snap['q_mean']):.3f} mL/s",
            fontsize=12,
        )
        for col, (site_key, site_name) in enumerate(sites):
            x_cm = float(snap[f"x_{site_key}_cm"])
            for row, (_qkey, ylabel, prefix) in enumerate(quantities):
                ax = axes[row, col]
                y = np.asarray(snap[f"{prefix}_{site_key}"], dtype=float)
                ax.plot(t, y, lw=1.5)
                if row == 0:
                    ax.set_title(f"{site_name}\n(x={x_cm:.2f} cm)")
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 2:
                    ax.set_xlabel("t (s)")
                ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"iter_{it:02d}_probes.png")
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        saved.append(path)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        x = np.asarray(snap["x"], dtype=float)
        ffr = np.asarray(snap["ffr"], dtype=float)
        ax2.plot(x, ffr, lw=2, color="C0")
        for site_key, site_name, color in (
            ("prox", "近端", "C1"),
            ("sten", "狭窄", "C3"),
            ("dist", "远端", "C2"),
        ):
            xi = float(snap[f"x_{site_key}_cm"])
            ax2.axvline(xi, color=color, ls="--", alpha=0.8, label=site_name)
        ax2.set_xlabel("x (cm)")
        ax2.set_ylabel("FFR")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_title(
            f"迭代 {it}: FFR(x), q_scale={float(snap['q_scale']):.4f}"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        path2 = os.path.join(out_dir, f"iter_{it:02d}_ffr.png")
        fig2.savefig(path2, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig2)
        saved.append(path2)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for snap in hist:
        it = int(snap["iter"])
        x = np.asarray(snap["x"], dtype=float)
        ffr = np.asarray(snap["ffr"], dtype=float)
        ax3.plot(
            x,
            ffr,
            lw=1.8,
            label=(
                f"iter{it} qs={float(snap['q_scale']):.3f} "
                f"peak={float(snap['p_in_peak']):.0f}"
            ),
        )
    ax3.set_xlabel("x (cm)")
    ax3.set_ylabel("FFR")
    ax3.set_ylim(0.0, 1.05)
    ax3.set_title("各迭代 FFR(x) 对比")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc="best")
    fig3.tight_layout()
    path3 = os.path.join(out_dir, "all_iters_ffr.png")
    fig3.savefig(path3, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig3)
    saved.append(path3)
    return saved


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="读取 q_scale 标定缓存并绘制探针时程 / FFR(x)"
    )
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default=os.path.join(
            "data", "cache", "q_scale_cal_stenoses_nx200_T3.00"
        ),
        help="含 summary.json 与 iter_XX.npz 的缓存目录",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="图片输出目录（默认写回 cache_dir）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示图像",
    )
    args = parser.parse_args(argv)

    cal = load_calibration_cache(args.cache_dir)
    print(
        f"loaded {cal['n_iters']} iters from {cal['cache_dir']}; "
        f"converged={cal['converged']}, "
        f"q_scale={cal['q_scale']:.4f}, "
        f"p(0)_peak={cal['p_in_peak']:.2f} mmHg"
    )
    paths = plot_calibration_iterations(
        cal, out_dir=args.out_dir, show=args.show
    )
    print(f"wrote {len(paths)} figures")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
