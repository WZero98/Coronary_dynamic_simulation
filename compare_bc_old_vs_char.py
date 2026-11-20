"""对比旧边界（入口面积零梯度 / 出口只定 A）与当前特征相容边界。

当前算例：``data/stenoses_data``，``target_nx=200``，``duration_s=3``，
与 ``example(target_nx=200, name='stenoses')`` 及
``docs/当前算例问题与改进方向.md`` 基线一致（默认参数，不做 q_scale 标定）。

输出目录：``docs/figures/bc_compare_stenoses_nx200/``
"""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

from geometry import build_stenosis_segment, resample_axial_profile
from navier_stokes_1d import (
    BloodFlowParameters,
    NavierStokes1D,
    area_from_lumen_pressure_mmhg,
    integral_mean,
)


OUT_DIR = os.path.join("docs", "figures", "bc_compare_stenoses_nx200")


class NavierStokes1DOldBC(NavierStokes1D):
    """修改前边界：入口 A(0)=A(1)；出口只按 Windkessel 改 A，保留 PDE 的 Q(L)。"""

    def _apply_boundaries(self, state: np.ndarray, advance: bool = False) -> np.ndarray:
        u = state.copy()
        if not advance:
            q_in = float(np.asarray(self.paras.inlet_flow(self.time)).reshape(-1)[0])
        else:
            q_in = float(
                np.asarray(self.paras.inlet_flow(self.time + self._outlet_dt)).reshape(-1)[0]
            )
        u[1, 0] = q_in
        u[0, 0] = u[0, 1]

        self._apply_side_branches(u, advance=advance)

        q_lumen = float(u[1, -1])
        if advance and self._outlet_dt > 0.0:
            p_out = self.outlet.advance(self._outlet_dt, q_lumen)
        else:
            p_out = self.outlet.apply_substep(q_lumen)
        u[0, -1] = float(
            np.asarray(
                area_from_lumen_pressure_mmhg(
                    p_out,
                    self.area_ref[-1],
                    self.beta[-1],
                    self.paras.p_ref_mmhg,
                )
            ).reshape(-1)[0]
        )
        return u


def _load_stenoses_case(target_nx: int = 200, sigma_nodes: float = 4.0):
    mm_per_pixel = 9.7 / 756
    dirname = os.path.join("data", "stenoses_data")
    area = np.array(joblib.load(os.path.join(dirname, "areas.joblib")))[::-1]
    area = area * (mm_per_pixel**2) / 100.0
    area_smooth = gaussian_filter1d(area, sigma=sigma_nodes, mode="nearest")
    nx = int(area_smooth.shape[0])

    branch_groups = joblib.load(os.path.join(dirname, "branches.joblib"))
    branch_d = np.array(joblib.load(os.path.join(dirname, "branch_diameters.joblib")))
    branch_d = branch_d[::-1] * mm_per_pixel / 10.0
    branch_idx = [nx - idxs[0] for idxs in branch_groups][::-1]

    area_smooth, length, nx, branch_idx, branch_d = resample_axial_profile(
        area_smooth,
        target_nx=target_nx,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
        frame_spacing_cm=0.02,
    )
    return area_smooth, float(length), int(nx), list(branch_idx), list(branch_d)


def _metrics(solver: NavierStokes1D) -> dict:
    cycle = 60.0 / float(solver.paras.heart_rate_bpm)
    bal = solver.mass_balance(cycle_s=cycle)
    diag = solver.wellposedness_diagnostics()
    assert solver.history_time is not None
    assert solver.history_pressure is not None
    assert solver.history_flow is not None
    t = np.asarray(solver.history_time, dtype=float)
    t0 = float(t[-1]) - cycle
    mask = t >= t0 if t0 > float(t[0]) else np.ones(t.shape[0], dtype=bool)
    p = solver.history_pressure[mask]
    q = solver.history_flow[mask]
    prox = int(solver.prox_idx)
    dist = int(solver.dist_idx)
    p_prox = float(integral_mean(p[:, prox], t[mask]))
    p_dist = float(integral_mean(p[:, dist], t[mask]))
    return {
        "mass_err": float(bal["mass_err"]),
        "q_in": float(bal["q_in"]),
        "q_out": float(bal["q_out"]),
        "q_side": float(bal["q_side"]),
        "n_choked": int(diag["n_choked"]),
        "lam_minus_max": float(diag["lam_minus_max"]),
        "p_prox": p_prox,
        "p_dist": p_dist,
        "ffr": float(p_dist / p_prox) if p_prox > 1e-12 else float("nan"),
        "p0_peak": float(np.max(p[:, 0])),
        "qin_peak": float(np.max(q[:, 0])),
    }


def _run_one(cls, area, length, branch_idx, branch_d, duration_s: float, label: str):
    paras = BloodFlowParameters()
    solver = cls(
        area,
        length,
        paras,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
    )
    print(f"\n=== run {label}: {cls.__name__} ===")
    solver.run(duration_s=duration_s, record_interval_steps=30, verbose=True)
    m = _metrics(solver)
    print(json.dumps(m, indent=2))
    html = os.path.join(OUT_DIR, f"{label}_plotly.html")
    solver.plot_results_1(title=label, show=False, save_path=html)
    return solver, m


def _save_compare_png(old_s: NavierStokes1D, new_s: NavierStokes1D, path: str) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    cycle = 60.0 / float(old_s.paras.heart_rate_bpm)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    def last_cycle(solver):
        t = np.asarray(solver.history_time, dtype=float)
        t0 = float(t[-1]) - cycle
        mask = t >= t0 if t0 > float(t[0]) else np.ones(t.shape[0], dtype=bool)
        return t[mask], mask

    t_old, m_old = last_cycle(old_s)
    t_new, m_new = last_cycle(new_s)
    xx = old_s.x
    prox, dist = int(old_s.prox_idx), int(old_s.dist_idx)

    ax = axes[0, 0]
    ax.plot(t_old, old_s.history_pressure[m_old, 0], "C0--", lw=1.4, label="旧BC p(0)")
    ax.plot(t_new, new_s.history_pressure[m_new, 0], "C0-", lw=1.6, label="特征BC p(0)")
    ax.plot(t_old, old_s.history_pressure[m_old, prox], "C1--", lw=1.2, label="旧BC p_prox")
    ax.plot(t_new, new_s.history_pressure[m_new, prox], "C1-", lw=1.4, label="特征BC p_prox")
    ax.plot(t_old, old_s.history_pressure[m_old, dist], "C3--", lw=1.2, label="旧BC p_dist")
    ax.plot(t_new, new_s.history_pressure[m_new, dist], "C3-", lw=1.4, label="特征BC p_dist")
    ax.set(xlabel="t (s)", ylabel="p (mmHg)", title="末周期压力探针")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[0, 1]
    ax.plot(t_old, old_s.history_flow[m_old, 0], "C0--", lw=1.4, label="旧BC Q(0)")
    ax.plot(t_new, new_s.history_flow[m_new, 0], "C0-", lw=1.6, label="特征BC Q(0)")
    ax.plot(t_old, old_s.history_flow[m_old, -1], "C2--", lw=1.2, label="旧BC Q(L)")
    ax.plot(t_new, new_s.history_flow[m_new, -1], "C2-", lw=1.4, label="特征BC Q(L)")
    ax.set(xlabel="t (s)", ylabel="Q (mL/s)", title="末周期入口/出口流量")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    p_old = np.mean(old_s.history_pressure[m_old], axis=0)
    p_new = np.mean(new_s.history_pressure[m_new], axis=0)
    ax.plot(xx, p_old, "C0--", lw=1.6, label="旧BC")
    ax.plot(xx, p_new, "C1-", lw=1.6, label="特征BC")
    ax.axvline(xx[prox], color="gray", ls=":", lw=1)
    ax.axvline(xx[dist], color="gray", ls=":", lw=1)
    ax.set(xlabel="x (cm)", ylabel="p (mmHg)", title="末周期沿程平均压力")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    q_old = np.mean(old_s.history_flow[m_old], axis=0)
    q_new = np.mean(new_s.history_flow[m_new], axis=0)
    ax.plot(xx, q_old, "C0--", lw=1.6, label="旧BC")
    ax.plot(xx, q_new, "C1-", lw=1.6, label="特征BC")
    ax.set(xlabel="x (cm)", ylabel="Q (mL/s)", title="末周期沿程平均流量")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("stenoses_data nx=200：旧边界 vs 特征相容边界", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved compare png: {path}")


def main():
    np.random.seed(2034)
    os.makedirs(OUT_DIR, exist_ok=True)
    area, length, nx, branch_idx, branch_d = _load_stenoses_case(target_nx=200)
    dx = length / max(nx - 1, 1)
    sten = build_stenosis_segment(area, frame_spacing_cm=dx)
    prox, dist = int(sten.prox_ref_idx), int(sten.dist_ref_idx)
    print(
        f"case: nx={nx}, L={length:.3f} cm, n_branch={len(branch_idx)}, "
        f"prox={prox}, dist={dist}, mla={sten.mla_idx}"
    )
    duration_s = 3.0

    old_s, old_m = _run_one(
        NavierStokes1DOldBC, area, length, branch_idx, branch_d, duration_s, "old_bc"
    )
    new_s, new_m = _run_one(
        NavierStokes1D, area, length, branch_idx, branch_d, duration_s, "char_bc"
    )

    summary = {
        "case": "stenoses_data",
        "nx": nx,
        "length_cm": length,
        "duration_s": duration_s,
        "parameters": {
            k: v
            for k, v in BloodFlowParameters().__dict__.items()
            if v is None or isinstance(v, (int, float, str, bool))
        },
        "old_bc": old_m,
        "char_bc": new_m,
        "delta": {
            k: new_m[k] - old_m[k]
            for k in old_m
            if isinstance(old_m[k], (int, float))
        },
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved summary: {summary_path}")

    _save_compare_png(old_s, new_s, os.path.join(OUT_DIR, "compare_old_vs_char.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
