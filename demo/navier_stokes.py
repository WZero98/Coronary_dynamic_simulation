"""
冠状动脉单支血管 — 1D 不可压缩 Navier-Stokes 血流模型

建模要点
--------
- 控制方程: 1D 面积守恒 + 动量守恒（摩擦源项），等价于不可压缩管流的一维化简
- 空间离散: TVD + MUSCL 线性重构 + minmod 限制器 + 界面 HLL 通量
- 时间离散: SSP-RK2
- 结构: 单入口、单出口
- 入口: inlet_boundary_condition.coronary_inlet_flow 生理脉动流量
- 出口: R-C Windkessel 微循环 + Murray 定律流量 + 血压匹配
         (outlet_boundary_condition.RCWindkesselMurrayOutletBC)
- 压力: 全脚本 mmHg；入口参考平均脉压 80 mmHg（tube law 基准）

守恒形式
--------
    ∂U/∂t + ∂F/∂x = S,   U = [A, Q]
    F = [Q, αQ²/A + p(A)·A/ρ]   (p 在通量中换算为 cgs)
    S = [0, -8πνQ/A]
    p(A) = P_ref + β(√A - √A₀)   [mmHg]
"""

import numpy as np
import matplotlib.pyplot as plt

from demo.units import P_INLET_REF_MMHG, MMHG_TO_CGS, pressure_mmhg_to_cgs
from inlet_boundary_condition import coronary_inlet_flow
from outlet_boundary_condition import (
    RCWindkesselMurrayOutletBC,
    area_from_pressure_mmhg,
)


def minmod(a, b):
    """向量化 minmod 限制器。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


class OneDBloodFlowTVD:
    """
    单支冠状动脉 1D TVD 求解器。

    参数
    ----
    length, nx : 血管长度 (cm) 与网格数
    rho, alpha, nu, cfl : 血液密度、动量修正、粘度、CFL
    p_inlet_ref_mmhg : tube law 入口参考压 (mmHg)，默认 80
    outlet_radius_cm : 出口等效半径 (cm)，None 时由 A₀ 估计
    """

    heart_rate = 75

    def __init__(
        self,
        length,
        nx,
        rho=1.05,
        alpha=1.1,
        nu=0.0035,
        cfl=0.5,
        p_inlet_ref_mmhg=P_INLET_REF_MMHG,
        outlet_radius_cm=None,
        p_venous_mmhg=5.0,
        q_mean_ml_s=3.33,
        wk_R=None,
        wk_C=0.8,
    ):
        self.length = length
        self.nx = nx
        self.rho = rho
        self.alpha = alpha
        self.nu = nu
        self.cfl = cfl
        self.p_inlet_ref_mmhg = float(p_inlet_ref_mmhg)

        self.x = np.linspace(0.0, length, nx)
        self.dx = length / (nx - 1)

        self.U = np.zeros((2, nx))
        self.p = np.zeros(nx)  # mmHg

        self.A0 = np.full(nx, 0.5)
        self.beta = np.full(nx, 750.0)  # mmHg / sqrt(cm)，对应原 ~1e6 dyne/cm² 量级

        self._outlet_radius_param = outlet_radius_cm
        self._bc_dt = 0.0
        self.outlet_bc = RCWindkesselMurrayOutletBC(
            outlet_radius_cm=0.4 if outlet_radius_cm is None else outlet_radius_cm,
            R_mmhg_s_per_ml=wk_R,
            C_ml_per_mmhg=wk_C,
            p_inlet_ref_mmhg=self.p_inlet_ref_mmhg,
            p_venous_mmhg=p_venous_mmhg,
            q_mean_ml_s=q_mean_ml_s,
        )

        self.current_time = 0.0
        self.p_history = None

    # ---- 管壁弹性律 (mmHg) -------------------------------------------------

    def pressure_from_area(self, A, A0=None, beta=None):
        """p(mmHg) = P_ref + β(√A - √A₀)。"""
        if A0 is None:
            A0 = self.A0
        if beta is None:
            beta = self.beta
        A0 = np.asarray(A0, dtype=float)
        beta = np.asarray(beta, dtype=float)
        A = np.maximum(np.asarray(A, dtype=float), 1e-12)
        return self.p_inlet_ref_mmhg + beta * (np.sqrt(A) - np.sqrt(A0))

    def update_pressure(self, U=None):
        A = self.U[0] if U is None else U[0]
        self.p = self.pressure_from_area(A)
        return self.p

    def set_vessel_parameters(self, A0=None, beta=None, stenosis=None):
        if A0 is not None:
            self.A0 = np.asarray(A0, dtype=float)
        if beta is not None:
            self.beta = np.asarray(beta, dtype=float)

        if stenosis is not None:
            for x0, x1, area_factor, beta_factor in stenosis:
                mask = (self.x >= x0) & (self.x <= x1)
                self.A0[mask] *= area_factor
                self.beta[mask] *= beta_factor

        self.U[0] = self.A0.copy()
        self.U[1] = 0.0
        self._sync_outlet_geometry()
        self.update_pressure()
        self.outlet_bc.reset(P_init_mmhg=self.p[-1])

    def _sync_outlet_geometry(self):
        r_in = float(np.sqrt(self.A0[0] / np.pi))
        r_out = (
            float(self._outlet_radius_param)
            if self._outlet_radius_param is not None
            else float(np.sqrt(self.A0[-1] / np.pi))
        )
        self.outlet_bc.set_radii(r_in, r_out)

    def wave_speed(self, A, beta):
        beta_cgs = np.asarray(beta, dtype=float) * MMHG_TO_CGS
        A_safe = np.maximum(A, 1e-12)
        return np.sqrt(beta_cgs / (2.0 * self.rho)) * A_safe ** (-0.25)

    def physical_flux(self, U, beta=None, A0=None):
        A = np.atleast_1d(np.asarray(U[0], dtype=float))
        Q = np.atleast_1d(np.asarray(U[1], dtype=float))
        if beta is None:
            beta = self.beta
        if A0 is None:
            A0 = self.A0
        beta = np.atleast_1d(np.asarray(beta, dtype=float))
        A0 = np.atleast_1d(np.asarray(A0, dtype=float))

        A_safe = np.maximum(A, 1e-12)
        p_mmhg = self.pressure_from_area(A_safe, A0, beta)
        p_cgs = pressure_mmhg_to_cgs(p_mmhg)
        F0 = Q
        F1 = self.alpha * Q ** 2 / A_safe + p_cgs * A_safe / self.rho
        if F0.size == 1:
            return np.array([F0.item(), F1.item()])
        return np.vstack([F0, F1])

    # ---- MUSCL + HLL --------------------------------------------------------

    def muscl_reconstruct(self, U):
        nx = self.nx
        slope = np.zeros((2, nx))
        if nx >= 3:
            slope[:, 1:-1] = minmod(
                U[:, 1:-1] - U[:, :-2],
                U[:, 2:] - U[:, 1:-1],
            )
        U_L_if = U[:, : nx - 1] + 0.5 * slope[:, : nx - 1]
        U_R_if = U[:, 1:nx] - 0.5 * slope[:, 1:nx]
        A_floor_L = 0.1 * self.A0[: nx - 1]
        A_floor_R = 0.1 * self.A0[1:nx]
        U_L_if[0] = np.maximum(U_L_if[0], A_floor_L)
        U_R_if[0] = np.maximum(U_R_if[0], A_floor_R)
        return U_L_if, U_R_if

    def hll_flux_batch(self, U_L, U_R, beta_L, beta_R, A0_L, A0_R):
        A_L = np.maximum(U_L[0], 1e-12)
        A_R = np.maximum(U_R[0], 1e-12)
        Q_L, Q_R = U_L[1], U_R[1]

        beta_L_cgs = beta_L * MMHG_TO_CGS
        beta_R_cgs = beta_R * MMHG_TO_CGS
        u_L, u_R = Q_L / A_L, Q_R / A_R
        c_L = np.sqrt(beta_L_cgs / (2.0 * self.rho)) * A_L ** (-0.25)
        c_R = np.sqrt(beta_R_cgs / (2.0 * self.rho)) * A_R ** (-0.25)

        S_L = np.minimum(u_L - c_L, u_R - c_R)
        S_R = np.maximum(u_L + c_L, u_R + c_R)

        p_L = pressure_mmhg_to_cgs(self.pressure_from_area(A_L, A0_L, beta_L))
        p_R = pressure_mmhg_to_cgs(self.pressure_from_area(A_R, A0_R, beta_R))
        F_L = np.vstack([Q_L, self.alpha * Q_L ** 2 / A_L + p_L * A_L / self.rho])
        F_R = np.vstack([Q_R, self.alpha * Q_R ** 2 / A_R + p_R * A_R / self.rho])

        F = np.zeros_like(F_L)
        m_left = S_L >= 0.0
        m_right = S_R <= 0.0
        m_mid = ~(m_left | m_right)

        F[:, m_left] = F_L[:, m_left]
        F[:, m_right] = F_R[:, m_right]
        denom = S_R[m_mid] - S_L[m_mid]
        F[:, m_mid] = (
            S_R[m_mid] * F_L[:, m_mid]
            - S_L[m_mid] * F_R[:, m_mid]
            + S_L[m_mid] * S_R[m_mid] * (U_R[:, m_mid] - U_L[:, m_mid])
        ) / denom
        return F

    def compute_interface_fluxes(self, U):
        U_L_if, U_R_if = self.muscl_reconstruct(U)
        j = np.arange(self.nx - 1)
        F_face = np.zeros((2, self.nx + 1))
        F_face[:, 1:-1] = self.hll_flux_batch(
            U_L_if, U_R_if,
            self.beta[j], self.beta[j + 1],
            self.A0[j], self.A0[j + 1],
        )
        F_face[:, 0] = self.physical_flux(U[:, 0], beta=self.beta[0], A0=self.A0[0])
        F_face[:, -1] = self.physical_flux(
            U[:, -1], beta=self.beta[-1], A0=self.A0[-1]
        )
        return F_face

    def rhs(self, U):
        F_face = self.compute_interface_fluxes(U)
        dUdt = -(F_face[:, 1:] - F_face[:, :-1]) / self.dx
        A = np.maximum(U[0], 1e-12)
        dUdt[1] += -8.0 * np.pi * self.nu * U[1] / A
        return dUdt

    # ---- 边界条件 ----------------------------------------------------------

    def apply_boundary_conditions(self, U, advance_windkessel=False):
        """
        入口: 定 Q（生理冠脉波形）+ A 外推
        出口: R-C Windkessel + Murray + 血压匹配 → Q_out, P_wk → A_out
        """
        U = U.copy()
        t = self.current_time
        Q_in = coronary_inlet_flow(
            t, heart_rate=self.heart_rate, waveform="physiological"
        )

        U[0, 0] = U[0, 1]
        U[1, 0] = Q_in

        p_in = float(
            self.pressure_from_area(U[0, 0], self.A0[0], self.beta[0])
        )
        if advance_windkessel and self._bc_dt > 0.0:
            Q_out, P_wk = self.outlet_bc.advance(
                self._bc_dt, U[1, -1], Q_in, p_in
            )
        else:
            Q_out, P_wk = self.outlet_bc.evaluate(Q_in, p_in, U[1, -1])

        U[1, -1] = Q_out
        U[0, -1] = area_from_pressure_mmhg(
            P_wk,
            self.A0[-1],
            self.beta[-1],
            p_ref=self.p_inlet_ref_mmhg,
        )
        return U

    def enforce_physical_bounds(self, U):
        U = U.copy()
        U[0] = np.maximum(U[0], 0.1 * self.A0)
        return U

    # ---- 时间推进 ----------------------------------------------------------

    def adaptive_dt(self, U):
        A = np.maximum(U[0], 1e-12)
        u = U[1] / A
        c = self.wave_speed(A, self.beta)
        lam = np.max(np.abs(u) + c)
        if lam < 1e-14:
            return 1e-4
        return min(self.cfl * self.dx / lam, 5e-4)

    def ssp_rk2_step(self, dt):
        self._bc_dt = dt
        U0 = self.apply_boundary_conditions(self.U)
        L0 = self.rhs(U0)
        U1 = self.enforce_physical_bounds(U0 + dt * L0)
        U1 = self.apply_boundary_conditions(U1)

        L1 = self.rhs(U1)
        self.U = self.enforce_physical_bounds(0.5 * U0 + 0.5 * (U1 + dt * L1))
        self.U = self.apply_boundary_conditions(self.U, advance_windkessel=True)
        self.update_pressure()

    def solve(self, T_total, save_every=20):
        times = [0.0]
        A_hist = [self.U[0].copy()]
        Q_hist = [self.U[1].copy()]
        p_hist = [self.p.copy()]
        step = 0

        print(
            f"TVD 求解: T={T_total}s, nx={self.nx}, dx={self.dx:.4f} cm, "
            f"P_ref={self.p_inlet_ref_mmhg} mmHg"
        )

        while self.current_time < T_total - 1e-15:
            print(
                f"current time: {self.current_time:.4f} s / Total: {T_total:.4f} s",
                flush=True,
                end="\r",
            )
            dt = self.adaptive_dt(self.U)
            if self.current_time + dt > T_total:
                dt = T_total - self.current_time

            self.ssp_rk2_step(dt)
            self.current_time += dt
            step += 1

            if step % save_every == 0:
                times.append(self.current_time)
                A_hist.append(self.U[0].copy())
                Q_hist.append(self.U[1].copy())
                p_hist.append(self.p.copy())

        self.times = np.array(times)
        self.A_history = np.array(A_hist)
        self.Q_history = np.array(Q_hist)
        self.p_history = np.array(p_hist)
        print(f"完成: {step} 步, t={self.current_time:.4f} s, dt={dt:.8f} s")
        return self.A_history, self.Q_history, self.p_history, self.times

    # ---- 后处理 -----------------------------------------------------------

    def pressure_field(self, A):
        return self.pressure_from_area(A, self.A0, self.beta)

    def compute_ffr(self):
        if self.p_history is not None:
            p_mean = np.mean(self.p_history, axis=0)
        else:
            p_mean = self.pressure_field(np.mean(self.A_history, axis=0))
        return p_mean / p_mean[0]

    def visualize(self):
        if self.p_history is None:
            raise RuntimeError("请先调用 solve() 以生成 p_history")

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

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=False)
        cbar_kw = dict(fraction=0.046, pad=0.06, aspect=25)

        ax_a = axes[0, 0]
        im_a = ax_a.contourf(
            self.times, self.x, self.A_history.T, levels=30, cmap="viridis"
        )
        ax_a.set(xlabel="t (s)", ylabel="x (cm)", title="A(t, x)")
        cbar_a = fig.colorbar(im_a, ax=ax_a, **cbar_kw)
        cbar_a.set_label("A (cm²)", fontsize=9)

        ax_p = axes[0, 1]
        p_mean_x = np.mean(self.p_history, axis=0)
        (line_p_x,) = ax_p.plot(
            self.x,
            p_mean_x,
            "g-",
            lw=1.8,
            label=r"$\langle p \rangle_t$ (mmHg)",
        )
        ax_p.set_xlabel("x (cm)")
        ax_p.set_ylabel("p (mmHg)")
        ax_p.set_title(r"时间平均压力 $\langle p(x) \rangle_t$")
        ax_p.grid(True, alpha=0.35)
        ax_p.legend(
            handles=[line_p_x],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            framealpha=0.92,
            edgecolor="0.75",
            fontsize=9,
        )

        x_point = 0
        ax_ts = axes[1, 0]
        (line_q,) = ax_ts.plot(
            self.times,
            self.Q_history[:, x_point],
            "r-",
            lw=1.5,
            label="Q (cm³/s)",
        )
        ax_ts.set_xlabel("t (s)")
        ax_ts.set_ylabel("Q (cm³/s)", color="r")
        ax_ts.tick_params(axis="y", labelcolor="r")

        ax_ts_p = ax_ts.twinx()
        (line_p,) = ax_ts_p.plot(
            self.times,
            self.p_history[:, x_point],
            "g--",
            lw=1.5,
            label="p (mmHg)",
            alpha=0.9,
        )
        ax_ts_p.set_ylabel("p (mmHg)", color="g")
        ax_ts_p.tick_params(axis="y", labelcolor="g")
        ax_ts.set_title(f"入口 x = {self.x[x_point]:.1f} cm")
        ax_ts.grid(True, alpha=0.35)
        ax_ts.legend(
            [line_q, line_p],
            [line_q.get_label(), line_p.get_label()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            borderaxespad=0.0,
            framealpha=0.92,
            edgecolor="0.75",
            fontsize=9,
        )

        ax_sp = axes[1, 1]
        (line_a,) = ax_sp.plot(
            self.x, self.A_history[-1], "b-", lw=1.5, label="A (cm²)"
        )
        ax_sp.set_xlabel("x (cm)")
        ax_sp.set_ylabel("A (cm²)", color="b")
        ax_sp.tick_params(axis="y", labelcolor="b")

        ax_sp_r = ax_sp.twinx()
        (line_q2,) = ax_sp_r.plot(
            self.x, self.Q_history[-1], "r-", lw=1.5, label="Q (cm³/s)"
        )
        (line_p2,) = ax_sp_r.plot(
            self.x,
            self.p_history[-1],
            "g--",
            lw=1.5,
            alpha=0.9,
            label="p (mmHg)",
        )
        ax_sp_r.set_ylabel("Q / p")
        ax_sp.set_title(f"沿程分布  t = {self.times[-1]:.3f} s")
        ax_sp.grid(True, alpha=0.35)
        ax_sp.legend(
            [line_a, line_q2, line_p2],
            [line_a.get_label(), line_q2.get_label(), line_p2.get_label()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=3,
            borderaxespad=0.0,
            framealpha=0.92,
            edgecolor="0.75",
            fontsize=9,
            columnspacing=1.0,
        )

        fig.subplots_adjust(
            left=0.07, right=0.86, top=0.90, bottom=0.16, hspace=0.52, wspace=0.42
        )
        plt.show()
        return fig


TVDBloodFlow = OneDBloodFlowTVD


if __name__ == "__main__":
    L, NX = 30.0, 151
    A0 = np.full(NX, 0.5)
    beta = np.full(NX, 750.0)

    solver = OneDBloodFlowTVD(length=L, nx=NX, cfl=0.45)
    solver.set_vessel_parameters(
        A0=A0,
        beta=beta,
        stenosis=[(12.0, 18.0, 0.7, 5.0)],
    )
    A_tvd, Q_tvd, p_tvd, t_tvd = solver.solve(T_total=3, save_every=20)
    solver.visualize()

    FFR = solver.compute_ffr()

    print(f"\n最小 A: {A_tvd[-1].min():.4f} cm²")
    print(f"最大 |Q|: {np.abs(Q_tvd).max():.2f} cm³/s")
    print(f"压力范围: [{p_tvd.min():.2f}, {p_tvd.max():.2f}] mmHg")
