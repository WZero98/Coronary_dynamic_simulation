"""
navier_stokes_1d.py — 冠状动脉一维轴向血流模拟

依据一维可变形管内的守恒型 Navier-Stokes 方程组，耦合管壁弹性压力关系，
对沿血管轴向 x ∈ [0, L] 的截面积 A 与体积流量 Q 进行数值求解。

控制方程（守恒形式，长度/面积/流量用 CGS，管腔压 p 用 mmHg）
------------------------------------------------------------
    ∂U/∂t + ∂F/∂x = S,    U = [A, Q]ᵀ

    F₀ = Q
    F₁ = α·Q²/A + p_mmHg(A)·A/ρ*
    S  = [0, −8πμ·Q/A]ᵀ

    ρ* = ρ / (1333.22 g·cm⁻¹·s⁻² per mmHg)，在 BloodFlowParameters 中一次性标定；
    与逐点把 p 换成 dyne/cm² 再代入原式等价。

管壁弹性（tube law，mmHg）
--------------------------
    p(A, x) = P_ref + β(x)·(√A − √A₀(x))

    p 为管腔标量压（各向同性，非轴向/径向矢量分量）。tube law 将其与截面面积 A 耦合
    （径向胀缩）；动量通量中 ∂(pA)/∂x 则体现沿 x 的压力梯度对轴向流量 Q 的驱动。

    A₀(x) 由用户给定的沿程管腔参考面积描述；β(x) 为管壁刚度（可常数或随 x 给定）。

边界条件
--------
- 入口：coronary_inlet.coronary_inlet_flow 提供生理脉动入口流量 Q_in(t)（mL/s ≡ cm³/s）
- 出口：coronary_outlet 中 Windkessel 模型提供出口管腔压；求解过程中按
        C·dP_wk/dt = Q_drive − P_wk/R_d 推进微循环状态（与 windkessel2/3 一致），
        Q_drive 取出口管腔流量 Q(L)（0D–1D 质量闭合）；P_wk/R_d 为微循环侧出流，
        不再强制赋值给 Q(L)。再由 tube law 确定出口 A(L)

初值与模块耦合（set_lumen_area_profile）
----------------------------------------
设置 A₀(x) 后同步初始化，使 tube law、入口 BC 与 Windkessel 在 t=0 自洽：
- A(x,0) = A₀(x)  →  tube law 得 p(x,0) ≈ P_ref
- Q(x,0) = Q_in(0)  →  与入口定流量 BC 一致（非 Q_mean）
- P_wk(0) = Q_mean·R_d = P_ref  →  R_d = P_ref/Q_mean 稳态标定
- 三元模型：P_lumen(0) = P_wk + R_p·Q_in(0)（outlet.reset 的 q_drive）

参数链：CO、coronary_fraction → Q_mean → R_d = P_ref/Q_mean，与 tube law 锚点 P_ref 对齐。

空间离散：单元中心有限体积法 + MUSCL 线性重构 + minmod TVD 限制器 + HLL 数值通量
时间离散：SSP-RK2（二阶强稳定保持 Runge-Kutta）
时间步长：CFL 自适应

主要输入
--------
1. 血管长度 L 与网格分辨率
2. 沿 [0, L] 的管腔参考截面积 A₀(x)（及可选 β(x)）
3. BloodFlowParameters 中的血液物性、数值参数、入口/出口 Windkessel 参数
"""
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from coronary_constants import P_INLET_REF_MMHG, rho_for_mmhg_pressure_coupling
from coronary_inlet import coronary_inlet_flow
from geometry import detect_stenoses, select_reference_indices


# ---------------------------------------------------------------------------
# 管壁弹性律
# ---------------------------------------------------------------------------
def lumen_pressure_mmhg(
    area: np.ndarray,
    area_ref: np.ndarray,
    beta: np.ndarray,
    p_ref: float = P_INLET_REF_MMHG,
) -> np.ndarray:
    """由 tube law 计算管腔标量压 p（mmHg）：p = P_ref + β(√A − √A₀)。

    p 为各向同性热力学压强，非某一坐标方向的应力分量；此处由 A 闭合，
    用于 tube law 径向胀缩关系及动量方程中的 ∂(pA)/∂x 项。
    """
    a0 = np.asarray(area_ref, dtype=float)
    a = np.maximum(np.asarray(area, dtype=float), 0.5 * a0)
    b = np.asarray(beta, dtype=float)
    return (p_ref + b * (np.sqrt(a) - np.sqrt(a0)))


def area_from_lumen_pressure_mmhg(
    pressure: float | np.ndarray,
    area_ref,
    beta,
    p_ref: float = P_INLET_REF_MMHG,
) -> np.ndarray:
    """tube law 反解 A。"""
    a0 = np.asarray(area_ref, dtype=float)
    b = np.asarray(beta, dtype=float)
    p = np.asarray(pressure, dtype=float)
    sqrt_a = np.sqrt(a0) + (p - p_ref) / b
    return np.maximum(sqrt_a**2, 0.5 * a0)
    
# ---------------------------------------------------------------------------
# TVD / MUSCL / HLL
# ---------------------------------------------------------------------------
def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """minmod 限制器。
    对于从左、右边界计算的斜率a, b, 符号一致则取绝对值更小的；符号不一致则取 0。
    """
    signs = a * b
    signs = np.where(signs > 0, 1, 0)
    return signs * np.minimum(np.abs(a), np.abs(b))


def _muscl_states(
    U: np.ndarray, 
    area_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """单元界面 j+1/2 的左右重构状态（内部界面）。"""
    nvar, nx = U.shape  # nvar 为未知函数量，nx 为划分单元数
    slope = np.zeros_like(U)
    if nx >= 3:
        slope[:, 1:-1] = _minmod(U[:, 1:-1] - U[:, :-2], U[:, 2:] - U[:, 1:-1])
    # 左状态计算，来自当前单元
    u_left = U[:, :-1] + 0.5 * slope[:, :-1]
    # 右状态计算，来自下一个单元
    u_right = U[:, 1:] - 0.5 * slope[:, 1:]
    floor = 0.5 * area_ref
    u_left[0] = np.maximum(u_left[0], floor[:-1])
    u_right[0] = np.maximum(u_right[0], floor[1:])
    return u_left, u_right


def _wave_speed(
    area: np.ndarray | float,
    beta: np.ndarray | float,
    rho_mmhg: float,
) -> np.ndarray:
    """小扰动波速；p、β 均为 mmHg 制，ρ* 为动量耦合有效密度。"""
    b = np.asarray(beta, dtype=float)
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    # c² = (A/ρ*)·dp/dA，tube law 下 dp/dA = β/(2√A) → c = √(β√A/(2ρ*))
    return np.sqrt(b * np.sqrt(a) / (2.0 * rho_mmhg))


def _flux_vector(
    area: np.ndarray,
    flow: np.ndarray,
    area_ref: np.ndarray,
    beta: np.ndarray,
    rho_mmhg: float,
    alpha: float,
    p_ref: float,
) -> np.ndarray:
    """物理通量 F(U)；p 由 tube law 直接以 mmHg 代入动量项。"""
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    q = np.asarray(flow, dtype=float)
    p_mmhg = lumen_pressure_mmhg(a, area_ref, beta, p_ref)
    return np.vstack([q, alpha * q**2 / a + p_mmhg * a / rho_mmhg])


def _hll_flux(
    u_l: np.ndarray,
    u_r: np.ndarray,
    beta_l: np.ndarray,
    beta_r: np.ndarray,
    a0_l: np.ndarray,
    a0_r: np.ndarray,
    rho_mmhg: float,
    alpha: float,
    p_ref: float,
) -> np.ndarray:
    """HLL Riemann 求解器计算界面数值通量。"""
    # 管腔截面积左右状态
    a_l = np.maximum(u_l[0], 1e-12)
    a_r = np.maximum(u_r[0], 1e-12)
    # 管腔体积流量左右状态
    q_l, q_r = u_l[1], u_r[1]
    # 平均流速左右状态
    uvel_l, uvel_r = q_l / a_l, q_r / a_r
    # 波速左右状态
    c_l = _wave_speed(a_l, beta_l, rho_mmhg)
    c_r = _wave_speed(a_r, beta_r, rho_mmhg)
    # HLL Riemann 求解器 估算的SL, SR
    s_l = np.minimum(np.minimum(uvel_l - c_l, uvel_r - c_r), 0)
    s_r = np.maximum(np.maximum(uvel_l + c_l, uvel_r + c_r), 0)

    f_l = _flux_vector(a_l, q_l, a0_l, beta_l, rho_mmhg, alpha, p_ref)
    f_r = _flux_vector(a_r, q_r, a0_r, beta_r, rho_mmhg, alpha, p_ref)
    f = np.zeros_like(f_l)

    # 根据波速值，选对应的通量
    left = s_l >= 0.0
    right = s_r <= 0.0
    mid = ~(left | right)
    f[:, left] = f_l[:, left]
    f[:, right] = f_r[:, right]
    denom = s_r[mid] - s_l[mid]
    f[:, mid] = (
        s_r[mid] * f_l[:, mid]
        - s_l[mid] * f_r[:, mid]
        + s_l[mid] * s_r[mid] * (u_r[:, mid] - u_l[:, mid])
    ) / denom
    return f


# ---------------------------------------------------------------------------
# 参数与出口 Windkessel 状态（方程同 coronary_outlet）
# ---------------------------------------------------------------------------
@dataclass
class BloodFlowParameters:
    """固定物性、数值、入口与出口参数。"""

    # 流体相关
    rho: float = 1.06  # 血液密度 (g/cm³)
    alpha: float = 1.1  # 动量修正系数（层流常取 1.1）
    mu: float = 0.0035  # 血液粘度 (cm²/s)
    cfl: float = 0.8  # 数值稳定性参数 CFL
    dt_max_s: float = 4e-4  # 时间步上限 (s)

    # 管壁相关 tube law
    p_ref_mmhg: float = P_INLET_REF_MMHG  # tube law 参考压 (mmHg)
    beta_mmhg_per_sqrt_cm: float = 2000.0  # 默认β，越大表示管腔越硬（管壁刚度）

    # 冠脉血流入口相关
    heart_rate_bpm: float = 75.0  # 心率 (bpm)
    cardiac_output_l_per_min: float = 5.5  # 心脏输出量 (L/min)
    coronary_flow_fraction: float = 0.03  # 左冠脉血流占心脏输出量的比例
    inlet_min_flow_fraction: float = 0.25  # 流量下限，即最小值占整个流量平均值的比例
    inlet_phase_offset_rad: float = 0.0  # 波形相位偏移
    inlet_fourier_coefficients: Sequence[tuple[int, float, float]] | None = None  # 自定义谐波（傅里叶变换的系数）；None 用模块内置默认系数

    # 出口 Windkesse模型 相关
    outlet_windkessel: Literal["2wk", "3wk"] = "3wk"
    outlet_r_distal_mmhg_s_per_ml: float | None = None  # Rd 远端阻力系数
    outlet_r_proximal_mmhg_s_per_ml: float | None = None  # Rp 近端阻力系数
    outlet_compliance_ml_per_mmhg: float = 0.05  # 血管顺应性，越大顺应性越好 
    outlet_proximal_fraction: float = 0.12  # 三元模型近端阻力比例

    def mean_coronary_flow_ml_s(self) -> float:
        """入口平均流量 (mL/s)。"""
        return (
            self.cardiac_output_l_per_min
            * 1000.0
            * self.coronary_flow_fraction
            / 60.0
        )

    def inlet_flow(self, t: float | np.ndarray) -> np.ndarray:
        """生成时间t的入口流量 (mL/s)。"""
        return np.asarray(
            coronary_inlet_flow(
                t,
                heart_rate=self.heart_rate_bpm,
                cardiac_output=self.cardiac_output_l_per_min,
                coronary_fraction=self.coronary_flow_fraction,
                fourier_coefficients=self.inlet_fourier_coefficients,
                min_flow_fraction=self.inlet_min_flow_fraction,
                phase_offset=self.inlet_phase_offset_rad,
            ),
            dtype=float,
        )

    def resolve_outlet_resistances(self) -> tuple[float, float]:
        """远端/近端阻力 (mmHg·s/mL)，与 coronary_outlet 默认标定一致。"""
        q_mean = self.mean_coronary_flow_ml_s()
        r_d = (
            self.p_ref_mmhg / max(q_mean, 1e-6)
            if self.outlet_r_distal_mmhg_s_per_ml is None
            else float(self.outlet_r_distal_mmhg_s_per_ml)
        )
        r_p = (
            self.outlet_proximal_fraction * r_d
            if self.outlet_r_proximal_mmhg_s_per_ml is None
            else float(self.outlet_r_proximal_mmhg_s_per_ml)
        )
        return r_d, r_p

    def momentum_rho(self) -> float:
        """与 mmHg 制管腔压配对的动量通量/波速有效密度 ρ*。"""
        return rho_for_mmhg_pressure_coupling(self.rho)


class OutletWindkesselState:
    """
    与 coronary_outlet 中 Windkessel ODE 一致的出口状态机。

    二元：P_wk 即出口管腔压。
    三元：P_out = P_wk + R_p·Q_drive。
    """

    def __init__(self, params: BloodFlowParameters):
        self.params = params
        self.is_three_element = params.outlet_windkessel == "3wk"
        self.r_d, self.r_p = params.resolve_outlet_resistances()
        self.c = max(params.outlet_compliance_ml_per_mmhg, 1e-9)
        q_mean = params.mean_coronary_flow_ml_s()
        self.p_wk = q_mean * self.r_d  # p_wk初始值
        self.p_lumen = self.p_wk + (
            self.r_p * q_mean if self.is_three_element else 0.0
        )

    def reset(
        self,
        p_wk: float | None = None,
        q_drive: float | None = None,
    ) -> None:
        """重置出口 Windkessel；P_wk 默认稳态标定，三元模型 P_lumen 用 q_drive。"""
        q_mean = self.params.mean_coronary_flow_ml_s()
        if q_drive is None:
            q_drive = float(
                np.asarray(self.params.inlet_flow(0.0)).reshape(-1)[0]
            )
        else:
            q_drive = float(q_drive)
        self.p_wk = q_mean * self.r_d if p_wk is None else float(p_wk)
        self.p_lumen = (
            self.p_wk + self.r_p * q_drive if self.is_three_element else self.p_wk
        )

    def lumen_pressure(self, q_drive: float) -> float:
        if self.is_three_element:
            return self.p_wk + self.r_p * q_drive
        return self.p_wk

    def outlet_flow_from_state(self) -> float:
        """微循环侧出流 Q_venous = P_wk/R_d（不等于管腔 Q(L)）。"""
        return self.p_wk / max(self.r_d, 1e-9)

    def apply_substep(self, q_lumen: float) -> tuple[float, float]:
        """SSP-RK 子步：固定 Windkessel 状态，返回 (Q_venous, P_lumen)。"""
        q_drive = float(q_lumen)
        q_venous = self.outlet_flow_from_state()
        return q_venous, self.lumen_pressure(q_drive)

    def advance(self, dt: float, q_lumen: float) -> tuple[float, float]:
        """完整时间步末：推进 C·dP_wk/dt = Q_drive − P_wk/R_d，Q_drive = Q(L)。"""
        if dt <= 0.0:
            return self.apply_substep(q_lumen)
        q_drive = float(q_lumen)
        self.p_wk += (dt / self.c) * (
            q_drive - self.p_wk / max(self.r_d, 1e-9)
        )
        q_venous = self.outlet_flow_from_state()
        self.p_lumen = self.lumen_pressure(q_drive)
        return q_venous, self.p_lumen


# ---------------------------------------------------------------------------
# 主求解器
# ---------------------------------------------------------------------------
class NavierStokes1D:
    """
    一维冠脉血流求解器。

    Parameters
    ----------
    area : ndarray
        参考管腔截面积 (cm²)
    vessel_length_cm : float
        血管长度 L (cm)
    n_nodes : int
        轴向网格节点数（含 x=0 与 x=L）
    parameters : BloodFlowParameters, optional
        物性与边界参数
    """

    def __init__(
        self,
        area: np.ndarray | list,
        vessel_length_cm: float,
        n_nodes: int,
        parameters: BloodFlowParameters | None = None,
    ):
        if n_nodes < 3:
            raise ValueError("n_nodes 至少为 3")
        self.length = float(vessel_length_cm)
        self.nx = int(n_nodes)
        self.paras = parameters if parameters is not None else BloodFlowParameters()

        self.x = np.linspace(0.0, self.length, self.nx)
        self.dx = self.length / (self.nx - 1)

        self.area_ref = np.asarray(area)  # A₀(x) 参考管腔面积 (cm²)
        self.beta = np.full(self.nx, self.paras.beta_mmhg_per_sqrt_cm) 
        self.state = np.zeros((2, self.nx))  # U = [A(x, t), Q(x, t)] 状态变量
        self.pressure = np.zeros(self.nx)  # p(x,t) 管腔标量压 (mmHg)，沿 x 各点取值；非方向性矢量

        self.outlet = OutletWindkesselState(self.paras)  # 出口状态
        self.time = 0.0
        self._outlet_dt = 0.0

        self.history_time: np.ndarray | None = None
        self.history_area: np.ndarray | None = None
        self.history_flow: np.ndarray | None = None
        self.history_pressure: np.ndarray | None = None

        self.area_ref = np.maximum(self.area_ref, 1e-8)

        # 判断是否有狭窄
        self.prox_idx, self.dist_idx = select_reference_indices(self.area_ref)
        lesions = detect_stenoses(self.area_ref, self.prox_idx, self.dist_idx)
        # pre_prox_mask = (self.x <= self.x[self.prox_idx])
        # self.area_ref[pre_prox_mask] = self.area_ref[self.prox_idx]
        if lesions:
            print(f"Detect {len(lesions)} stenoses. ")
            for lesion in lesions:
                mask = (self.x >= self.x[lesion.i0]) & (self.x <= self.x[lesion.i1])
                b_scale = self.area_ref[self.prox_idx] / self.area_ref[lesion.mla_idx]
                self.beta[mask] *= b_scale

        # 初值：A=A₀；Q=Q_in(0)；P_wk=Q_mean·R_d（R_d 用 outlet 构造时的标定值，勿在抬高 p_ref 后再 resolve）
        q0 = float(np.asarray(self.paras.inlet_flow(0.0)).reshape(-1)[0])
        q_mean = self.paras.mean_coronary_flow_ml_s()
        self.outlet.reset(p_wk=q_mean * self.outlet.r_d, q_drive=q0)
        if self.outlet.is_three_element:
            # tube law 锚点 = mean 流量下出口管腔压 P_wk + R_p·Q_mean，与 Windkessel 稳态自洽
            self.paras.p_ref_mmhg = self.outlet.p_wk + self.outlet.r_p * q_mean

        self.state[0] = self.area_ref.copy()
        self.state[1] = q0
        self._update_pressure()

    def _update_pressure(self, state: np.ndarray | None = None) -> np.ndarray:
        """由当前 A 经 tube law 更新沿程管腔标量压 p(x)；p 不单独求解守恒方程。"""
        u = self.state if state is None else state
        self.pressure = lumen_pressure_mmhg(
            u[0], self.area_ref, self.beta, self.paras.p_ref_mmhg
        )
        return self.pressure

    def _spatial_operator(self, state: np.ndarray) -> np.ndarray:
        """有限体积右端项：−∂F/∂x + S。"""
        # TVD限制器下的MUSCL重构，左状态、右状态值
        u_l, u_r = _muscl_states(state, self.area_ref)
        # 内部网格点间通量
        n_if = self.nx - 1
        
        # HLL Riemann求解器计算界面通量
        j = np.arange(n_if)
        rho_star = self.paras.momentum_rho()
        f_inner = _hll_flux(
            u_l,
            u_r,
            self.beta[j],
            self.beta[j + 1],
            self.area_ref[j],
            self.area_ref[j + 1],
            rho_star,
            self.paras.alpha,
            self.paras.p_ref_mmhg,
        )
        f_face = np.zeros((2, self.nx + 1))
        f_face[:, 1:-1] = f_inner
        # 边界通量计算
        f_face[:, 0] = _flux_vector(
            state[0, 0],
            state[1, 0],
            self.area_ref[0],
            self.beta[0],
            rho_star,
            self.paras.alpha,
            self.paras.p_ref_mmhg,
        ).ravel()
        f_face[:, -1] = _flux_vector(
            state[0, -1],
            state[1, -1],
            self.area_ref[-1],
            self.beta[-1],
            rho_star,
            self.paras.alpha,
            self.paras.p_ref_mmhg,
        ).ravel()

        dudt = -(f_face[:, 1:] - f_face[:, :-1]) / self.dx
        a = np.maximum(state[0], 1e-12)
        dudt[1] -= 8.0 * np.pi * self.paras.mu * state[1] / a
        return dudt

    def _apply_boundaries(
        self, state: np.ndarray, advance_outlet: bool = False
    ) -> np.ndarray:
        u = state.copy()  #[A, Q]
        q_in = float(np.asarray(self.paras.inlet_flow(self.time)).reshape(-1)[0])

        # u[0, 0] = u[0, 1]
        u[1, 0] = q_in

        q_lumen = float(u[1, -1])  # x=L 处管腔流量
        if advance_outlet and self._outlet_dt > 0.0:
            _, p_out = self.outlet.advance(self._outlet_dt, q_lumen)
        else:
            _, p_out = self.outlet.apply_substep(q_lumen)

        # 出口：Windkessel + tube law 闭合 A(L)；Q(L) 保留管腔 PDE 值，不强制为 P_wk/R_d
        u[0, -1] = area_from_lumen_pressure_mmhg(
            p_out,
            self.area_ref[-1],
            self.beta[-1],
            self.paras.p_ref_mmhg,
        )
        return u

    @staticmethod
    def _clip_state(state: np.ndarray, area_ref: np.ndarray) -> np.ndarray:
        u = state.copy()
        u[0] = np.maximum(u[0], 0.5 * area_ref)
        # u[1] = np.maximum(u[1], 0.0)
        return u

    def _stable_timestep(self, state: np.ndarray) -> float:
        a = np.maximum(state[0], 1e-12)
        u = state[1] / a
        c = _wave_speed(a, self.beta, self.paras.momentum_rho())
        lam = float(np.max(np.abs(u) + c))
        if lam < 1e-14:
            return 1e-4
        return min(self.paras.cfl * self.dx / lam, self.paras.dt_max_s)

    def _ssp_rk2_step(self, dt: float) -> None:
        self._outlet_dt = dt
        u0 = self._apply_boundaries(self.state)
        k0 = self._spatial_operator(u0)
        u1 = self._clip_state(u0 + dt * k0, self.area_ref)
        u1 = self._apply_boundaries(u1)
        k1 = self._spatial_operator(u1)
        self.state = self._clip_state(
            0.5 * u0 + 0.5 * (u1 + dt * k1), self.area_ref
        )
        self.state = self._apply_boundaries(self.state, advance_outlet=True)
        self._update_pressure()

    def run(
        self,
        duration_s: float,
        record_interval_steps: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        时间推进至 duration_s。

        Returns
        -------
        area_hist, flow_hist, pressure_hist, time_hist
        """
        t_hist = [0.0]
        a_hist = [self.state[0].copy()]
        q_hist = [self.state[1].copy()]
        p_hist = [self.pressure.copy()]
        step = 0

        print(
            f"Navier-Stokes 1D: L={self.length:.2f} cm, nx={self.nx}, "
            f"T={duration_s:.3f} s, outlet={self.paras.outlet_windkessel}"
        )

        t0 = time.perf_counter()
        while self.time < duration_s - 1e-15:
            print(
                f"current time: {self.time:.6f} s / Total: {duration_s:.2f} s",
                flush=True,
                end="\r",
            )
            dt = self._stable_timestep(self.state)
            if self.time + dt > duration_s:
                dt = duration_s - self.time
            self._ssp_rk2_step(dt)
            self.time += dt
            step += 1
            if step % record_interval_steps == 0:
                t_hist.append(self.time)
                a_hist.append(self.state[0].copy())
                q_hist.append(self.state[1].copy())
                p_hist.append(self.pressure.copy())


        self.history_time = np.array(t_hist)
        self.history_area = np.array(a_hist)
        self.history_flow = np.array(q_hist)
        self.history_pressure = np.array(p_hist)
        t1 = time.perf_counter()
        print(f"完成 {step} 步, t={self.time:.4f} s, 耗时 {t1-t0:.3f} s")
        return (
            self.history_area,
            self.history_flow,
            self.history_pressure,
            self.history_time,
        )

    def ffr_ratio(self) -> float:
        """简化 FFR：远端/近端时间平均管腔压之比。"""
        if self.history_pressure is None:
            p_mean = self.pressure
        else:
            p_mean = np.mean(self.history_pressure, axis=0)
        return p_mean / p_mean[0]
    
    def plot_results(self):
        assert self.history_time is not None, "请先调用 run()"
        assert self.history_area is not None, "请先调用 run()"
        assert self.history_flow is not None, "请先调用 run()"
        assert self.history_pressure is not None, "请先调用 run()"

        import matplotlib.pyplot as plt

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
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
        tt, xx = self.history_time, self.x

        ax = axes[0, 0]
        cf = ax.contourf(tt, xx, self.history_area.T, levels=30, cmap="viridis")
        ax.set(xlabel="t (s)", ylabel="x (cm)", title="A(t, x) [cm²]")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[0, 1]
        q_mean = np.mean(self.history_flow[3000:, :], axis=0)
        ax.plot(xx, q_mean, "g-", lw=1.8, label="Q")
        ax.set(xlabel="x (cm)", ylabel="Q (cm³/s)", title="沿程流量（时间平均）")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(tt, self.history_pressure[:, 348], "r-", lw=1.5, label="x=348")
        ax.plot(tt, self.history_pressure[:, 0], "b-", lw=1.5, label="x=0")
        ax.set(xlabel="t (s)", ylabel="p(mmHg)", title="x = 0(blue) / 300(red)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        ax = axes[1, 1]
        ax.plot(xx, self.history_area[-1], "b-", lw=1.5, label="A")
        ax.set(xlabel="x (cm)", ylabel="A (cm²)", title=f"t = {tt[-1]:.3f} s")
        axr = ax.twinx()
        axr.plot(xx, self.history_pressure[-1], "g--", lw=1.5, label="p")
        axr.set_ylabel("p (mmHg)")
        lines_l, labels_l = ax.get_legend_handles_labels()
        lines_r, labels_r = axr.get_legend_handles_labels()
        ax.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def plot_results_1(self, show: bool = True, renderer: str | None = None):
        """
        与 plot_results 相同的四幅图，使用 Plotly 在浏览器中交互显示。

        Parameters
        ----------
        show : bool
            是否调用 fig.show() 打开浏览器
        renderer : str, optional
            传给 plotly fig.show(renderer=...)，如 "browser"
        """
        if self.history_time is None:
            raise RuntimeError("请先调用 run()")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        assert self.history_time is not None, "请先调用 run()"
        assert self.history_area is not None, "请先调用 run()"
        assert self.history_flow is not None, "请先调用 run()"
        assert self.history_pressure is not None, "请先调用 run()" 

        tt = self.history_time
        xx = self.x
        t_end = float(tt[-1])

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "A(t, x) [cm²]",
                "流量FFR（时间平均）",
                "x = 近端(blue) / 远端(red)",
                f"t = {t_end:.3f} s",
            ),
            specs=[[{}, {}], [{}, {"secondary_y": True}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        fig.add_trace(
            go.Heatmap(
                x=tt,
                y=xx,
                z=self.history_area.T,
                colorscale="Viridis",
                colorbar=dict(title="A [cm²]", len=0.45, y=0.78),
                hovertemplate="t=%{x:.4f} s<br>x=%{y:.3f} cm<br>A=%{z:.4f} cm²<extra></extra>",
            ),
            row=1,
            col=1,
        )

        p_hist = self.history_pressure
        p_slice = p_hist[200:, :] if p_hist.shape[0] > 200 else p_hist
        p_mean = np.mean(p_slice, axis=0)
        ffr = np.clip(p_mean / p_mean[0], 0.0, 1.0)
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=ffr,
                mode="lines",
                name="FFR",
                line=dict(color="green", width=2),
                hovertemplate="x=%{x:.3f} cm<br>FFR=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        idx_far = min(348, self.nx - 1)
        idx_near = 0
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=self.history_flow[:, idx_near],
                mode="lines",
                name="x=0",
                line=dict(color="blue", width=1.5),
                hovertemplate="t=%{x:.4f} s<br>Q=%{y:.3f} cm³/s<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=self.history_flow[:, idx_far],
                mode="lines",
                name=f"x={idx_far}",
                line=dict(color="red", width=1.5),
                hovertemplate="t=%{x:.4f} s<br>Q=%{y:.3f} cm³/s<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=self.history_area[-1],
                mode="lines",
                name="A",
                line=dict(color="blue", width=1.5),
                hovertemplate="x=%{x:.3f} cm<br>A=%{y:.4f} cm²<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=self.history_pressure[-1],
                mode="lines",
                name="p",
                line=dict(color="green", width=1.5, dash="dash"),
                hovertemplate="x=%{x:.3f} cm<br>p=%{y:.3f} mmHg<extra></extra>",
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        fig.update_xaxes(title_text="t (s)", row=1, col=1)
        fig.update_yaxes(title_text="x (cm)", row=1, col=1)
        fig.update_xaxes(title_text="x (cm)", row=1, col=2)
        fig.update_yaxes(title_text="FFR", row=1, col=2)
        fig.update_xaxes(title_text="t (s)", row=2, col=1)
        fig.update_yaxes(title_text="Q (cm³/s)", row=2, col=1)
        fig.update_xaxes(title_text="x (cm)", row=2, col=2)
        fig.update_yaxes(title_text="A (cm²)", row=2, col=2)
        fig.update_yaxes(title_text="p (mmHg)", row=2, col=2, secondary_y=True)

        fig.update_layout(
            height=1000,
            width=1000,
            title_text="Navier-Stokes 1D 结果",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")

        if show:
            show_kw = {"renderer": renderer} if renderer is not None else {}
            fig.show(**show_kw)
        return fig


def demo():
    from scipy.ndimage import gaussian_filter1d
    np.random.seed(2034)
    sigma_nodes = 4   # 按网格点，2–5 试起
    pullback_speed = 20  # mm/s
    frames_per_s = 200 # fps
    pullback_time = 2.7  # s
    nx = 400
    # nx = int(pullback_time * frames_per_s)
    length = pullback_speed * pullback_time / 10  # cm
    
    
    duration_s = 3  # s
    area_file = np.load('data/pred_masks20260721/area.npy')
    area = area_file[::-1] / 100 # / 20   # 冠脉管腔横截面积约为 1.8 -9 mm² (小于3可能就算狭窄了)
    area_smooth = gaussian_filter1d(area, sigma=sigma_nodes, mode="nearest")
    nx = area.shape[0]
    length = 0.3 * area.shape[0] / 10 
    area = np.full((nx, ), 0.07) # + np.random.random((nx, )) * 0.01
    print(nx, length)

    parameters = BloodFlowParameters()
    solver = NavierStokes1D(area_smooth, length, nx, parameters)
    solver.run(duration_s=duration_s, record_interval_steps=30)
    solver.plot_results_1()

    assert solver.history_time is not None, "请先调用 run()"
    assert solver.history_area is not None, "请先调用 run()"
    assert solver.history_flow is not None, "请先调用 run()"
    assert solver.history_pressure is not None, "请先调用 run()"

    q_in  = np.mean(solver.history_flow[:, 0])
    q_out = np.mean(solver.history_flow[:, -1])
    diff = np.mean(solver.history_flow[:, 0] - solver.history_flow[:, -1])
    print(f"Q_in={q_in:.3f}, Q_out={q_out:.3f}, diff={diff:.3f}")
    # 若 diff 长期显著 > 0，压力下降很可能是质量失衡

    p_mean = np.nanmean(solver.history_pressure[200:, :], axis=0)
    ffr = p_mean / p_mean[0]
    ffr = np.clip(ffr, 0, 1)
    return ffr

if __name__ == "__main__":
    ffr = demo()
    print(f"max ffr: {np.max(ffr)}\n min ffr: {np.min(ffr)}")
