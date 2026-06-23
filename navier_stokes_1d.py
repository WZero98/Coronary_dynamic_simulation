"""
navier_stokes_1d.py — 冠状动脉一维轴向血流模拟

依据一维可变形管内的守恒型 Navier-Stokes 方程组，耦合管壁弹性压力关系，
对沿血管轴向 x ∈ [0, L] 的截面积 A 与体积流量 Q 进行数值求解。

控制方程（守恒形式，长度/面积/流量用 CGS，管腔压 p 用 mmHg）
------------------------------------------------------------
    ∂U/∂t + ∂F/∂x = S,    U = [A, Q]ᵀ

    F₀ = Q
    F₁ = α·Q²/A + p_mmHg(A)·A/ρ*
    S  = [0, −8πμ·Q/A − K(x)·ρ·Q|Q|/(2A·ρ*)]ᵀ

    狭窄段 K(x) > 0：基于 Bernoulli 型局部形阻（见 docs/navier_stokes_1d.md §3.4）。

    ρ* = ρ / (1333.22 g·cm⁻¹·s⁻² per mmHg)，在 BloodFlowParameters 中一次性标定；
    与逐点把 p 换成 dyne/cm² 再代入原式等价。

管壁弹性（tube law，mmHg）
--------------------------
    p(A, x) = P_ref + β(x)·(√A − √A₀(x))

    A₀(x) 由用户给定的沿程管腔参考面积描述；β(x) 为管壁刚度（可常数或随 x 给定）。

边界条件
--------
- 入口：coronary_inlet.coronary_inlet_flow 提供生理脉动入口流量 Q_in(t)（mL/s ≡ cm³/s）
- 出口：coronary_outlet 中 Windkessel 模型提供出口管腔压；求解过程中按
        C·dP_wk/dt = Q_drive − P_wk/R_d 推进微循环状态（与 windkessel2/3 一致），
        再由 tube law 确定出口 A，流量与管腔/微循环闭合

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
from coronary_outlet import (
    windkessel2_outlet_pressure,
    windkessel3_outlet_pressure,
)


# ---------------------------------------------------------------------------
# 管壁弹性律
# ---------------------------------------------------------------------------
def lumen_pressure_mmhg(
    area: np.ndarray,
    area_ref: np.ndarray,
    beta: np.ndarray,
    p_ref: float = P_INLET_REF_MMHG,
) -> np.ndarray:
    """p = P_ref + β(√A − √A₀)。"""
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    a0 = np.asarray(area_ref, dtype=float)
    b = np.asarray(beta, dtype=float)
    return p_ref + b * (np.sqrt(a) - np.sqrt(a0))


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
    return np.maximum(sqrt_a**2, 0.1 * a0)


def stenosis_loss_k_total(area_scale: float) -> float:
    """
    由狭窄几何比估算总形阻系数 K（无量纲）。

    基于收缩断面处速度头损失 Δp = K·ρv²/2，取 K ≈ (A_prox/A_sten − 1)²，
    其中 A_sten/A_prox = area_scale（lesions 中对 A₀ 的缩放因子）。
    """
    scale = float(np.clip(area_scale, 1e-3, 1.0))
    return float((1.0 / scale - 1.0) ** 2)


def stenosis_form_loss_source(
    flow: np.ndarray,
    area: np.ndarray,
    loss_k_per_cm: np.ndarray,
    rho_g_per_cm3: float,
    rho_mmhg: float,
) -> np.ndarray:
    """
    狭窄区形阻动量源项（与 Poiseuille 摩擦并列加入 S₁）。

    将总损失 Δp = K_total·ρv²/2 沿狭窄段长度 L 均布：
        (dp_loss/dx) = K_total·ρ·(Q/A)² / (2L)
    等价动量源（与 tube-law / mmHg 动量通量同一 ρ* 标定）：
        S_sten = −(A/ρ*)·(dp_loss/dx) = −K(x)·ρ·Q|Q| / (2A·ρ*)
    其中 K(x) = K_total / L（单位 1/cm）。
    """
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    q = np.asarray(flow, dtype=float)
    k = np.asarray(loss_k_per_cm, dtype=float)
    return -k * rho_g_per_cm3 * np.abs(q) * q / (2.0 * a * rho_mmhg)


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
    """单元界面 j+½ 的左右重构状态（内部界面）。"""
    nvar, nx = U.shape  # nvar 为未知函数量，nx 为划分单元数
    slope = np.zeros_like(U)
    if nx >= 3:
        slope[:, 1:-1] = _minmod(U[:, 1:-1] - U[:, :-2], U[:, 2:] - U[:, 1:-1])
    # 左状态计算，来自当前单元
    u_left = U[:, :-1] + 0.5 * slope[:, :-1]
    # 右状态计算，来自下一个单元
    u_right = U[:, 1:] - 0.5 * slope[:, 1:]
    floor = 0.1 * area_ref
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
    cfl: float = 1.0  # 数值稳定性参数 CFL
    dt_max_s: float = 5e-4  # 时间步上限 (s)

    # 管壁相关 tube law
    p_ref_mmhg: float = P_INLET_REF_MMHG  # tube law 参考压 (mmHg)
    beta_mmhg_per_sqrt_cm: float = 3250  # 默认β，越大表示管腔越硬（管壁刚度）
    stenosis_loss_coefficient: float = 1.0  # 狭窄形阻系数 K 的全局标定因子

    # 冠脉血流入口相关
    heart_rate_bpm: float = 75.0  # 心率 (bpm)
    cardiac_output_l_per_min: float = 5.0  # 心脏输出量 (L/min)
    coronary_flow_fraction: float = 0.03  # 左冠脉血流占心脏输出量的比例
    inlet_min_flow_fraction: float = 0.1  # 流量下限，即最小值占整个流量平均值的比例
    inlet_phase_offset_rad: float = 0.0  # 波形相位偏移
    inlet_fourier_coefficients: Sequence[tuple[int, float, float]] | None = None  # 自定义谐波（傅里叶变换的系数）；None 用模块内置默认系数

    # 出口 Windkesse模型 相关
    outlet_windkessel: Literal["2wk", "3wk"] = "3wk"
    outlet_r_distal_mmhg_s_per_ml: float | None = None  # Rd 远端阻力系数
    outlet_r_proximal_mmhg_s_per_ml: float | None = None  # Rp 近端阻力系数
    outlet_compliance_ml_per_mmhg: float = 0.08  # 血管顺应性，越大顺应性越好 
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


class _OutletWindkesselState:
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

    def reset(self, p_wk: float | None = None):
        if p_wk is not None:
            self.p_wk = float(p_wk)
        self.p_lumen = self.p_wk

    def lumen_pressure(self, q_drive: float) -> float:
        if self.is_three_element:
            return self.p_wk + self.r_p * q_drive
        return self.p_wk

    def outlet_flow_from_state(self) -> float:
        return self.p_wk / max(self.r_d, 1e-9)

    def apply_substep(self, q_lumen: float, q_inlet: float = 0.0) -> tuple[float, float]:
        """SSP-RK 子步：固定 Windkessel 状态，返回 (Q_out, P_lumen)。"""
        q_drive = 0.5 * (float(q_lumen) + float(q_inlet))
        # q_drive = float(q_lumen)  # 更强调管腔与微循环质量守恒，只用 q_lumen
        q_out = self.outlet_flow_from_state()
        return q_out, self.lumen_pressure(q_drive)

    def advance(self, dt: float, q_lumen: float, q_inlet: float = 0.0) -> tuple[float, float]:
        """完整时间步末：推进 C·dP_wk/dt = Q_drive − P_wk/R_d。"""
        if dt <= 0.0:
            return self.apply_substep(q_lumen, q_inlet)
        q_drive = 0.5 * (float(q_lumen) + float(q_inlet))
        # q_drive = float(q_lumen)  # 更强调管腔与微循环质量守恒，只用 q_lumen
        self.p_wk += (dt / self.c) * (
            q_drive - self.p_wk / max(self.r_d, 1e-9)
        )
        q_out = self.outlet_flow_from_state()
        self.p_lumen = self.lumen_pressure(q_drive)
        return q_out, self.p_lumen


# ---------------------------------------------------------------------------
# 主求解器
# ---------------------------------------------------------------------------
class NavierStokes1D:
    """
    一维冠脉血流求解器。

    Parameters
    ----------
    vessel_length_cm : float
        血管长度 L (cm)
    n_nodes : int
        轴向网格节点数（含 x=0 与 x=L）
    parameters : BloodFlowParameters, optional
        物性与边界参数
    """

    def __init__(
        self,
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

        self.area_ref = np.full(self.nx, 0.5)  # TODO 需修改为管腔分割后计算出的管腔截面积
        self.beta = np.full(self.nx, self.paras.beta_mmhg_per_sqrt_cm)  # TODO 识别到斑块处，弹性降低，β越大
        self.stenosis_loss_k = np.zeros(self.nx)  # 狭窄形阻强度 K(x)，单位 1/cm
        self.state = np.zeros((2, self.nx))  # U = [A(x, t), Q(x, t)] 状态变量
        self.pressure = np.zeros(self.nx)  # P(x, t) 管腔壁所受侧压力

        self.outlet = _OutletWindkesselState(self.paras)  # 出口状态
        self.time = 0.0
        self._outlet_dt = 0.0

        self.history_time: np.ndarray | None = None
        self.history_area: np.ndarray | None = None
        self.history_flow: np.ndarray | None = None
        self.history_pressure: np.ndarray | None = None

    def set_lumen_area_profile(
        self,
        area: np.ndarray,
        x: np.ndarray | None = None,
        beta: np.ndarray | float | None = None,
        lesions: Sequence[tuple[float, float, float, float]] | None = None,
    ) -> None:
        """
        设置 [0, L] 上的参考管腔面积 A₀(x)。

        Parameters
        ----------
        area : ndarray
            参考截面积 (cm²)
        x : ndarray, optional
            与 area 对应的轴向位置 (cm)；缺省则在 [0, L] 上均匀采样
        beta : ndarray or float, optional
            弹性系数 β(x) (mmHg/√cm)
        lesions : optional
            狭窄列表 [(x_start, x_end, area_scale, beta_scale), ...]；
            同时按 §docs/navier_stokes_1d.md 填充 stenosis_loss_k 形阻场
        """
        area = np.asarray(area, dtype=float)
        if area.size < 2:
            raise ValueError("area 至少 2 个点")
        if x is None:
            x = np.linspace(0.0, self.length, area.size)
        else:
            x = np.asarray(x, dtype=float)
            if x.size != area.size:
                raise ValueError("x 与 area 长度须一致")

        if x[0] > 1e-9:
            x = np.concatenate(([0.0], x))
            area = np.concatenate(([area[0]], area))
        if x[-1] < self.length - 1e-9:
            x = np.concatenate((x, [self.length]))
            area = np.concatenate((area, [area[-1]]))

        self.area_ref = np.interp(self.x, x, area)
        self.area_ref = np.maximum(self.area_ref, 1e-8)

        if beta is None:
            self.beta = np.full(self.nx, self.paras.beta_mmhg_per_sqrt_cm)
        elif np.ndim(beta) == 0:
            self.beta = np.full(self.nx, float(beta))
        else:
            beta = np.asarray(beta, dtype=float)
            bx = x if beta.size == area.size else self.x
            self.beta = np.interp(self.x, bx, beta)

        self.stenosis_loss_k = np.zeros(self.nx)
        if lesions:
            loss_scale = self.paras.stenosis_loss_coefficient
            for x0, x1, a_scale, b_scale in lesions:
                mask = (self.x >= x0) & (self.x <= x1)
                self.area_ref[mask] *= a_scale
                self.beta[mask] *= b_scale
                lesion_length = max(float(x1) - float(x0), self.dx)
                k_total = loss_scale * stenosis_loss_k_total(a_scale)
                self.stenosis_loss_k[mask] = k_total / lesion_length

        self.state[0] = self.area_ref.copy()
        self.state[1] = 0.0
        self._update_pressure()  # 管腔各处压力初始值
        self.outlet.reset(p_wk=float(self.pressure[-1]))  # 出口压力初始值

    def _update_pressure(self, state: np.ndarray | None = None) -> np.ndarray:
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
        if np.any(self.stenosis_loss_k > 0.0):
            dudt[1] += stenosis_form_loss_source(
                state[1],
                a,
                self.stenosis_loss_k,
                self.paras.rho,
                rho_star,
            )
        return dudt

    def _apply_boundaries(
        self, state: np.ndarray, advance_outlet: bool = False
    ) -> np.ndarray:
        u = state.copy()
        q_in = float(np.asarray(self.paras.inlet_flow(self.time)).reshape(-1)[0])

        u[0, 0] = u[0, 1]
        u[1, 0] = q_in

        if advance_outlet and self._outlet_dt > 0.0:
            q_out, p_out = self.outlet.advance(
                self._outlet_dt, u[1, -1], q_in
            )
        else:
            q_out, p_out = self.outlet.apply_substep(u[1, -1], q_in)

        u[1, -1] = q_out
        u[0, -1] = area_from_lumen_pressure_mmhg(
            p_out,
            self.area_ref[-1],
            self.beta[-1],
            self.paras.p_ref_mmhg,
        )
        return u

    @staticmethod
    def _clip_area(state: np.ndarray, area_ref: np.ndarray) -> np.ndarray:
        u = state.copy()
        u[0] = np.maximum(u[0], 0.1 * area_ref)
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
        u1 = self._clip_area(u0 + dt * k0, self.area_ref)
        u1 = self._apply_boundaries(u1)
        k1 = self._spatial_operator(u1)
        self.state = self._clip_area(
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

    @staticmethod
    def reference_outlet_pressure(
        times: np.ndarray,
        parameters: BloodFlowParameters,
        q_override: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        调用 coronary_outlet 离线计算参考出口压力曲线 P_out(t)（mmHg），
        用于与耦合求解结果对比，不参与时间推进。
        """
        t = np.asarray(times, dtype=float)
        if q_override is None:
            q = parameters.inlet_flow(t)
        else:
            q = np.asarray(q_override, dtype=float)
        wk_kw = dict(
            r_distal_mmhg_s_per_ml=parameters.outlet_r_distal_mmhg_s_per_ml,
            compliance_ml_per_mmhg=parameters.outlet_compliance_ml_per_mmhg,
        )
        if parameters.outlet_windkessel == "2wk":
            return windkessel2_outlet_pressure(t, q, **wk_kw)
        p_out, p_wk = windkessel3_outlet_pressure(
            t,
            q,
            r_proximal_mmhg_s_per_ml=parameters.outlet_r_proximal_mmhg_s_per_ml,
            proximal_fraction=parameters.outlet_proximal_fraction,
            **wk_kw,
        )
        return p_out, p_wk

    def plot_results(self):
        if self.history_time is None:
            raise RuntimeError("请先调用 run()")
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
        q_mean = np.mean(self.history_flow[1500:, :], axis=0)
        ax.plot(xx, q_mean, "g-", lw=1.8, label="p tube law")
        ax.set(xlabel="x (cm)", ylabel="mmHg", title="沿程流量（时间平均）")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(tt, self.history_pressure[:, 300], "r-", lw=1.5, label="x=300 p")
        ax.plot(tt, self.history_pressure[:, 10], "b-", lw=1.5, label="x=10 p")
        ax.set(xlabel="t (s)", ylabel="p(mmHg)", title="x = 10(blue) / 300(red)")
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


def _demo():
    from scipy.ndimage import gaussian_filter1d
    np.random.seed(569)
    sigma_nodes = 2.0   # 按网格点，2–5 试起
    pullback_speed = 36  # mm/s
    frames_per_s = 200 # fps
    pullback_time = 2  # s
    # nx = int(pullback_time * frames_per_s)
    length = pullback_speed * pullback_time / 10  # cm
    duration_s = 3  # s
    area_file = np.load('area.npy')
    area = area_file[::-1] / 100
    area_smooth = gaussian_filter1d(area, sigma=sigma_nodes, mode="nearest")
    nx = area.shape[0]
    x = np.linspace(0, length, nx)
    # area = np.full((nx, ), 0.53) + np.random.random((nx, )) * 0.01
    print(nx, length)

    # lesions = [(3, 4, 0.95, 10), (7, 8, 0.95, 10)]
    lesions = None

    par = BloodFlowParameters()
    solver = NavierStokes1D(length, nx, par)
    solver.set_lumen_area_profile(
        area_smooth,
        x=x,
        lesions=lesions,
    )
    solver.run(duration_s=duration_s, record_interval_steps=30)
    solver.plot_results()

    q_mean = np.nanmean(solver.history_flow, axis=0)
    ffr = q_mean / par.mean_coronary_flow_ml_s()
    return ffr

if __name__ == "__main__":
    ffr = _demo()
    print(f"max ffr: {np.max(ffr)}\n min ffr: {np.min(ffr)}")
