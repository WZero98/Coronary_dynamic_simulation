"""
navier_stokes_1d.py — 冠状动脉一维轴向血流模拟

依据一维可变形管内的守恒型 Navier-Stokes 方程组，耦合管壁弹性压力关系，
对沿血管轴向 x ∈ [0, L] 的截面积 A 与体积流量 Q 进行数值求解。

控制方程（长度/面积/流量用 CGS，管腔压 p 用 mmHg）
------------------------------------------------------------
    ∂U/∂t + ∂F/∂x = S,    U = [A, Q]ᵀ

    F₀ = Q
    F₁ = α·Q²/A                         （仅对流；不含 pA/ρ*）
    S  = [0, −(A/ρ*)·∂p/∂x − 8πμ·Q/A]ᵀ

    等价于文献标准式：
        ∂Q/∂t + ∂(α Q²/A)/∂x + (A/ρ*) ∂p/∂x = −8πμ Q/A

    注意：旧版曾写 F₁ = α Q²/A + p A/ρ* 且无 (p/ρ*)∂A/∂x 源项，
    与上式不等价，变截面处会产生虚假压力梯度（见 docs/navier_stokes_1d.md §3.1.1）。

    ρ* = ρ / (1333.22 g·cm⁻¹·s⁻² per mmHg)，在 BloodFlowParameters 中一次性标定；
    与逐点把 p 换成 dyne/cm² 再代入原式等价。

管壁弹性（tube law，mmHg）
--------------------------
    p(A, x) = P_ref + β(x)·(√A − √A₀(x))

    p 为管腔标量压（各向同性，非轴向/径向矢量分量）。tube law 将其与截面面积 A 耦合
    （径向胀缩）；动量源项 −(A/ρ*)∂p/∂x 体现沿 x 的压力梯度对轴向流量 Q 的驱动。

    A₀(x) 由用户给定的沿程管腔参考面积描述；β(x) 为管壁刚度（可常数或随 x 给定）。

边界条件
--------
- 入口：coronary_inlet.coronary_inlet_flow 提供生理脉动入口流量 Q_in(t)（mL/s ≡ cm³/s）
- 侧支：各开口接 0D Windkessel；由主支开口管腔压 P_ostium 反解
        Q_b=(P_ostium−P_wk)/R_coup（3wk 用 R_p，2wk 用 R_d），
        连续方程质量汇 −Q_b/Δx；Murray 份额仅用于标定各支 R、C（flow_share）
- 出口：主支远端 Windkessel；C·dP_wk/dt = Q(L) − P_wk/R_d，再反解 A(L)

初值（构造 `NavierStokes1D` 时完成）
------------------------------------
- A(x,0) = A₀(x)（经近/远端参考段平整；狭窄段自动放大 β）
- Q(x,0) 按 Murray 份额沿程分流剖面
- 主支远端 P_wk(0) = Q_mean·f_distal·R_d
- 3wk（及默认）将 p_ref 对齐到出口稳态管腔压 P_wk+R_p·Q_mean·f_distal，
  使 A=A₀ 时 tube law 与出口 Windkessel 压位一致
- 侧支默认 apply_murray_scale=True（Murray 闭合）；并校验直径单位/是否大于主支

主要输入
--------
1. 与 n_nodes 等长的管腔参考截面积 A₀（cm²）及血管长度 L
2. BloodFlowParameters（物性、入口、出口 Windkessel）
3. geometry.select_reference_indices / detect_stenoses（构造内自动调用）

空间离散：单元中心有限体积法 + MUSCL 线性重构 + minmod TVD 限制器 + HLL 数值通量
时间离散：SSP-RK2（二阶强稳定保持 Runge-Kutta）
时间步长：CFL 自适应
"""
import time
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from branch import prepare_side_branches
from coronary_constants import MINIMUM_AREA_RATIO, P_INLET_REF_MMHG, rho_for_mmhg_pressure_coupling
from coronary_inlet import coronary_inlet_flow
from geometry import detect_stenoses, select_reference_indices, resample_axial_profile


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
    用于 tube law 径向胀缩关系及动量源项 −(A/ρ*)∂p/∂x。
    """
    a0 = np.asarray(area_ref, dtype=float)
    a = np.maximum(np.asarray(area, dtype=float), MINIMUM_AREA_RATIO * a0)
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
    return np.maximum(sqrt_a**2, MINIMUM_AREA_RATIO * a0)

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
    floor = MINIMUM_AREA_RATIO * area_ref
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
    alpha: float,
) -> np.ndarray:
    """物理通量 F(U)：质量通量 Q，动量通量仅对流项 α Q²/A。

    压力梯度不以 pA/ρ* 进入通量（旧错误写法），而由空间算子中的
    −(A/ρ*) ∂p/∂x 源项施加，与标准 1D 血流方程一致。
    """
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    q = np.asarray(flow, dtype=float)
    return np.vstack([q, alpha * q**2 / a])


def _hll_flux(
    u_l: np.ndarray,
    u_r: np.ndarray,
    beta_l: np.ndarray,
    beta_r: np.ndarray,
    rho_mmhg: float,
    alpha: float,
) -> np.ndarray:
    """HLL Riemann 求解器计算界面数值通量（对流部分）。

    波速估计仍用弹性波速 u±c，使 CFL/耗散与压力波尺度一致；
    压力驱动本身在单元中心以 −(A/ρ*)∂p/∂x 处理。
    """
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

    f_l = _flux_vector(a_l, q_l, alpha)
    f_r = _flux_vector(a_r, q_r, alpha)
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
    beta_mmhg_per_sqrt_cm: float = 1000.0  # 默认β，越大表示管腔越硬（管壁刚度）

    # 冠脉血流入口相关
    heart_rate_bpm: float = 75.0  # 心率 (bpm)
    cardiac_output_l_per_min: float = 5.5  # 心脏输出量 (L/min)
    coronary_flow_fraction: float = 0.03  # 左冠脉血流占心脏输出量的比例
    inlet_min_flow_fraction: float = 0.25  # 流量下限，即最小值占整个流量平均值的比例
    inlet_phase_offset_rad: float = 0.0  # 波形相位偏移
    inlet_fourier_coefficients: Sequence[tuple[int, float, float]] | None = None  # 自定义谐波（傅里叶变换的系数）；None 用模块内置默认系数

    # 出口 Windkesse模型 相关
    outlet_windkessel: Literal["2wk", "3wk"] = "3wk"
    outlet_r_distal_mmhg_s_per_ml: float | None = 25  # Rd 远端阻力系数
    outlet_r_proximal_mmhg_s_per_ml: float | None = None  # Rp 近端阻力系数
    outlet_compliance_ml_per_mmhg: float = 0.05  # 血管顺应性，越大顺应性越好 
    outlet_proximal_fraction: float = 0.15  # 三元模型近端阻力比例

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
        """与 mmHg 制管腔压配对的有效密度 ρ*（用于 (A/ρ*)∂p/∂x 与波速）。"""
        return rho_for_mmhg_pressure_coupling(self.rho)


class OutletWindkesselState:
    """
    与 coronary_outlet 中 Windkessel ODE 一致的出口/侧支状态机。

    二元：P_wk 即管腔压。
    三元：P_out = P_wk + R_p·Q_drive。

    flow_share : 该终端占入口平均流量的 Murray 份额；阻力按 1/share 放大，
    顺应性按 share 缩小，使并联终端在均值上与单出口标定一致。

    两种耦合用法
    ------------
    - 主支远端出口：由管腔流量 Q(L) 驱动，反解 P_lumen → 定 A(L)
      （``apply_substep`` / ``advance``）
    - 侧支开口：由主支开口管腔压 P_ostium 驱动，反解分流量 Q_b
      （``apply_substep_from_pressure`` / ``advance_from_pressure``）
      从而使狭窄抬高近端压时，近端侧支自动多分流量。
    """

    def __init__(self, params: BloodFlowParameters, flow_share: float = 1.0):
        self.params = params
        self.is_three_element = params.outlet_windkessel == "3wk"
        share = max(float(flow_share), 1e-6)
        self.flow_share = share
        r_d0, r_p0 = params.resolve_outlet_resistances()
        self.r_d = r_d0 / share
        self.r_p = r_p0 / share
        self.c = max(params.outlet_compliance_ml_per_mmhg * share, 1e-9)

    def reset(
        self,
        p_wk: float | None = None,
        q_drive: float | None = None,
    ) -> None:
        """重置；P_wk 默认与该终端份额的稳态流量对齐。"""
        q_share_mean = self.params.mean_coronary_flow_ml_s() * self.flow_share
        if q_drive is None:
            q_drive = q_share_mean
        else:
            q_drive = float(q_drive)
        self.p_wk = q_share_mean * self.r_d if p_wk is None else float(p_wk)
        self.p_lumen = (
            self.p_wk + self.r_p * q_drive if self.is_three_element else self.p_wk
        )

    def lumen_pressure(self, q_drive: float) -> float:
        if self.is_three_element:
            return self.p_wk + self.r_p * float(q_drive)
        return self.p_wk

    def coupling_resistance(self) -> float:
        """由管腔压反解流量时的串联阻力：3wk 用 R_p，2wk 用 R_d。"""
        if self.is_three_element:
            return max(self.r_p, 1e-9)
        return max(self.r_d, 1e-9)

    def flow_from_lumen_pressure(self, p_lumen: float) -> float:
        """给定开口/出口管腔压，反解进入 Windkessel 的流量 Q_drive。

        3wk: Q = (P_lumen − P_wk) / R_p
        2wk: Q = (P_lumen − P_wk) / R_d
        """
        return (float(p_lumen) - self.p_wk) / self.coupling_resistance()

    def outlet_flow_from_state(self) -> float:
        """微循环侧出流 Q_venous = P_wk/R_d。"""
        return self.p_wk / max(self.r_d, 1e-9)

    def apply_substep(self, q_lumen: float) -> tuple[float, float]:
        """SSP-RK 子步（流量驱动）：冻结 P_wk，返回 (Q_venous, P_lumen)。"""
        q_drive = float(q_lumen)
        return self.outlet_flow_from_state(), self.lumen_pressure(q_drive)

    def advance(self, dt: float, q_lumen: float) -> tuple[float, float]:
        """完整时间步末（流量驱动）：C·dP_wk/dt = Q_drive − P_wk/R_d。"""
        if dt <= 0.0:
            return self.apply_substep(q_lumen)
        q_drive = float(q_lumen)
        self.p_wk += (dt / self.c) * (
            q_drive - self.p_wk / max(self.r_d, 1e-9)
        )
        self.p_lumen = self.lumen_pressure(q_drive)
        return self.outlet_flow_from_state(), self.p_lumen

    def apply_substep_from_pressure(self, p_lumen: float) -> float:
        """SSP-RK 子步（压力驱动）：冻结 P_wk，返回 Q_drive。"""
        q_drive = self.flow_from_lumen_pressure(p_lumen)
        self.p_lumen = float(p_lumen)
        return q_drive

    def advance_from_pressure(self, dt: float, p_lumen: float) -> float:
        """完整时间步末（压力驱动）：由 P_lumen 得 Q，再推进 P_wk。"""
        q_drive = self.flow_from_lumen_pressure(p_lumen)
        if dt > 0.0:
            self.p_wk += (dt / self.c) * (
                q_drive - self.p_wk / max(self.r_d, 1e-9)
            )
            # 推进后用新 P_wk 与同一 P_lumen 再一致化一次 Q
            q_drive = self.flow_from_lumen_pressure(p_lumen)
        self.p_lumen = float(p_lumen)
        return q_drive

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
        area_cm2: np.ndarray | list,
        vessel_length_cm: float,
        n_nodes: int,
        parameters: BloodFlowParameters | None = None,
        branch_frame_indices: Optional[Sequence[int]] = None,
        branch_diameters_cm: Optional[Sequence[float]] = None,
        apply_murray_scale: bool = True,
        align_p_ref_to_outlet: bool = True,
    ):
        if n_nodes < 3:
            raise ValueError("n_nodes 至少为 3")
        self.length = float(vessel_length_cm)
        self.nx = int(n_nodes)
        self.paras = parameters if parameters is not None else BloodFlowParameters()

        self.x = np.linspace(0.0, self.length, self.nx)
        self.dx = self.length / (self.nx - 1)

        self.area_ref = np.asarray(area_cm2, dtype=float)
        self.beta = np.full(self.nx, self.paras.beta_mmhg_per_sqrt_cm)
        self.state = np.zeros((2, self.nx))
        self.pressure = np.zeros(self.nx)

        self.time = 0.0
        self._outlet_dt = 0.0

        self.history_time: np.ndarray | None = None
        self.history_area: np.ndarray | None = None
        self.history_flow: np.ndarray | None = None
        self.history_pressure: np.ndarray | None = None
        self.history_branch_flow: np.ndarray | None = None

        self.area_ref = np.maximum(self.area_ref, 1e-8)
        if self.area_ref.shape[0] != self.nx:
            raise ValueError(
                f"area 长度 ({self.area_ref.shape[0]}) 须等于 n_nodes ({self.nx})"
            )

        # #######其它状态初值设置#######
        # 狭窄检测与 β 放大
        self.prox_idx, self.dist_idx = select_reference_indices(self.area_ref)
        self.lesions = detect_stenoses(self.area_ref, self.prox_idx, self.dist_idx)
        if self.lesions:
            print(f"Detect {len(self.lesions)} stenoses.")
            for lesion in self.lesions:
                mask = (self.x >= self.x[lesion.i0]) & (self.x <= self.x[lesion.i1])
                b_scale = self.area_ref[self.prox_idx] / self.area_ref[lesion.mla_idx] if self.area_ref[lesion.mla_idx] <= 0.03 else 1.0
                self.beta[mask] *= b_scale

        # 近/远端参考段平整
        prox_mask = self.x <= self.x[self.prox_idx]
        dist_mask = self.x >= self.x[self.dist_idx]
        self.area_ref[prox_mask] = self.area_ref[self.prox_idx]
        self.area_ref[dist_mask] = self.area_ref[self.dist_idx]

        # 舍去近/远端参考段以外的侧支（只保留 prox_idx ≤ i ≤ dist_idx）
        if branch_frame_indices is not None and branch_diameters_cm is not None:
            kept_idx: list[int] = []
            kept_d: list[float] = []
            dropped = 0
            for i, d in zip(branch_frame_indices, branch_diameters_cm):
                ii = int(i)
                if ii < self.prox_idx or ii > self.dist_idx:
                    dropped += 1
                    continue
                kept_idx.append(ii)
                kept_d.append(float(d))
            if dropped:
                print(
                    f"Drop {dropped} side branch(es) outside reference "
                    f"[{self.prox_idx}, {self.dist_idx}]; kept {len(kept_idx)}."
                )
            branch_frame_indices = kept_idx
            branch_diameters_cm = kept_d

        # Murray 侧支：闭合直径 + 流量份额；远端主支终端份额 = distal_flow_fraction
        self.side_branches, self.distal_flow_fraction = prepare_side_branches(
            self.area_ref,
            self.prox_idx,
            self.dist_idx,
            branch_frame_indices,
            branch_diameters_cm,
            apply_murray_scale=apply_murray_scale,
        )
        self._branch_outflow = np.zeros(len(self.side_branches), dtype=float)
        if self.side_branches:
            print(
                f"Side branches: {len(self.side_branches)}, "
                f"distal Murray share={self.distal_flow_fraction:.3f}"
            )
            for b in self.side_branches:
                print(
                    f"  frame={b.frame_index}, D={b.diameter_cm:.4f} cm, "
                    f"Murray share f={b.flow_fraction:.3f} (WK R/C 标定), "
                    f"Q_share≈{b.flow_fraction * self.paras.mean_coronary_flow_ml_s():.4f} mL/s"
                )

        # 主支远端 + 各侧支 Windkessel（Murray share 标定 R、C）
        self.outlet = OutletWindkesselState(
            self.paras, flow_share=self.distal_flow_fraction
        )
        self.branch_outlets = [
            OutletWindkesselState(
                self.paras, flow_share=max(b.flow_fraction, 1e-6)
            )
            for b in self.side_branches
        ]

        q0 = float(np.asarray(self.paras.inlet_flow(0.0)).reshape(-1)[0])
        q_mean = self.paras.mean_coronary_flow_ml_s()
        q_distal_mean = q_mean * self.distal_flow_fraction
        self.outlet.reset(
            p_wk=q_distal_mean * self.outlet.r_d,
            q_drive=q0 * self.distal_flow_fraction,
        )
        # 将 tube law 的 p_ref 对齐到出口稳态管腔压（均值流量），避免 A=A₀ 时
        # 整段压位与 Windkessel 脱节（常见表现：入口压系统性低于出口）
        p_ref_user = float(self.paras.p_ref_mmhg)
        if align_p_ref_to_outlet:
            p_anchor = float(self.outlet.lumen_pressure(q_distal_mean))
            self.paras.p_ref_mmhg = p_anchor
            print(
                f"p_ref aligned to outlet steady lumen pressure: "
                f"{p_ref_user:.2f} → {p_anchor:.2f} mmHg "
                f"(outlet={self.paras.outlet_windkessel})"
            )
        p_ref = float(self.paras.p_ref_mmhg)
        # 侧支：按 P≈p_ref 时 Q_b≈Murray 份额初始化 P_wk，便于起步
        for br, wk in zip(self.side_branches, self.branch_outlets):
            q_share = q_mean * br.flow_fraction
            r_coup = wk.coupling_resistance()
            wk.reset(p_wk=p_ref - r_coup * q_share, q_drive=q_share)

        self.state[0] = self.area_ref.copy()
        self.state[1] = self._main_flow_profile(q0)
        self._update_pressure()
        self._branch_outflow = np.array(
            [
                wk.flow_from_lumen_pressure(float(self.pressure[br.frame_index]))
                for br, wk in zip(self.side_branches, self.branch_outlets)
            ],
            dtype=float,
        )
        print(
            f"p_ref={self.paras.p_ref_mmhg:.2f} mmHg; "
            f"outlet P_wk0={self.outlet.p_wk:.2f}, P_lumen0={self.outlet.p_lumen:.2f} mmHg"
        )
        if self.branch_outlets:
            qb0 = ", ".join(f"{q:.3f}" for q in self._branch_outflow)
            print(f"side-branch WK: n={len(self.branch_outlets)}, Q_b0=[{qb0}] mL/s")

    def _main_flow_profile(self, q_in: float) -> np.ndarray:
        """按 Murray 份额沿程累减侧支分流后的主支轴向流量初值/参考剖面。"""
        q = np.full(self.nx, float(q_in), dtype=float)
        removed = 0.0
        for br in self.side_branches:
            removed += br.flow_fraction * float(q_in)
            # 开口及以远主支流量减去该侧支份额
            q[br.frame_index :] = float(q_in) - removed
        q[0] = float(q_in)
        return np.maximum(q, 0.0)

    def _update_pressure(self, state: np.ndarray | None = None) -> np.ndarray:
        """由当前 A 经 tube law 更新沿程管腔标量压 p(x)。"""
        u = self.state if state is None else state
        self.pressure = lumen_pressure_mmhg(
            u[0], self.area_ref, self.beta, self.paras.p_ref_mmhg
        )
        return self.pressure

    def _apply_side_branches(
        self, state: np.ndarray, advance: bool = False
    ) -> None:
        """侧支：由开口管腔压经 Windkessel 反解 Q_b（质量汇）；不强制开口 A。

        Q_b = (P_ostium − P_wk) / R_coup，故近端压因狭窄升高时分流自动增大。
        """
        if not self.side_branches:
            return
        p = lumen_pressure_mmhg(
            state[0], self.area_ref, self.beta, self.paras.p_ref_mmhg
        )
        dt = self._outlet_dt if advance else 0.0
        for k, (br, wk) in enumerate(
            zip(self.side_branches, self.branch_outlets)
        ):
            p_ost = float(p[br.frame_index])
            if advance and dt > 0.0:
                self._branch_outflow[k] = wk.advance_from_pressure(dt, p_ost)
            else:
                self._branch_outflow[k] = wk.apply_substep_from_pressure(p_ost)

    def _spatial_operator(self, state: np.ndarray) -> np.ndarray:
        """有限体积右端项：−∂F/∂x + S。

        F 仅含对流；压力驱动为 −(A/ρ*)∂p/∂x；摩擦 −8πμ Q/A；
        侧支处连续方程含质量汇 −Q_b/Δx。
        """
        u_l, u_r = _muscl_states(state, self.area_ref)
        n_if = self.nx - 1
        j = np.arange(n_if)
        rho_star = self.paras.momentum_rho()
        f_inner = _hll_flux(
            u_l,
            u_r,
            self.beta[j],
            self.beta[j + 1],
            rho_star,
            self.paras.alpha,
        )
        f_face = np.zeros((2, self.nx + 1))
        f_face[:, 1:-1] = f_inner
        f_face[:, 0] = _flux_vector(
            state[0, 0], state[1, 0], self.paras.alpha
        ).ravel()
        f_face[:, -1] = _flux_vector(
            state[0, -1], state[1, -1], self.paras.alpha
        ).ravel()

        dudt = -(f_face[:, 1:] - f_face[:, :-1]) / self.dx
        a = np.maximum(state[0], 1e-12)

        # 压力梯度源项：−(A/ρ*) ∂p/∂x（p 由 tube law 在单元中心计算）
        p = lumen_pressure_mmhg(
            state[0], self.area_ref, self.beta, self.paras.p_ref_mmhg
        )
        dpdx = np.empty(self.nx, dtype=float)
        if self.nx >= 3:
            dpdx[1:-1] = (p[2:] - p[:-2]) / (2.0 * self.dx)
            dpdx[0] = (p[1] - p[0]) / self.dx
            dpdx[-1] = (p[-1] - p[-2]) / self.dx
        elif self.nx == 2:
            dpdx[:] = (p[1] - p[0]) / self.dx
        else:
            dpdx[:] = 0.0
        dudt[1] -= a * dpdx / rho_star

        dudt[1] -= 8.0 * np.pi * self.paras.mu * state[1] / a

        # 侧支质量汇：主支连续方程 ∂A/∂t + ∂Q/∂x = −q_branch
        for k, br in enumerate(self.side_branches):
            dudt[0, br.frame_index] -= self._branch_outflow[k] / self.dx
        return dudt

    def _apply_boundaries(
        self, state: np.ndarray, advance: bool = False
    ) -> np.ndarray:
        u = state.copy()
    
        if not advance:
            q_in = float(np.asarray(self.paras.inlet_flow(self.time)).reshape(-1)[0])
        else:
            q_in = float(np.asarray(self.paras.inlet_flow(self.time + self._outlet_dt)).reshape(-1)[0])
        u[1, 0] = q_in
        # 入口面积：零梯度外推（压力由 tube law 从 A 得到）
        u[0, 0] = u[0, 1]

        # 侧支：Windkessel 压力驱动分流（与主支出口同节拍推进）
        self._apply_side_branches(u, advance=advance)

        # 主支远端出口：Windkessel → P → A
        q_lumen = float(u[1, -1])
        if advance and self._outlet_dt > 0.0:
            _, p_out = self.outlet.advance(self._outlet_dt, q_lumen)
        else:
            _, p_out = self.outlet.apply_substep(q_lumen)
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
        u[0] = np.maximum(u[0], MINIMUM_AREA_RATIO * area_ref)
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
        # 求k1
        u_n = self._apply_boundaries(self.state)  # 使用控制方程更新通量前，先应用边界条件，避免边界不合理
        k1 = self._spatial_operator(u_n)  # 更新0-L内所有物理通量估计

        # 求k2
        u_n1 = self._clip_state(u_n + dt * k1, self.area_ref)
        u_n1 = self._apply_boundaries(u_n1, advance=True)  # 使用控制方程更新通量前，先应用边界条件，避免边界不合理
        k2 = self._spatial_operator(u_n1)

        self.state = u_n + dt * 0.5 * (k1 + k2)
        self.state = self._clip_state(
            self.state, self.area_ref
        )
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
        qb_hist = [self._branch_outflow.copy()]
        step = 0

        print(
            f"Navier-Stokes 1D: L={self.length:.2f} cm, nx={self.nx}, "
            f"T={duration_s:.3f} s, outlet={self.paras.outlet_windkessel}, "
            f"n_branch={len(self.side_branches)}"
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
                qb_hist.append(self._branch_outflow.copy())

        self.history_time = np.array(t_hist)
        self.history_area = np.array(a_hist)
        self.history_flow = np.array(q_hist)
        self.history_pressure = np.array(p_hist)
        self.history_branch_flow = np.array(qb_hist)
        t1 = time.perf_counter()
        print(f"完成 {step} 步, t={self.time:.4f} s, 耗时 {t1-t0:.3f} s")
        self.print_pressure_diagnostics()
        return (
            self.history_area,
            self.history_flow,
            self.history_pressure,
            self.history_time,
        )

    def pressure_diagnostics(
        self, warmup_fraction: float = 0.5
    ) -> dict[str, float]:
        """近端/远端（及入口/出口）压力诊断，便于检查是否出现 p_prox < p_dist。"""
        ip = int(self.prox_idx)
        id_ = int(self.dist_idx)
        p_ref = float(self.paras.p_ref_mmhg)

        if self.history_pressure is None:
            p = self.pressure
            p_in, p_out = float(p[0]), float(p[-1])
            p_prox, p_dist = float(p[ip]), float(p[id_])
            return {
                "p_ref_mmhg": p_ref,
                "p_in_mean": p_in,
                "p_out_mean": p_out,
                "p_prox_mean": p_prox,
                "p_dist_mean": p_dist,
                "dp_prox_minus_dist": p_prox - p_dist,
                "dp_in_minus_out": p_in - p_out,
                "prox_idx": float(ip),
                "dist_idx": float(id_),
            }

        n = self.history_pressure.shape[0]
        i0 = int(max(n * warmup_fraction, 0))
        p_avg = np.mean(self.history_pressure[i0:], axis=0)
        p_in, p_out = float(p_avg[0]), float(p_avg[-1])
        p_prox, p_dist = float(p_avg[ip]), float(p_avg[id_])
        return {
            "p_ref_mmhg": p_ref,
            "p_in_mean": p_in,
            "p_out_mean": p_out,
            "p_prox_mean": p_prox,
            "p_dist_mean": p_dist,
            "dp_prox_minus_dist": p_prox - p_dist,
            "dp_in_minus_out": p_in - p_out,
            "p_prox_min": float(np.min(self.history_pressure[i0:, ip])),
            "p_prox_max": float(np.max(self.history_pressure[i0:, ip])),
            "p_dist_min": float(np.min(self.history_pressure[i0:, id_])),
            "p_dist_max": float(np.max(self.history_pressure[i0:, id_])),
            "inversion_fraction": float(
                np.mean(
                    self.history_pressure[i0:, ip] < self.history_pressure[i0:, id_]
                )
            ),
            "prox_idx": float(ip),
            "dist_idx": float(id_),
        }

    def print_pressure_diagnostics(
        self, warmup_fraction: float = 0.5
    ) -> dict[str, float]:
        """打印近端/远端压力对照；返回诊断字典。"""
        stats = self.pressure_diagnostics(warmup_fraction=warmup_fraction)
        ip = int(stats["prox_idx"])
        id_ = int(stats["dist_idx"])
        tag = (
            "instant"
            if self.history_pressure is None
            else f"mean after {warmup_fraction:.0%}T"
        )
        print(
            f"[P diag {tag}] p_ref={stats['p_ref_mmhg']:.2f} | "
            f"in={stats['p_in_mean']:.2f} out={stats['p_out_mean']:.2f} "
            f"(Δ={stats['dp_in_minus_out']:+.2f}) | "
            f"prox[{ip}]={stats['p_prox_mean']:.2f} dist[{id_}]={stats['p_dist_mean']:.2f} "
            f"(Δ={stats['dp_prox_minus_dist']:+.2f}) mmHg"
        )
        if "inversion_fraction" in stats:
            print(
                f"  prox range [{stats['p_prox_min']:.2f}, {stats['p_prox_max']:.2f}], "
                f"dist range [{stats['p_dist_min']:.2f}, {stats['p_dist_max']:.2f}], "
                f"p_prox<p_dist fraction={stats['inversion_fraction']:.1%}"
            )
        if stats["dp_prox_minus_dist"] < 0.0:
            print("  !! warning: p_prox < p_dist (pressure inversion)")
        return stats

    def ffr_ratio(self, warmup_fraction: float = 0.5) -> np.ndarray:
        """沿程压力 FFR：时间平均 p(x) / p(prox)，期望单调非增。"""
        from scipy.signal import medfilt
        if self.history_pressure is None:
            p_mean = self.pressure.copy()
        else:
            n = self.history_pressure.shape[0]
            i0 = int(max(n * warmup_fraction, 0))
            p_mean = np.mean(self.history_pressure[i0:], axis=0)
        p0 = float(p_mean[self.prox_idx])
        if p0 <= 1e-9:
            return np.ones(self.nx)
        ffr = p_mean / p0
        return np.clip(ffr, 0.0, 1.0)
    
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

    def plot_results_1(
        self,
        show: bool = True,
        title: str = "",
        renderer: str | None = None,
        save_path: str | None = None,
    ):
        """
        与 plot_results 相同的四幅图，使用 Plotly 在浏览器中交互显示。

        Parameters
        ----------
        show : bool
            是否调用 fig.show() 打开浏览器
        renderer : str, optional
            传给 plotly fig.show(renderer=...)，如 "browser"
        save_path : str, optional
            保存路径；``.html`` 写交互页，其它后缀走 ``write_image``（需 kaleido）
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
        t_show = float(tt[max(-100, -len(tt))])

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "A(0, x) [cm²]",
                "FFR pullback (pressure)",
                "x = 近端(blue) / 远端(red)",
                f"t = {t_show:.3f} s",
            ),
            specs=[[{}, {}], [{}, {"secondary_y": True}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=self.area_ref,
                mode="lines",
                name="Initial Areas",
                line=dict(color="black", width=2),
                hovertemplate="x=%{x:.1f} cm<br>area=%{y:.4f} cm²<extra></extra>",
            ),
            row=1,
            col=1,
        )

        ffr = self.ffr_ratio()
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=ffr,
                mode="lines",
                name="FFR (with branches)" if self.side_branches else "FFR",
                line=dict(color="darkred", width=2),
                hovertemplate="x=%{x:.1f} cm<br>FFR=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=0.80,
            line_dash="dot",
            line_color="black",
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=tt,
                y=self.history_pressure[:, self.prox_idx],
                mode="lines",
                name="x=0",
                line=dict(color="blue", width=1.5),
                hovertemplate="t=%{x:.4f} s<br>P=%{y:.3f} mmHg<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=self.history_pressure[:, self.dist_idx],
                mode="lines",
                name=f"x={self.dist_idx}",
                line=dict(color="red", width=1.5),
                hovertemplate="t=%{x:.4f} s<br>P=%{y:.3f} mmHg<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=self.history_area[-100],
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
                y=self.history_pressure[-100],
                mode="lines",
                name="Q",
                line=dict(color="green", width=1.5, dash="dash"),
                hovertemplate="x=%{x:.3f} cm<br>P=%{y:.3f} mmHg<extra></extra>",
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        fig.update_xaxes(title_text="A (cm²)", row=1, col=1)
        fig.update_yaxes(title_text="x (cm)", row=1, col=1)
        fig.update_xaxes(title_text="x (cm)", row=1, col=2)
        fig.update_yaxes(title_text="FFR", row=1, col=2)
        fig.update_xaxes(title_text="t (s)", row=2, col=1)
        fig.update_yaxes(title_text="P (mmHg)", row=2, col=1)
        fig.update_xaxes(title_text="x (cm)", row=2, col=2)
        fig.update_yaxes(title_text="A (cm²)", row=2, col=2)
        fig.update_yaxes(title_text="P (mmHg)", row=2, col=2, secondary_y=True)

        fig.update_layout(
            height=1000,
            width=1000,
            title_text=f"Navier-Stokes 1D 结果 ({title})",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")

        if save_path is not None:
            import os

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


def demo():
    from scipy.ndimage import gaussian_filter1d
    import os

    name = 'case3'
    dirname = f"D:/WPY/Projects/FFR_test/output/lumen_area/{name}"
    np.random.seed(2034)
    sigma_nodes = 4
    duration_s = 3.0

    # 原始 npy：面积为 mm²，直径为 mm，读入后一律换成 cm / cm²
    area_file = np.load(os.path.join(dirname, "area.npy"))
    area = area_file[::-1] / 100.0  # mm² → cm²
    area_smooth = gaussian_filter1d(area, sigma=sigma_nodes, mode="nearest")
    
    is_branch = np.load(os.path.join(dirname, "is_branch.npy"))[::-1]
    branch_d_cm = np.load(os.path.join(dirname, "branch_diameter.npy"))[::-1] / 10.0  # mm → cm

    # 按参考点重新设置长度
    # area_smooth = area_smooth[:200]
    # is_branch = is_branch[:200]
    # branch_d_cm = branch_d_cm[:200]

    nx = area_smooth.shape[0]
    length = 0.02 * nx  
    print(nx, length)

    prox_idx, dist_idx = select_reference_indices(area_smooth)
    branch_idx = np.where(is_branch > 0)[0]
    merged_idx: list[int] = []
    merged_d: list[float] = []
    for i in branch_idx:
        if i < prox_idx or i > dist_idx:
            continue
        if merged_idx and i - merged_idx[-1] <= 3:
            if float(branch_d_cm[i]) > merged_d[-1]:
                merged_idx[-1] = int(i)
                merged_d[-1] = float(branch_d_cm[i])
        else:
            merged_idx.append(int(i))
            merged_d.append(float(branch_d_cm[i]))

    parameters = BloodFlowParameters()
    solver = NavierStokes1D(
        area_smooth,
        length,
        nx,
        parameters,
        branch_frame_indices=merged_idx,
        branch_diameters_cm=merged_d,
        # 默认 apply_murray_scale=True、align_p_ref_to_outlet=True
    )
    solver.run(duration_s=duration_s, record_interval_steps=30)
    solver.plot_results_1(title=name)

    assert solver.history_time is not None
    assert solver.history_flow is not None
    assert solver.history_pressure is not None

    q_in = np.mean(solver.history_flow[:, 0])
    q_out = np.mean(solver.history_flow[:, -1])
    q_side = (
        float(np.mean(np.sum(solver.history_branch_flow, axis=1)))
        if solver.history_branch_flow is not None and solver.history_branch_flow.size
        else 0.0
    )
    print(f"Q_in={q_in:.3f}, Q_out={q_out:.3f}, Q_side_sum={q_side:.3f}")
    print(f"mass check: Q_in ≈ Q_out+Q_side → {q_in:.3f} vs {q_out + q_side:.3f}")

    ffr = solver.ffr_ratio()
    return ffr


def roi_process(roi_idx, area, branch_idx, branch_d):
    """按 ROI 下标裁剪面积与侧支。

    Parameters
    ----------
    roi_idx :
        ROI 在原始序列中的下标列表，例如 ``[0, 1, ..., 199]`` 或任意子集
        （须为 ``area`` 的合法下标）。
    area :
        原始沿程截面积，形状 ``(nx,)``
    branch_idx, branch_d :
        原始网格上的侧支开口下标与直径；等长

    Returns
    -------
    area_roi, branch_idx_roi, branch_d_roi
        - ``area_roi``：按 ``roi_idx`` 顺序取出的面积
        - 落在 ROI 内的侧支下标映射为 ROI 内局部下标（``enumerate(roi_idx)``）；
          不在 ROI 内的侧支删除
    """
    roi = np.asarray(roi_idx, dtype=int).ravel()
    area_arr = np.asarray(area, dtype=float).ravel()
    nx = int(area_arr.shape[0])
    if roi.size == 0:
        raise ValueError("roi_idx 不能为空")
    if np.any(roi < 0) or np.any(roi >= nx):
        raise ValueError(f"roi_idx 须落在 [0, {nx})，当前范围 [{roi.min()}, {roi.max()}]")

    area_roi = area_arr[roi].copy()
    old_to_new = {int(old): int(new) for new, old in enumerate(roi)}

    idxs = list(branch_idx) if branch_idx is not None else []
    diams = list(branch_d) if branch_d is not None else []
    if len(idxs) != len(diams):
        raise ValueError(
            f"branch_idx 与 branch_d 长度须一致 ({len(idxs)} vs {len(diams)})"
        )

    branch_idx_roi: list[int] = []
    branch_d_roi: list[float] = []
    dropped = 0
    for i, d in zip(idxs, diams):
        old_i = int(i)
        if old_i in old_to_new:
            branch_idx_roi.append(old_to_new[old_i])
            branch_d_roi.append(float(d))
        else:
            dropped += 1

    if dropped:
        print(
            f"roi_process: drop {dropped} branch(es) outside ROI; "
            f"kept {len(branch_idx_roi)}, area nx {nx} → {len(area_roi)}"
        )
    return area_roi, branch_idx_roi, branch_d_roi



def example(
    target_nx: int | None = 100,
    frame_spacing_cm: float = 0.02,
    duration_s: float = 3.0,
    sigma_nodes: float = 4,
    name: str = 'Null',
):
    """下游 joblib 数据示例；``target_nx`` 控制轴向降采样（None=不降采样）。"""
    import joblib
    from scipy.ndimage import gaussian_filter1d
    import os
    np.random.seed(2034)
    mm_per_pixel = (9.7 / 756)

    dirname = f"D:/WPY/Projects/lumen_area/results/output_for_downstrem"
    # 原始 npy：面积为 pixel²，直径为 pixel，读入后一律换成 cm / cm²
    area_file = joblib.load(os.path.join(dirname, "areas.joblib"))
    area = np.array(area_file)[::-1] * (mm_per_pixel**2) / 100.0  # mm² → cm²
    area_smooth = gaussian_filter1d(area, sigma=sigma_nodes, mode="nearest")

    nx = area_smooth.shape[0]
    print(f"raw nx={nx}")

    # 分支信息（与 area 同向：近端→远端）
    branch_groups = joblib.load(os.path.join(dirname, "branches.joblib"))
    branch_d = np.array(joblib.load(os.path.join(dirname, "branch_diameters.joblib")))
    branch_d = branch_d[::-1] * mm_per_pixel / 10.0  # mm → cm
    branch_idx = [nx - idxs[0] for idxs in branch_groups][::-1]

    # 只选择目标区域
    roi_idx = list(range(10, 240))
    area_smooth, branch_idx, branch_d = roi_process(roi_idx, area_smooth, branch_idx, branch_d)

    # 轴向重采样：降低 nx 以加速；物理长度保持 frame_spacing_cm * nx_raw
    area_smooth, length, nx, branch_idx, branch_d = resample_axial_profile(
        area_smooth,
        target_nx=target_nx,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
        frame_spacing_cm=frame_spacing_cm,
    )
    print(nx, length, f"n_branch={len(branch_idx)}")

    parameters = BloodFlowParameters()
    solver = NavierStokes1D(
        area_smooth,
        length,
        nx,
        parameters,
        branch_frame_indices=branch_idx,
        branch_diameters_cm=branch_d,
        # 默认 apply_murray_scale=True、align_p_ref_to_outlet=True
    )
    solver.run(duration_s=duration_s, record_interval_steps=30)
    solver.plot_results_1(title=f"{name}_nx{nx}")

    assert solver.history_time is not None
    assert solver.history_flow is not None
    assert solver.history_pressure is not None

    q_in = np.mean(solver.history_flow[:, 0])
    q_out = np.mean(solver.history_flow[:, -1])
    q_side = (
        float(np.mean(np.sum(solver.history_branch_flow, axis=1)))
        if solver.history_branch_flow is not None and solver.history_branch_flow.size
        else 0.0
    )
    print(f"Q_in={q_in:.3f}, Q_out={q_out:.3f}, Q_side_sum={q_side:.3f}")
    print(f"mass check: Q_in ≈ Q_out+Q_side → {q_in:.3f} vs {q_out + q_side:.3f}")

    ffr = solver.ffr_ratio()
    return ffr


if __name__ == "__main__":
    # ffr = demo()
    # target_nx=100 降采样；改为 None 可跑原始分辨率对比准确率
    ffr = example(target_nx=200)
    print(f"max ffr: {np.max(ffr)}/n min ffr: {np.min(ffr)}")
    dffr = np.diff(ffr)
    print(f"FFR non-increasing violations: {int(np.sum(dffr > 1e-3))}")
