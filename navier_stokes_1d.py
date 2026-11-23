"""
navier_stokes_1d.py — 冠状动脉一维轴向血流模拟

依据一维可变形管内的守恒型 Navier-Stokes 方程组，耦合管壁弹性压力关系，
对沿血管轴向 x ∈ [0, L] 的截面积 A 与体积流量 Q 进行数值求解。

控制方程（长度/面积/流量用 CGS，管腔压 p 用 mmHg）
------------------------------------------------------------
    ∂U/∂t + ∂F/∂x = S,    U = [A, Q]ᵀ

    F₀ = Q
    F₁ = α·Q²/A                         （仅对流；不含 pA/ρ*）
    S  = [0, −(A/ρ*)·∂p/∂x − 8πν·Q/A]ᵀ

    等价于文献标准式：
        ∂Q/∂t + ∂(α Q²/A)/∂x + (A/ρ*) ∂p/∂x = −8πν Q/A

    注意：旧版曾写 F₁ = α Q²/A + p A/ρ* 且无 (p/ρ*)∂A/∂x 源项，
    与上式不等价，变截面处会产生虚假压力梯度（见 docs/navier_stokes_1d.md §3.1.1）。

    ρ* = ρ / (1333.22 g·cm⁻¹·s⁻² per mmHg)，在 BloodFlowParameters 中一次性标定；
    与逐点把 p 换成 dyne/cm² 再代入原式等价。

    摩擦项中的 ν 为**运动粘度**（cm²/s），不是动力粘度：
        由 (1/ρ)∂p/∂x = −8πμ_dyn Q/(ρ A²) 归一化得 (A/ρ)∂p/∂x = −8πν Q/A，ν = μ_dyn/ρ。
        故 ν ≈ 3.5 cP / 1.06 g·cm⁻³ = 0.033 cm²/s（全血 37°C）。

管壁弹性（tube law，mmHg）
--------------------------
    p(A, x) = P_ref + β(x)·(√A − √A₀(x))

    p 为管腔标量压（各向同性，非轴向/径向矢量分量）。tube law 将其与截面面积 A 耦合
    （径向胀缩）；动量源项 −(A/ρ*)∂p/∂x 体现沿 x 的压力梯度对轴向流量 Q 的驱动。

    A₀(x) 由用户给定的沿程管腔参考面积描述；β(x) 为管壁刚度（可常数或随 x 给定）。

边界条件
--------
- 入口：给定 Q_in(t)；面积由出射 Riemann 不变量 W− = u − 4c 反解（亚临界）
- 侧支：各开口接 0D 三元 Windkessel（RCR）；由主支开口管腔压 P_ostium 反解
        Q_b=(P_ostium−P_wk)/R_p，
        连续方程质量汇 −Q_b/Δx；Murray 份额仅用于标定各支 R、C（flow_share）
- 出口：Windkessel 给定管腔压并反解 A(L)；流量由出射不变量 W+ = u + 4c 反解
        （亚临界）。C·dP_wk/dt = Q(L) − P_wk/R_d

初值（构造 `NavierStokes1D` 时完成）
------------------------------------
- A(x,0) = A₀(x)（经近/远端参考段平整；β 全段取常数，不随狭窄放大）
- Q(x,0) 按 Murray 份额沿程分流剖面
- 主支远端 P_wk(0) = Q_mean·f_distal·R_d（R 已按份额放大）
- 构造各终端 Windkessel 之前，用未缩放的 R_d、R_p 与 Q_mean 将 p_ref
  写成 Q_mean·(R_d+R_p)，使 A=A₀ 时 tube law 与出口稳态管腔压一致。
  该对齐不可关闭：用已按 1/份额放大的出口阻力来反推 p_ref 会把份额混进去
- 侧支默认 apply_murray_scale=True（Murray 闭合）；并校验直径单位/是否大于主支

主要输入
--------
1. 管腔参考截面积 A₀（cm²，一维数组；节点数 n_x = len(A₀)）及血管长度 L
2. BloodFlowParameters（物性、入口、出口 Windkessel）
3. geometry.build_stenosis_segment（构造内自动调用：MLA + 近/远端参考）

空间离散：单元中心有限体积法 + MUSCL 线性重构 + MC（monotonized central）TVD 限制器 + HLL 数值通量
时间离散：SSP-RK2（二阶强稳定保持 Runge-Kutta）
时间步长：CFL 自适应
"""
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from branch import prepare_side_branches
from calibration_cache import iteration_snapshot, save_calibration_cache
from coronary_constants import (
    DEFAULT_PROXIMAL_FRACTION,
    MINIMUM_AREA_RATIO,
    MINIMUM_INLET_FLOW_ML_S,
    P_INLET_REF_MMHG,
    rho_for_mmhg_pressure_coupling,
)
from coronary_inlet import coronary_inlet_flow
from geometry import build_stenosis_segment, smooth_axial_area, resample_axial_profile


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
# 数值工具
# ---------------------------------------------------------------------------
def integral_mean(y: np.ndarray | Sequence[float], x: np.ndarray | Sequence[float]) -> float:
    """梯形法离散计算一维采样的积分均值。

    .. math::
        \\bar{y} = \\frac{1}{x_N - x_0}\\int_{x_0}^{x_N} y(x)\\,\\mathrm{d}x
        \\approx \\frac{\\mathrm{trapz}(y, x)}{x[-1] - x[0]}

    Parameters
    ----------
    y :
        函数值采样，一维，长度 ≥ 2
    x :
        自变量采样点，一维，与 ``y`` 等长；通常单调

    Returns
    -------
    float
        积分均值
    """
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float).ravel()
    if yy.size != xx.size:
        raise ValueError(f"y 与 x 长度须一致 ({yy.size} vs {xx.size})")
    if yy.size < 2:
        raise ValueError(f"采样点数至少为 2，当前 {yy.size}")
    span = float(xx[-1] - xx[0])
    if abs(span) <= 1e-15:
        raise ValueError("x 首尾跨度过小，无法计算积分均值")
    return float(np.trapezoid(yy, xx) / span)


# ---------------------------------------------------------------------------
# TVD / MUSCL / HLL
# ---------------------------------------------------------------------------
def _minmod(slope_central_difference: np.ndarray, slope_l: np.ndarray, slope_r: np.ndarray) -> np.ndarray:
    """van Leer 的 MC (Monotonized Central) 限制器。
    将中心差分斜率与两侧的单侧斜率放入 minmod 函数中取限制值。
    若 a,b,c 同号，则取绝对值最小者并保留符号；否则取零。
    """
    # 三者同正
    signs_all_positive = np.where(slope_central_difference > 0, 1, 0) * np.where(slope_l > 0, 1, 0) * np.where(slope_r > 0, 1, 0)
    # 三者同负
    signs_all_negative = np.where(slope_central_difference < 0, 1, 0) * np.where(slope_l < 0, 1, 0) * np.where(slope_r < 0, 1, 0)
    
    # 绝对值最小
    abs_min = np.minimum(np.abs(slope_central_difference), np.minimum(2 * np.abs(slope_l), 2 * np.abs(slope_r)))
    result = (signs_all_positive * abs_min
    + (-signs_all_negative) * abs_min)
    return result


def _muscl_states(
    U: np.ndarray,
    area_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """单元界面 j+1/2 的左右重构状态（内部界面）。"""
    nvar, nx = U.shape  # nvar 为未知函数量，nx 为划分单元数
    slope = np.zeros_like(U)
    if nx >= 3:
        slope_central_difference = (U[:, 2:] - U[:, :-2]) * 0.5  # dx直接约去，范围 [1, nx-2]，无第0, nx-1 (首尾两单元)
        slope_left = U[:, 1:-1] - U[:, :-2]  # dx直接约去，范围 [1, nx-2]，无第0, nx-1 (首尾两单元)
        slope_right = U[:, 2:] - U[:, 1:-1]  # dx直接约去，范围 [1, nx-2]，无第0, nx-1 (首尾两单元)
        slope[:, 1:-1] = _minmod(slope_central_difference, slope_left, slope_right)
    # 左状态计算，来自当前单元
    u_left = U[:, :-1] + 0.5 * slope[:, :-1]  # 共 nx-1 个
    # 右状态计算，来自下一个单元
    u_right = U[:, 1:] - 0.5 * slope[:, 1:]  # 共 nx-1 个
    floor = MINIMUM_AREA_RATIO * area_ref
    u_left[0] = np.maximum(u_left[0], floor[:-1])
    u_right[0] = np.maximum(u_right[0], floor[1:])
    return u_left, u_right


def _wave_speed(
    area: np.ndarray | float,
    beta: np.ndarray | float,
    rho_mmhg: float,
) -> np.ndarray | float:
    """小扰动弹性波速 c = √(β√A/(2ρ*))；p、β 均为 mmHg 制，ρ* 为动量耦合有效密度。

    **本函数返回纯弹性波速，不含动量修正 α**；需要含 α 的特征速度
    λ± 请用 :func:`_characteristic_speeds`。

    ``area``、``beta`` 均可为标量或数组（支持广播）。标量 ``beta`` 且
    ``area=1`` 时即波速前置因子 γ = √(β/(2ρ*))——原先独立的
    ``_c_prefactor`` 已并入本函数，用 ``_wave_speed(1.0, beta, rho_mmhg)``
    取代。
    """
    b = np.asarray(beta, dtype=float)
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    # c² = (A/ρ*)·dp/dA，tube law 下 dp/dA = β/(2√A) → c = √(β√A/(2ρ*))
    # β 可为标量或与 A 同形的数组；标量 β 且 A=1 时即前置因子 γ = √(β/(2ρ*))
    return np.sqrt(b / (2.0 * float(rho_mmhg))) * (a ** 0.25)


def _characteristic_speeds(
    area: np.ndarray | float,
    flow: np.ndarray | float,
    beta: np.ndarray | float,
    rho_mmhg: float,
    alpha: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """返回 (λ+, λ−)。**特征速度的唯一实现**，含动量修正 α。

    λ± = αu ± √(c² + α(α−1)u²)，其中 c = √(β√A/(2ρ*))。
    ``beta`` 可为标量或与 ``area`` 同形的数组；标量输入返回 ``float``。
    HLL 波速、CFL 步长、边界相容与适定性诊断均调用本函数，避免多处
    各自展开公式导致不同步。
    """
    a = np.maximum(np.asarray(area, dtype=float), 1e-12)
    u = np.asarray(flow, dtype=float) / a
    c = np.asarray(_wave_speed(a, beta, rho_mmhg), dtype=float)
    disc = np.sqrt(c * c + alpha * (alpha - 1.0) * u * u)
    lam_p = alpha * u + disc
    lam_m = alpha * u - disc
    if np.ndim(a) == 0:
        return float(lam_p), float(lam_m)
    return lam_p, lam_m


def _riemann_invariants(
    area: float,
    flow: float,
    beta: float,
    rho_mmhg: float,
) -> tuple[float, float]:
    """tube law 的 Riemann 不变量 W± = u ± 4c。

    c = √(β √A / (2ρ*))，故 ∫(c/A) dA = 4c。这是 α = 1 时的标准不变量，
    不是浅水方程的 u ± 2c。通量里仍使用参数 α；α 接近 1 时边界相容用 4c。
    """
    a = max(float(area), 1e-12)
    u = float(flow) / a
    c = float(_wave_speed(a, float(beta), rho_mmhg))
    return u + 4.0 * c, u - 4.0 * c


def area_from_flow_and_w_minus(
    flow: float,
    w_minus: float,
    beta: float,
    rho_mmhg: float,
    area_ref: float,
    area_guess: float,
) -> float:
    """由入口流量与出射不变量反解面积：Q/A − 4c(A) = W−。

    Q > 0 时左端对 A 严格递减，正根唯一，用二分。根低于面积下限时返回下限
    （此时不变量无法精确满足，与 tube law 钳位一致）。
    """
    # γ = √(β/(2ρ*))：即 A=1 时的波速，等价于旧 _c_prefactor(beta, rho)
    gamma = float(_wave_speed(1.0, beta, rho_mmhg))
    a_min = MINIMUM_AREA_RATIO * float(area_ref)
    q = float(flow)
    w = float(w_minus)

    def residual(area: float) -> float:
        aa = max(float(area), 1e-16)
        return q / aa - 4.0 * gamma * aa**0.25 - w

    if q > 0.0:
        lo = 1e-14
        hi = max(float(area_guess), a_min, 1e-8)
        g_hi = residual(hi)
        for _ in range(60):
            if g_hi <= 0.0:
                break
            hi *= 2.0
            g_hi = residual(hi)
        else:
            return max(float(area_guess), a_min)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return max(0.5 * (lo + hi), a_min)

    if abs(q) <= 1e-14 and w < 0.0 and gamma > 0.0:
        z = -w / (4.0 * gamma)
        return max(z**4, a_min)
    return max(float(area_guess), a_min)


def _outlet_flow_from_w_plus(
    w_plus: float,
    p_wk: float,
    r_p: float,
    r_d: float,
    compliance: float,
    dt: float,
    area_ref: float,
    beta: float,
    p_ref: float,
    rho_mmhg: float,
    q_guess: float,
) -> float | None:
    """出口：给定出射不变量 W+，求与 RCR 管腔压相容的 Q。

    冻结步（``dt==0``）：P = P_wk + R_p Q。
    推进步（``dt>0``）：P_wk 按与 ``OutletWindkesselState.advance`` 相同的
    显式欧拉用该 Q 预测，P = P_wk_new + R_p Q。
    A 由 tube law 反解，约束为 Q = A (W+ − 4c(A))。
    不收敛时返回 None，调用方退回「保留 PDE 流量、只定 A」。
    """
    a0 = float(area_ref)
    beta_f = float(beta)
    p_ref_f = float(p_ref)
    r_d_f = float(r_d)  # 调用方传入已在 OutletWindkesselState 初始化时钉下限
    r_p_f = float(r_p)
    c_wk = max(float(compliance), 1e-9)
    dt_f = max(float(dt), 0.0)
    p_wk_f = float(p_wk)

    def p_lumen(q: float) -> float:
        p_node = p_wk_f
        if dt_f > 0.0:
            p_node = p_node + (dt_f / c_wk) * (q - p_node / r_d_f)
        return p_node + r_p_f * q

    def q_compatible(q: float) -> float:
        a = float(
            np.asarray(
                area_from_lumen_pressure_mmhg(p_lumen(q), a0, beta_f, p_ref_f)
            ).reshape(-1)[0]
        )
        c = float(_wave_speed(max(a, 1e-12), beta_f, rho_mmhg))
        return a * (float(w_plus) - 4.0 * c)

    q = float(q_guess)
    for _ in range(16):
        r = q - q_compatible(q)
        scale = max(1.0, abs(q))
        if abs(r) <= 1e-8 * scale:
            return q
        dq = 1e-6 * scale
        slope = (q + dq - q_compatible(q + dq) - r) / dq
        if not np.isfinite(slope) or abs(slope) < 1e-14:
            return None
        step = r / slope
        accepted = False
        for _backtrack in range(8):
            q_new = q - step
            if np.isfinite(q_new) and abs(q_new) < 1.0e6:
                r_new = q_new - q_compatible(q_new)
                if abs(r_new) < abs(r):
                    q = q_new
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return None
    r_end = q - q_compatible(q)
    if abs(r_end) <= 1e-6 * max(1.0, abs(q)):
        return q
    return None


def _flux_vector(
    area: np.ndarray,
    flow: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """物理通量 F(U)：质量通量 Q，动量通量仅对流项 α Q²/A。
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

    波速估计用 ``_characteristic_speeds`` 的 λ±（含动量修正 α），使耗散与
    真实特征速度一致；压力驱动本身在单元中心以 −(A/ρ*)∂p/∂x 处理。
    """
    # 管腔截面积左右状态
    a_l = np.maximum(u_l[0], 1e-12)
    a_r = np.maximum(u_r[0], 1e-12)
    # 管腔体积流量左右状态
    q_l, q_r = u_l[1], u_r[1]
    # 特征速度左右状态（λ± 唯一实现，含 α）
    lambda_plus_l, lambda_minus_l = _characteristic_speeds(
        a_l, q_l, beta_l, rho_mmhg, alpha
    )
    lambda_plus_r, lambda_minus_r = _characteristic_speeds(
        a_r, q_r, beta_r, rho_mmhg, alpha
    )
    # HLL Riemann 求解器 估算的SL, SR
    s_l = np.minimum(np.minimum(lambda_minus_l, lambda_minus_r), 0)
    s_r = np.maximum(np.maximum(lambda_plus_l, lambda_plus_r), 0)

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
    mu: float = 0.033  # 血液运动粘度 ν (cm²/s) = 3.5 cP ÷ 1.06 g·cm⁻³；见 momentum_rho 说明
    cfl: float = 0.8  # 数值稳定性参数 CFL
    dt_max_s: float = 4e-4  # 时间步上限 (s)

    # 管壁相关 tube law
    p_ref_mmhg: float = P_INLET_REF_MMHG  # tube law 参考压 (mmHg)
    beta_mmhg_per_cm: float = 3500.0  # 管壁刚度 β (mmHg/cm)，越大管腔越硬

    # 冠脉血流入口相关
    heart_rate_bpm: float = 75.0  # 心率 (bpm)
    cardiac_output_l_per_min: float = 5.5  # 心脏输出量 (L/min)  静息心输出量设为5L/min, 充血状态时心输出量设为8L/min
    coronary_flow_fraction: float = 0.03  # 左冠脉血流占心脏输出量的比例（生理分流比，不因狭窄改语义）
    q_scale: float = 1.0  # 狭窄有效流量缩放；经 coronary_fraction*q_scale 传入入口生成器
    inlet_min_flow_fraction: float = 0.25  # 流量下限，即最小值占整个流量平均值的比例
    inlet_phase_offset_rad: float = 0.0  # 波形相位偏移
    inlet_fourier_coefficients: Sequence[tuple[int, float, float]] | None = None  # 自定义谐波（傅里叶变换的系数）；None 用模块内置默认系数

    # 出口 Windkessel 模型相关（三元 RCR）
    outlet_r_distal_mmhg_s_per_ml: float | None = 25  # Rd 远端阻力系数
    outlet_r_proximal_mmhg_s_per_ml: float | None = None  # Rp 近端阻力系数
    outlet_compliance_ml_per_mmhg: float = 0.012  # 血管顺应性，越大顺应性越好 
    outlet_proximal_fraction: float = DEFAULT_PROXIMAL_FRACTION  # 近端阻力比例

    def mean_coronary_flow_ml_s(self) -> float:
        """入口平均流量 (mL/s)，含 ``q_scale``。"""
        return (
            self.cardiac_output_l_per_min
            * 1000.0
            * self.coronary_flow_fraction
            / 60.0
            * float(self.q_scale)
        )

    def inlet_flow(self, t: float | np.ndarray) -> np.ndarray:
        """生成时间t的入口流量 (mL/s)。

        ``q_scale`` 乘在 ``coronary_fraction`` 上再交给入口波形生成，
        使均值、傅里叶谐波与 ``min_flow_fraction`` 下限共用同一 ``Q_mean``；
        返回值再钳到 ``MINIMUM_INLET_FLOW_ML_S`` (0.01 mL/s)。
        """
        q = np.asarray(
            coronary_inlet_flow(
                t,
                heart_rate=self.heart_rate_bpm,
                cardiac_output=self.cardiac_output_l_per_min,
                coronary_fraction=self.coronary_flow_fraction * float(self.q_scale),
                fourier_coefficients=self.inlet_fourier_coefficients,
                min_flow_fraction=self.inlet_min_flow_fraction,
                phase_offset=self.inlet_phase_offset_rad,
            ),
            dtype=float,
        )
        return np.maximum(q, float(MINIMUM_INLET_FLOW_ML_S))

    def resolve_outlet_resistances(self) -> tuple[float, float]:
        """远端/近端阻力 (mmHg·s/mL)，与 coronary_outlet 默认标定一致。"""
        q_mean = self.mean_coronary_flow_ml_s()
        r_d = (
            self.p_ref_mmhg / max(q_mean, 1e-6) / max(1 + self.outlet_proximal_fraction, 1e-6)
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
        """与 mmHg 制管腔压配对的有效密度 ρ*（用于 (A/ρ*)∂p/∂x 与波速）。

        注意：这里的 ρ* 只服务于**压力项与波速**（因 p 用 mmHg 而非 dyne/cm²）。
        **摩擦项不复用 ρ***：动量式 −8πμQ/A 要求 μ 为运动粘度 ν = μ_dyn/ρ，
        归一化后已不含 ρ，见 :attr:`mu` 的取值说明。
        """
        return rho_for_mmhg_pressure_coupling(self.rho)


class OutletWindkesselState:
    """
    与 coronary_outlet 中三元 Windkessel（RCR）ODE 一致的出口/侧支状态机。

    P_out = P_wk + R_p·Q_drive，P_wk 为微循环顺应性节点压。

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
        share = max(float(flow_share), 1e-6)
        self.flow_share = share
        r_d0, r_p0 = params.resolve_outlet_resistances()
        # 阻力下限一次钉死，后续除法不再反复 max(..., 1e-9)
        self.r_d = max(float(r_d0) / share, 1e-9)
        self.r_p = max(float(r_p0) / share, 1e-9)
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
        self.p_lumen = self.lumen_pressure(q_drive)

    def lumen_pressure(self, q_drive: float) -> float:
        """出口/开口管腔压：P_lumen = P_wk + R_p·Q_drive。"""
        return self.p_wk + self.r_p * float(q_drive)

    def flow_from_lumen_pressure(self, p_lumen: float) -> float:
        """给定开口/出口管腔压，反解进入 Windkessel 的流量 Q_drive。

        Q = (P_lumen − P_wk) / R_p
        """
        return (float(p_lumen) - self.p_wk) / self.r_p

    def apply_substep(self, q_lumen: float) -> float:
        """SSP-RK 子步（流量驱动）：冻结 P_wk，返回 P_lumen。"""
        q_drive = float(q_lumen)
        self.p_lumen = self.lumen_pressure(q_drive)
        return self.p_lumen

    def advance(self, dt: float, q_lumen: float) -> float:
        """完整时间步末（流量驱动）：C·dP_wk/dt = Q_drive − P_wk/R_d，返回 P_lumen。

        微循环出流 P_wk/R_d 仅出现在本 ODE 右端，不单独向外返回。
        """
        if dt <= 0.0:
            return self.apply_substep(q_lumen)
        q_drive = float(q_lumen)
        self.p_wk += (dt / self.c) * (q_drive - self.p_wk / self.r_d)
        self.p_lumen = self.lumen_pressure(q_drive)
        return self.p_lumen

    def apply_substep_from_pressure(self, p_lumen: float) -> float:
        """SSP-RK 子步（压力驱动）：冻结 P_wk，返回 Q_drive，用于计算分支分流量"""
        q_drive = self.flow_from_lumen_pressure(p_lumen)
        self.p_lumen = float(p_lumen)
        return q_drive

    def advance_from_pressure(self, dt: float, p_lumen: float) -> float:
        """完整时间步末（压力驱动）：由 P_lumen 得 Q，再推进 P_wk，用于计算分支分流量"""
        q_drive = self.flow_from_lumen_pressure(p_lumen)
        if dt > 0.0:
            self.p_wk += (dt / self.c) * (q_drive - self.p_wk / self.r_d)
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
    area_cm2 : ndarray
        参考管腔截面积 (cm²)；轴向节点数由 ``len(area_cm2)`` 决定（含 x=0 与 x=L）
    vessel_length_cm : float
        血管长度 L (cm)
    parameters : BloodFlowParameters, optional
        物性与边界参数
    """

    def __init__(
        self,
        area_cm2: np.ndarray | list,
        vessel_length_cm: float,
        parameters: BloodFlowParameters | None = None,
        branch_frame_indices: Optional[Sequence[int]] = None,
        branch_diameters_cm: Optional[Sequence[float]] = None,
        apply_murray_scale: bool = True,
    ):
        self.raw_area = np.asarray(area_cm2, dtype=float).ravel()
        self.area_ref = self.raw_area.copy()
        self.area_ref = np.maximum(self.area_ref, 1e-8)
        self.nx = int(self.area_ref.shape[0])
        if self.nx < 3:
            raise ValueError(f"area 长度至少为 3，当前为 {self.nx}")
        self.length = float(vessel_length_cm)
        self.paras = parameters if parameters is not None else BloodFlowParameters()

        self.x = np.linspace(0.0, self.length, self.nx)
        self.dx = self.length / (self.nx - 1)

        self.beta = np.full(self.nx, float(self.paras.beta_mmhg_per_cm), dtype=float)
        self.state = np.zeros((2, self.nx))
        self.pressure = np.zeros(self.nx)
        
        self.time = 0.0
        self._outlet_dt = 0.0

        # run() 写入；n_save ≈ 1 + floor(n_steps / record_interval_steps)
        self.history_time: np.ndarray | None = None  # (n_save,) 保存时刻 (s)
        self.history_area: np.ndarray | None = None  # (n_save, nx) 截面积 A (cm²)
        self.history_flow: np.ndarray | None = None  # (n_save, nx) 流量 Q (mL/s)
        self.history_pressure: np.ndarray | None = None  # (n_save, nx) 管腔压 p (mmHg)
        self.history_branch_flow: np.ndarray | None = None  # (n_save, n_branch) 侧支出流 Q_b

        # #######其它状态初值设置#######
        # 狭窄检测（仅用于诊断与绘图标注；不修改 β）
        # β 保持 paras.beta_mmhg_per_cm 的常数值。
        sten = build_stenosis_segment(self.area_ref, frame_spacing_cm=self.dx)
        self.prox_idx = int(sten.prox_ref_idx)
        self.dist_idx = int(sten.dist_ref_idx)
        self.lesions = [sten]
        print(
            f"Detect stenosis: MLA@{sten.mla_idx} "
            f"(A={sten.mla_area_cm2:.4f} cm²), "
            f"prox_ref@{sten.prox_ref_idx}, dist_ref@{sten.dist_ref_idx}"
        )

        # 近/远端参考段平整
        smooth_area = smooth_axial_area(self.area_ref, sten)
        self.area_ref= smooth_area

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
            0,
            self.nx - 1,
            branch_frame_indices,
            branch_diameters_cm,
            apply_murray_scale=apply_murray_scale,
        )
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

        # 用未缩放阻力对齐 p_ref，再构造各终端。份额放大后
        # (Q_mean·f)·(R/f) 与 Q_mean·(R_d+R_p) 相同；若用 outlet.r_* 反推，
        # 会把已经除过份额的阻力当成整树阻力。
        q0 = float(np.asarray(self.paras.inlet_flow(0.0)).reshape(-1)[0])
        q_mean = self.paras.mean_coronary_flow_ml_s()
        r_d, r_p = self.paras.resolve_outlet_resistances()
        p_ref_user = float(self.paras.p_ref_mmhg)
        p_anchor = float(q_mean) * (float(r_d) + float(r_p))
        self.paras.p_ref_mmhg = p_anchor
        print(
            f"p_ref aligned to outlet steady lumen pressure: "
            f"{p_ref_user:.2f} → {p_anchor:.2f} mmHg (Q_mean·(R_d+R_p))"
        )
        p_ref = float(self.paras.p_ref_mmhg)

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
        q_distal_mean = q_mean * self.distal_flow_fraction
        self.outlet.reset(
            p_wk=q_distal_mean * self.outlet.r_d,
            q_drive=q0 * self.distal_flow_fraction,
        )
        # 侧支：按 P≈p_ref 时 Q_b≈Murray 份额初始化 P_wk，便于起步
        for br, wk in zip(self.side_branches, self.branch_outlets):
            q_share = q_mean * br.flow_fraction
            wk.reset(p_wk=p_ref - wk.r_p * q_share, q_drive=q_share)

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
        n_if = self.nx - 1 # 有限体积法的内部界面数 number of interfaces between cells
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

        # 摩擦项 −8πνQ/A：ν 为运动粘度 (cm²/s)，故此处不再除以 rho
        # （Poiseuille 形式 8πμ_dyn·Q/(ρ·A²)·A = 8πνQ/A 已把密度归一化掉）
        dudt[1] -= 8.0 * np.pi * self.paras.mu * state[1] / a

        # 侧支质量汇：主支连续方程 ∂A/∂t + ∂Q/∂x = −q_branch
        for k, br in enumerate(self.side_branches):
            dudt[0, br.frame_index] -= self._branch_outflow[k] / self.dx
        return dudt

    def _apply_boundaries(
        self, state: np.ndarray, advance: bool = False
    ) -> np.ndarray:
        """特征相容边界：每端只替换入射波，出射 Riemann 不变量从相邻内点外推。

        亚临界时每端恰好一条特征进入计算域。

        - 入口入射为 λ+：规定 ``Q(0)=Q_in(t)``，用内点的 ``W−=u−4c`` 反解 ``A(0)``。
          不再把面积做成零梯度 ``A(0)=A(1)``，否则边界状态与出射特征不一致，
          物理通量 ``F(U_0)`` 不是该边界的相容通量。
        - 出口入射为 λ−：Windkessel 规定管腔压（经 tube law 成为 ``A(L)``），
          用内点的 ``W+=u+4c`` 反解 ``Q(L)``，并以此流量推进 RCR。
        - 若该端出射特征不存在（入口 ``λ−≥0`` 或出口 ``λ+≤0``），退回旧做法：
          入口面积零梯度，出口保留 PDE 流量、只按压力改面积。
        """
        u = state.copy()
        rho = self.paras.momentum_rho()
        alpha = float(self.paras.alpha)

        if not advance:
            q_in = float(np.asarray(self.paras.inlet_flow(self.time)).reshape(-1)[0])
        else:
            q_in = float(
                np.asarray(self.paras.inlet_flow(self.time + self._outlet_dt)).reshape(-1)[0]
            )

        a_in = float(u[0, 1])
        q_neighbor = float(u[1, 1])
        _, lam_m = _characteristic_speeds(
            a_in, q_neighbor, float(self.beta[1]), rho, alpha
        )
        if lam_m < 0.0:
            _, w_minus = _riemann_invariants(
                a_in, q_neighbor, float(self.beta[1]), rho
            )
            u[0, 0] = area_from_flow_and_w_minus(
                q_in,
                w_minus,
                float(self.beta[0]),
                rho,
                float(self.area_ref[0]),
                a_in,
            )
        else:
            u[0, 0] = a_in
        u[1, 0] = q_in

        # 侧支：Windkessel 压力驱动分流（与主支出口同节拍推进）
        self._apply_side_branches(u, advance=advance)

        # 主支远端：出射 W+ 与 RCR 压力联立；失败则只定 A、保留 PDE 流量
        q_pde = float(u[1, -1])
        a_up = float(u[0, -2])
        q_up = float(u[1, -2])
        lam_p, _ = _characteristic_speeds(
            a_up, q_up, float(self.beta[-2]), rho, alpha
        )
        q_drive = q_pde
        used_characteristic = False
        if lam_p > 0.0:
            w_plus, _ = _riemann_invariants(
                a_up, q_up, float(self.beta[-2]), rho
            )
            q_char = _outlet_flow_from_w_plus(
                w_plus,
                p_wk=float(self.outlet.p_wk),
                r_p=float(self.outlet.r_p),
                r_d=float(self.outlet.r_d),
                compliance=float(self.outlet.c),
                dt=float(self._outlet_dt) if advance else 0.0,
                area_ref=float(self.area_ref[-1]),
                beta=float(self.beta[-1]),
                p_ref=float(self.paras.p_ref_mmhg),
                rho_mmhg=rho,
                q_guess=q_pde,
            )
            if q_char is not None:
                q_drive = float(q_char)
                used_characteristic = True

        if advance and self._outlet_dt > 0.0:
            p_out = self.outlet.advance(self._outlet_dt, q_drive)
        else:
            p_out = self.outlet.apply_substep(q_drive)
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
        if used_characteristic:
            u[1, -1] = q_drive
        return u

    @staticmethod
    def _clip_state(state: np.ndarray, area_ref: np.ndarray) -> np.ndarray:
        u = state.copy()
        u[0] = np.maximum(u[0], MINIMUM_AREA_RATIO * area_ref)
        # u[1] = np.maximum(u[1], 0.0)
        return u

    def _stable_timestep(self, state: np.ndarray) -> float:
        lam_p, lam_m = _characteristic_speeds(
            state[0], state[1], self.beta, self.paras.momentum_rho(), self.paras.alpha
        )
        lam = float(np.max(np.maximum(np.abs(lam_p), np.abs(lam_m))))
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
        verbose: bool = True,
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

        if verbose:
            print(
                f"Navier-Stokes 1D: L={self.length:.2f} cm, nx={self.nx}, "
                f"T={duration_s:.3f} s, outlet=RCR, "
                f"n_branch={len(self.side_branches)}"
            )

        t0 = time.perf_counter()
        while self.time < duration_s - 1e-15:
            if verbose:
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
        if verbose:
            print(f"完成 {step} 步, t={self.time:.4f} s, 耗时 {t1-t0:.3f} s")
            self.print_pressure_diagnostics()
            self.print_wellposedness_diagnostics()
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

    def wellposedness_diagnostics(
        self, warmup_fraction: float = 0.5
    ) -> dict[str, float]:
        """适定性诊断：以特征速度 λ± 判定亚/跨临界；并报告临界流量与面积钳位。

        一维双曲系统 ``U = [A, Q]`` 的系数矩阵
        ``[[0, 1], [c² − αu², 2αu]]`` 给出两条特征速度

            λ± = α·u ± √(c² + α(α−1)·u²)

        其中 ``u = Q/A``，``c = √(β√A/(2ρ*))``（见 :func:`_wave_speed`）。

        判据（含 α≠1）
        ----------------
        - **亚临界**：处处 ``λ₋ < 0 < λ₊``（本函数用后段时间上 ``λ₋`` 最大值 ≤ 0）
        - **跨临界/壅塞**：某处 ``λ₋ > 0``，两特征同向，出口定压 BC 不适定
        - ``transcritical`` / ``well_posed`` / ``n_choked`` 均由 ``λ₋`` 定义

        Parameters
        ----------
        warmup_fraction :
            只用仿真后段 ``(1 − warmup_fraction)`` 的数据做统计（默认后 50%）

        Returns
        -------
        dict
            ``transcritical`` / ``well_posed`` / ``n_choked`` / ``lam_minus_max`` /
            ``worst_idx`` / ``worst_x`` 等。
        """
        rho_star = rho_for_mmhg_pressure_coupling(self.paras.rho)
        alpha = float(self.paras.alpha)

        has_hist = self.history_time is not None and self.history_area is not None
        if has_hist:
            n = self.history_area.shape[0]
            i0 = int(max(n * warmup_fraction, 0))
            a = np.asarray(self.history_area[i0:], dtype=float)
            q = np.asarray(self.history_flow[i0:], dtype=float)
        else:
            a = np.asarray(self.state[0], dtype=float)[None, :]
            q = np.asarray(self.state[1], dtype=float)[None, :]

        u = q / np.maximum(a, 1e-12)
        c = np.asarray(_wave_speed(a, self.beta, rho_star), dtype=float)
        lam_p, lam_m = _characteristic_speeds(a, q, self.beta, rho_star, alpha)

        # λ₋ > 0 ⇒ 该处两特征线同向，出口 BC 无法施加
        lam_m_space = lam_m.max(axis=0)
        bad = np.where(lam_m_space > 0.0)[0]
        n_choked = int(bad.size)
        lam_minus_max = float(lam_m_space.max())
        lam_plus_min = float(lam_p.min())
        i_worst = int(np.argmax(lam_m_space))

        q_crit = float(np.mean(c * a, axis=0)[i_worst])
        q_peak = float(q.max(axis=0)[i_worst])

        # 面积钳位：A/A₀ 触到 MINIMUM_AREA_RATIO 的采样点数
        ratio = a / self.area_ref
        n_clamp = int((ratio <= MINIMUM_AREA_RATIO + 1e-6).sum())
        clamp_frac = float((ratio <= MINIMUM_AREA_RATIO + 1e-6).mean())

        return {
            "worst_idx": float(i_worst),
            "worst_x": float(self.x[i_worst]),
            "transcritical": bool(n_choked > 0),
            "n_choked": float(n_choked),
            "choked_x_min": float(self.x[bad].min()) if n_choked else float("nan"),
            "choked_x_max": float(self.x[bad].max()) if n_choked else float("nan"),
            "lam_minus_max": lam_minus_max,
            "lam_plus_min": lam_plus_min,
            "well_posed": bool(n_choked == 0),
            "q_crit_worst": q_crit,
            "q_peak_worst": q_peak,
            "c_worst": float(np.mean(c, axis=0)[i_worst]),
            "u_worst": float(np.abs(u).max(axis=0)[i_worst]),
            "n_clamp": float(n_clamp),
            "clamp_fraction": clamp_frac,
        }

    def print_wellposedness_diagnostics(
        self, warmup_fraction: float = 0.5
    ) -> dict[str, float]:
        """打印适定性诊断；返回 :meth:`wellposedness_diagnostics` 的字典。"""
        s = self.wellposedness_diagnostics(warmup_fraction=warmup_fraction)
        tag = "instant" if self.history_time is None else (
            f"max over last {1.0 - warmup_fraction:.0%}T"
        )
        iw = int(s["worst_idx"])
        print(
            f"[WP diag {tag}] lam- max={s['lam_minus_max']:+.1f}, "
            f"lam+ min={s['lam_plus_min']:+.1f} | "
            f"worst λ₋ @ idx={iw} (x={s['worst_x']:.3f} cm, "
            f"c={s['c_worst']:.1f}, u={s['u_worst']:.1f} cm/s)"
        )
        if s["well_posed"]:
            print(
                f"  流态: 亚临界 ✓ (lambda- < 0 处处成立，每端 1 个 BC 适定)"
            )
        else:
            print(
                f"  !! 跨临界/壅塞: {int(s['n_choked'])} 个节点 lambda- > 0，"
                f"x ∈ [{s['choked_x_min']:.3f}, {s['choked_x_max']:.3f}] cm；"
                f"出口定压 BC 与特征线方向矛盾"
            )
            print(
                f"     该处 Q_crit = c·A = {s['q_crit_worst']:.3f} mL/s，"
                f"实际 Q_peak = {s['q_peak_worst']:.3f} mL/s "
                f"(超 {s['q_peak_worst'] / max(s['q_crit_worst'], 1e-12):.2f} 倍)"
            )
        if s["n_clamp"] > 0:
            print(
                f"  !! 面积钳位: {int(s['n_clamp'])} 个采样点触及 "
                f"A/A0 = {MINIMUM_AREA_RATIO:.2f} 下限 ({s['clamp_fraction']:.1%})；"
                f"tube law 在该处已非线性削平"
            )
        return s

    def mass_balance(self, cycle_s: float = 0.8) -> dict[str, float]:
        """末段 ``cycle_s`` 秒的质量收支（默认一个心动周期）。

        ``mass_err = <Q(0)> − <Q(L)> − <Σ Q_b>``，与问题清单中的定义一致：
        周期状态下若格式守恒且管腔体积不再趋势性增减，该值应接近 0。
        ``dvdt`` 是同一时段管腔体积变化率 ``d/dt Σ A Δx`` 的积分均值。
        ``residual = dvdt − mass_err`` 把「流量账」和「体积账」对上。
        """
        if self.history_time is None or self.history_flow is None or self.history_area is None:
            raise RuntimeError("请先调用 run()")
        t = np.asarray(self.history_time, dtype=float)
        t0 = float(t[-1]) - float(cycle_s)
        mask = np.ones(t.shape[0], dtype=bool) if t0 <= float(t[0]) else (t >= t0)
        tt = t[mask]
        if tt.size < 2:
            raise RuntimeError("末段采样点不足，无法做积分均值")
        q_in = integral_mean(self.history_flow[mask, 0], tt)
        q_out = integral_mean(self.history_flow[mask, -1], tt)
        if self.history_branch_flow is not None and self.history_branch_flow.size:
            q_side = integral_mean(np.sum(self.history_branch_flow[mask], axis=1), tt)
        else:
            q_side = 0.0
        volume = np.asarray(self.history_area[mask], dtype=float).sum(axis=1) * self.dx
        dvdt = integral_mean(np.gradient(volume, tt), tt)
        mass_err = q_in - q_out - q_side
        return {
            "q_in": q_in,
            "q_out": q_out,
            "q_side": q_side,
            "mass_err": mass_err,
            "dvdt": dvdt,
            "residual": dvdt - mass_err,
        }

    def ffr_ratio(self, warmup_fraction: float = 0.5) -> np.ndarray:
        """沿程压力 FFR：时间平均 p(x) / p(prox)，期望单调非增。"""
        n_step = len(self.history_time)
        # 流量FFR
        # if self.history_flow is None:
        #     assert ValueError("请先调用 run()")
        # else:
        #     i0 = int(max(n_step * warmup_fraction, 0))
        #     p_mean = [integral_mean(self.history_flow[i0:, i], self.history_time[i0:]) for i in range(self.nx)]
        #     p_mean = np.asarray(p_mean, dtype=float).ravel()
        # # 近端参考流量
        # p0 = float(integral_mean(self.history_flow[i0:, self.prox_idx], self.history_time[i0:]))
        # ffr = p_mean / p0

        # 积分均值，压力沿程分布的均值计算FFR
        if self.history_flow is None:
            assert ValueError("请先调用 run()")
        else:
            i0 = int(max(n_step * warmup_fraction, 0))
            p_mean = [integral_mean(self.history_pressure[i0:, i], self.history_time[i0:]) for i in range(self.nx)]
            p_mean = np.asarray(p_mean, dtype=float).ravel()
        # 近端参考流量
        p0 = float(integral_mean(self.history_pressure[i0:, self.prox_idx], self.history_time[i0:]))
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
                name="P",
                line=dict(color="green", width=1.5, dash="dash"),
                hovertemplate="x=%{x:.3f} cm<br>P=%{y:.3f} mmHg<extra></extra>",
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        fig.update_xaxes(title_text="x (cm)", row=1, col=1)
        fig.update_yaxes(title_text="A (cm²)", row=1, col=1)
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


def run_uniform_tube_mass_check(
    nx: int = 200,
    length_cm: float = 5.8,
    area_cm2: float = 0.056729,
    duration_s: float = 2.4,
    cardiac_output_l_per_min: float = 5.0,
    verbose: bool = False,
) -> dict[str, float]:
    """方向 A：等截面直管、无侧支、无狭窄上的质量守恒检查。

    几何取当前狭窄算例的长度与近端参考面积量级。若 ``mass_err`` 相对
    ``Q_in`` 可忽略，则泄漏不是通用离散或边界通量的共性缺陷。
    """
    paras = BloodFlowParameters(cardiac_output_l_per_min=cardiac_output_l_per_min)
    solver = NavierStokes1D(
        np.full(int(nx), float(area_cm2)),
        float(length_cm),
        paras,
    )
    solver.run(duration_s=duration_s, record_interval_steps=8, verbose=verbose)
    report = solver.mass_balance(cycle_s=60.0 / paras.heart_rate_bpm)
    wp = solver.wellposedness_diagnostics(
        warmup_fraction=max(0.0, 1.0 - (60.0 / paras.heart_rate_bpm) / duration_s)
    )
    report["lam_minus_max"] = float(wp["lam_minus_max"])
    report["n_choked"] = float(wp["n_choked"])
    report["n_lesion"] = float(len(solver.lesions))
    report["n_branch"] = float(len(solver.side_branches))
    print(
        f"[uniform tube] Q_in={report['q_in']:.6f} Q_out={report['q_out']:.6f} "
        f"mass_err={report['mass_err']:.3e} dV/dt={report['dvdt']:.3e} "
        f"lam- max={report['lam_minus_max']:+.4f}, n_choked={int(report['n_choked'])}"
    )
    return report


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



def inlet_pressure_peak_mmhg(
    solver: "NavierStokes1D",
    cycle_s: float | None = None,
) -> float:
    """末段一个心动周期内入口 ``p(0,t)`` 的峰值 (mmHg)。"""
    if solver.history_time is None or solver.history_pressure is None:
        raise RuntimeError("请先调用 run()")
    t = np.asarray(solver.history_time, dtype=float)
    if cycle_s is None:
        cycle_s = 60.0 / float(solver.paras.heart_rate_bpm)
    t0 = float(t[-1]) - float(cycle_s)
    mask = np.ones(t.shape[0], dtype=bool) if t0 <= float(t[0]) else (t >= t0)
    if int(np.count_nonzero(mask)) < 2:
        raise RuntimeError("末段采样点不足，无法取入口压力峰值")
    return float(np.max(solver.history_pressure[mask, 0]))


def calibrate_q_scale_to_inlet_peak(
    area_cm2: np.ndarray | list,
    vessel_length_cm: float,
    parameters: BloodFlowParameters | None = None,
    *,
    branch_frame_indices: Optional[Sequence[int]] = None,
    branch_diameters_cm: Optional[Sequence[float]] = None,
    apply_murray_scale: bool = True,
    duration_s: float = 3.0,
    record_interval_steps: int = 30,
    peak_lo_mmhg: float = 90.0,
    peak_hi_mmhg: float = 105.0,
    q_scale_min: float = 0.05,
    q_scale_max: float = 2.0,
    max_iters: int = 5,
    verbose: bool = True,
    cache_dir: str | None = None,
) -> tuple["NavierStokes1D", dict]:
    """外环二分标定 ``q_scale``，使入口压力峰值落入 ``(peak_lo, peak_hi)``。

    不改 ``coronary_flow_fraction``：生成入口时用 ``fraction * q_scale``。
    假定定流量入口下峰值压随流量近似单调：峰值过高则减小 ``q_scale``，过低则增大。

    若给定 ``cache_dir``，每次迭代写入 ``iter_XX.npz`` + ``summary.json``；
    事后读图用 ``python calibration_cache.py <cache_dir>``。

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
            f"max_iters={max_iters}"
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
        solver.run(
            duration_s=duration_s,
            record_interval_steps=record_interval_steps,
            verbose=True,
        )
        peak = inlet_pressure_peak_mmhg(solver)
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


def example(
    target_nx: int | None = 100,
    frame_spacing_cm: float = 0.02,
    duration_s: float = 3.0,
    sigma_nodes: float = 4,
    name: str = 'Null',
):
    """下游 joblib 数据示例；``target_nx`` 控制轴向降采样（None=不降采样）。

    先用 ``calibrate_q_scale_to_inlet_peak`` 把入口峰值压收到目标窗，
    再在该 ``q_scale`` 下绘图并返回 FFR。
    """
    import joblib
    import os
    np.random.seed(2034)
    mm_per_pixel = (9.7 / 756)

    dirname = f"./data/stenoses_data"
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
    cache_dir = os.path.join(
        "data",
        "cache",
        f"q_scale_cal_{name}_nx{nx}_T{duration_s:.2f}",
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


if __name__ == "__main__":
    # target_nx=100 降采样；改为 None 可跑原始分辨率对比准确率
    ffr = example(target_nx=200, name="stenoses")
    print(f"max ffr: {np.max(ffr)}\n min ffr: {np.min(ffr)}")
