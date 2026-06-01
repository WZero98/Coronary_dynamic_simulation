"""
冠状动脉入口脉动血流量 — 傅里叶级数拟合模型

生理要点（心肌挤压效应）：
- 收缩期：心室收缩时心肌挤压冠脉，壁内小分支可压闭，流量骤降甚至短暂停滞
- 舒张期：心肌松弛、外压解除、阻力最低，主动脉舒张压驱动血液快速灌注
- 约 80%–90% 灌注发生在舒张期

数学形式（周期 T = 60/HR）：
    Q(t) = Q_mean + Σ_n  A_n sin(n·ω·t_mod + φ_n)
    A_n = (amp_ratio_n) · Q_mean，系数由生理模板波形最小二乘 / FFT 拟合得到。
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

# 拟合时使用的谐波数
_DEFAULT_N_HARMONICS = 7

# 默认生理模板参数（经标定：PI ∈ [0.9, 1.3]，收缩峰/舒张峰 ≈ 70%）
_DEFAULT_SYSTOLIC_DURATION_FRAC = 0.30
_DEFAULT_DIASTOLIC_PEAK = 1.2
_DEFAULT_SYSTOLIC_TO_DIASTOLIC_PEAK_RATIO = 0.65
_DEFAULT_BASELINE_LEVEL = 0.30
_DEFAULT_SYSTOLIC_EARLY_PEAK_FRAC = 0.1
_DEFAULT_JUNCTION_FLOW_FRAC = 0.7
_DEFAULT_DIASTOLIC_PEAK_FRAC = 0.3


def coronary_flow_template(
    tau: np.ndarray,
    systolic_duration_frac: float = _DEFAULT_SYSTOLIC_DURATION_FRAC,
    diastolic_peak: float = _DEFAULT_DIASTOLIC_PEAK,
    systolic_to_diastolic_peak_ratio: float = _DEFAULT_SYSTOLIC_TO_DIASTOLIC_PEAK_RATIO,
    baseline_level: float = _DEFAULT_BASELINE_LEVEL,
    systolic_early_peak_frac: float = _DEFAULT_SYSTOLIC_EARLY_PEAK_FRAC,
    junction_flow_frac: float = _DEFAULT_JUNCTION_FLOW_FRAC,
    diastolic_peak_frac: float = _DEFAULT_DIASTOLIC_PEAK_FRAC,
    systolic_level: float | None = None,
) -> np.ndarray:
    """
    单周期归一化生理模板（均值约为 1.0），用于傅里叶拟合。

    波形构造要点
    ------------
    1. 收缩早期峰值 = ``systolic_to_diastolic_peak_ratio`` × 舒张峰值（默认 70%）。
    2. 收缩晚期不再跌至波谷：峰值后经由平滑升支在收缩期末达到 ``junction_flow_frac``×舒张峰，
       与舒张期升支 C⁰ 连续融合（未降至局部最低即进入舒张灌注）。
    3. 舒张早期升至峰值，末期回落至基线，提高脉动幅度且保持 PI 标定空间。
    """
    tau = np.asarray(tau, dtype=float) % 1.0
    t_sys_end = float(systolic_duration_frac)
    d_peak = float(diastolic_peak)
    peak_ratio = float(systolic_to_diastolic_peak_ratio)

    baseline = float(baseline_level) * d_peak
    s_peak = peak_ratio * d_peak if systolic_level is None else float(systolic_level)
    q_join = max(float(junction_flow_frac) * d_peak, s_peak + 1e-6)
    t_dia_peak = float(np.clip(diastolic_peak_frac, 0.08, 0.55))

    q = np.empty_like(tau)
    sys_mask = tau < t_sys_end
    dia_mask = ~sys_mask

    t_norm = tau[sys_mask] / max(t_sys_end, 1e-6)
    t_sp = float(np.clip(systolic_early_peak_frac, 0.05, 0.45))

    early = t_norm <= t_sp
    late = ~early
    q_sys = np.empty_like(t_norm)
    q_sys[early] = baseline + (s_peak - baseline) * np.sin(0.5 * np.pi * t_norm[early] / t_sp)
    u = (t_norm[late] - t_sp) / max(1.0 - t_sp, 1e-6)
    q_sys[late] = s_peak + (q_join - s_peak) * 0.5 * (1.0 - np.cos(np.pi * u))
    q[sys_mask] = q_sys

    t_dia = (tau[dia_mask] - t_sys_end) / max(1.0 - t_sys_end, 1e-6)
    rise = t_dia <= t_dia_peak
    fall = ~rise
    q_dia = np.empty_like(t_dia)
    q_dia[rise] = q_join + (d_peak - q_join) * np.sin(
        0.5 * np.pi * t_dia[rise] / t_dia_peak
    )
    v = (t_dia[fall] - t_dia_peak) / max(1.0 - t_dia_peak, 1e-6)
    q_dia[fall] = d_peak - (d_peak - baseline) * np.sin(0.5 * np.pi * v)
    q[dia_mask] = q_dia

    q = q / np.mean(q)
    return q


def template_peak_metrics(
    tau: np.ndarray | None = None,
    systolic_duration_frac: float = _DEFAULT_SYSTOLIC_DURATION_FRAC,
    **kwargs,
) -> dict[str, float]:
    """模板单周期峰值指标：早期收缩峰/舒张峰、融合点流量等。以方便调节模版参数。"""
    if tau is None:
        tau = np.linspace(0.0, 1.0, 4096, endpoint=False)
    t_sp = float(
        kwargs.get("systolic_early_peak_frac", _DEFAULT_SYSTOLIC_EARLY_PEAK_FRAC)
    )

    q = coronary_flow_template(
        tau, systolic_duration_frac=systolic_duration_frac, **kwargs
    )
    t_sys = float(systolic_duration_frac)
    sys_mask = tau < t_sys
    dia_mask = ~sys_mask
    early_sys_mask = tau < (t_sp * t_sys)
    q_early_sys_peak = (
        float(np.max(q[early_sys_mask])) if np.any(early_sys_mask) else 0.0
    )
    q_dia_peak = float(np.max(q[dia_mask])) if np.any(dia_mask) else 0.0
    join_idx = int(np.clip(np.searchsorted(tau, t_sys) - 1, 0, len(tau) - 1))
    late_start = int(0.55 * join_idx)
    return {
        "systolic_early_peak": q_early_sys_peak,
        "diastolic_peak": q_dia_peak,
        "peak_ratio": q_early_sys_peak / q_dia_peak if q_dia_peak > 0 else 0.0,
        "flow_at_systole_end": float(q[join_idx]),
        "pre_systole_end_local_min": float(
            np.min(q[late_start : join_idx + 1]) if join_idx > late_start else q[0]
        ),
    }


def fit_fourier_coefficients(
    n_harmonics: int = _DEFAULT_N_HARMONICS,
    n_samples: int = 2048,
    **template_kwargs,
) -> list[tuple[int, float, float]]:
    """
    对生理模板波形做傅里叶级数拟合（正弦谐波 + 直流分量分离）。

    返回 [(n, amp_ratio, phase_rad), ...]，满足
    template(τ) ≈ 1 + Σ amp_ratio_n sin(2π n τ + phase_n)
    """
    tau = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    y = coronary_flow_template(tau, **template_kwargs)
    y_ac = y - 1.0  # 去除直流，拟合脉动部分

    coeffs: list[tuple[int, float, float]] = []
    for n in range(1, n_harmonics + 1):
        sin_n = np.sin(2 * np.pi * n * tau)
        cos_n = np.cos(2 * np.pi * n * tau)
        a_n = 2.0 * np.mean(y_ac * sin_n)
        b_n = 2.0 * np.mean(y_ac * cos_n)
        amp = np.hypot(a_n, b_n)
        phase = np.arctan2(b_n, a_n)
        amp_ratio = amp
        coeffs.append((n, float(amp_ratio), float(phase)))

    return coeffs


def _get_fourier_coefficients(
    fourier_coefficients: Sequence[tuple[int, float, float]] | None,
) -> tuple[tuple[int, float, float], ...]:
    if fourier_coefficients is not None:
        return tuple(fourier_coefficients)
    return _DEFAULT_FOURIER_COEFFICIENTS


def _build_default_fourier_coefficients() -> tuple[tuple[int, float, float], ...]:
    return tuple(fit_fourier_coefficients())


_DEFAULT_FOURIER_COEFFICIENTS: tuple[tuple[int, float, float], ...] = (
    (1, 0.41, -1.09),
    (2, 0.13, 0.72),
    (3, 0.09, -0.61),
    (4, 0.03, -0.071),
    (5, 0.03, -0.32),
    (6, 0.03, -0.24),
    (7, 0.02, -0.77)
) if 1 else _build_default_fourier_coefficients()  # 已通过模版拟合出傅里叶系数后不再需要每次重新计算


def coronary_inlet_flow(
    t,
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.045,
    fourier_coefficients: Sequence[tuple[int, float, float]] | None = None,
    min_flow_fraction: float = 0.1,
    phase_offset: float = 0.0,
):
    """
    傅里叶级数模拟冠状动脉入口血流量 Q(t)（mL/s）。

    Parameters
    ----------
    t : float or array
        时间 (s)
    heart_rate : float
        心率 (bpm)
    cardiac_output : float
        心输出量 (L/min)
    coronary_fraction : float
        冠脉血流量占心输出量比例，约 4–5%
    fourier_coefficients : sequence of (n, amp_ratio, phase_rad), optional
        谐波系数；默认使用对生理模板拟合得到的系数
    min_flow_fraction : float
        流量下限 = min_flow_fraction × Q_mean（避免非物理负流量）
    phase_offset : float
        波形整体相位偏移 (rad)，用于与 ECG/压力波形对齐

    Returns
    -------
    Q : float or ndarray
        冠脉入口流量 (mL/s)，非负
    """
    t_arr = np.asarray(t, dtype=float)
    cardiac_period = 60.0 / heart_rate
    omega = 2.0 * np.pi / cardiac_period
    q_mean = cardiac_output * 1000.0 * coronary_fraction / 60.0

    t_mod = np.mod(t_arr, cardiac_period)
    q = np.full_like(t_arr, q_mean, dtype=float)

    fourier_coefficients_set = _get_fourier_coefficients(fourier_coefficients)
    for n, amp_ratio, phase in fourier_coefficients_set:
        q += q_mean * amp_ratio * np.sin(n * omega * t_mod + phase + phase_offset)

    q_min = min_flow_fraction * q_mean
    return np.maximum(q, q_min)


def diastolic_flow_fraction(
    t,
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.045,
    systolic_duration_frac: float = _DEFAULT_SYSTOLIC_DURATION_FRAC,
    **kwargs,
) -> float:
    """
    计算单个心动周期内舒张期灌注量占总量比例。
    """
    period = 60.0 / heart_rate
    n = 2000
    tau = np.linspace(0.0, period, n, endpoint=False)
    q = coronary_inlet_flow(
        tau,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        **kwargs,
    )
    t_sys = systolic_duration_frac * period
    systolic_mask = tau < t_sys
    q_sys = np.trapezoid(q[systolic_mask], tau[systolic_mask])
    q_total = np.trapezoid(q, tau)
    if q_total <= 0:
        return 0.0
    return float(1.0 - q_sys / q_total)


def coronary_inlet_flow_stats(
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.045,
    duration_cycles: float = 3.0,
    **kwargs,
) -> dict:
    """返回平均/峰/谷流量、脉动指数、舒张期灌注比例等统计量。"""
    period = 60.0 / heart_rate
    t = np.linspace(0.0, duration_cycles * period, int(2000 * duration_cycles))
    q = coronary_inlet_flow(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        **kwargs,
    )
    q_mean = float(np.mean(q))
    q_max = float(np.max(q))
    q_min = float(np.min(q))
    pi = (q_max - q_min) / q_mean if q_mean > 0 else 0.0
    dia_frac = diastolic_flow_fraction(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        **kwargs,
    )
    return {
        "Q_mean_mL_s": q_mean,
        "Q_max_mL_s": q_max,
        "Q_min_mL_s": q_min,
        "pulsatility_index": pi,
        "diastolic_flow_fraction": dia_frac,
    }


def plot_coronary_flow(
    heart_rate: float = 65.,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.03,
    duration_s: float = 3.0,
):
    """绘制傅里叶冠脉入口流量及单周期模板对比。"""
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

    period = 60.0 / heart_rate
    t = np.linspace(0.0, duration_s, int(1000 * duration_s))
    q = coronary_inlet_flow(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )

    tau_one = np.linspace(0.0, 1.0, 500, endpoint=False)
    template = coronary_flow_template(tau_one) * np.mean(q)

    stats = coronary_inlet_flow_stats(
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1 = axes[0]
    ax1.plot(t, q, "b-", lw=2, label="傅里叶级数 Q(t)")
    ax1.axhline(stats["Q_mean_mL_s"], color="r", ls="--", label=f"均值 {stats['Q_mean_mL_s']:.2f} mL/s")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("血流量 (mL/s)")
    ax1.set_title("冠状动脉入口流量（傅里叶拟合）")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    t_one = tau_one * period
    q_one = coronary_inlet_flow(
        t_one,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )
    ax2.plot(t_one, q_one, "b-", lw=2, label="傅里叶重建")
    ax2.plot(t_one, template, "k--", lw=1.5, alpha=0.7, label="生理模板（标定后）")
    ax2.axvline(
        _DEFAULT_SYSTOLIC_DURATION_FRAC * period,
        color="gray",
        ls=":",
        label=f"收缩期结束 (~{_DEFAULT_SYSTOLIC_DURATION_FRAC*100:.0f}% 周期)",
    )
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("血流量 (mL/s)")
    ax2.set_title("单周期：模板 vs 傅里叶")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"舒张期灌注占比 ≈ {stats['diastolic_flow_fraction']*100:.1f}%  |  PI = {stats['pulsatility_index']:.2f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    print("冠状动脉血流（傅里叶模型）:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    refit = fit_fourier_coefficients()
    print("默认傅里叶系数（由模板拟合）:")
    for row in refit:
        print(f"  {row}")
    print()
    print("模板峰值指标:", template_peak_metrics())
    
    heart_rate: float = 75.
    cardiac_output: float = 5.
    coronary_fraction: float = 0.03
    duration_s: float = 2.4
    t = np.linspace(0.0, duration_s, int(1000 * duration_s))
    q = coronary_inlet_flow(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )
    print(len(q))
    plot_coronary_flow(heart_rate=heart_rate, cardiac_output=cardiac_output, coronary_fraction=coronary_fraction, duration_s=duration_s)