"""
冠状动脉出口管腔内压力 — Windkessel 边界条件

生理背景
--------
出口处管腔压由动脉末端与微循环之间的 Windkessel 等效电路决定；
以冠脉入口脉动流量 Q(t) 为驱动，积分得到出口压力 P_out(t)（mmHg）。
静脉压相对动脉/冠脉管腔压可忽略，电路出口端按零压参考处理。

二元 Windkessel（RC：远端阻力 + 顺应性）
    C · dP/dt = Q(t) − P / R_d
    出口管腔压即顺应性节点压力 P。

三元 Windkessel（RCR：近端阻力 + 远端阻力 + 顺应性）
    C · dP_wk/dt = Q(t) − P_wk / R_d
    P_out(t) = P_wk(t) + R_p · Q(t)
    R_p 表征近端特征阻抗/反射，P_wk 为微循环顺应性节点压。
"""

import numpy as np
from typing import Literal

from coronary_constants import P_INLET_REF_MMHG
from coronary_inlet import coronary_inlet_flow

# 默认微循环参数（以下划线开头的常量仅在本模块内使用）
_DEFAULT_C_ML_PER_MMHG = 0.08
_DEFAULT_Q_MEAN_ML_S = 3


def _default_distal_resistance(
    q_mean_ml_s: float = _DEFAULT_Q_MEAN_ML_S,
    p_ref_mmhg: float = P_INLET_REF_MMHG,
) -> float:
    """参数 R_d 由平均流量与参考压标定远端阻力 R_d (mmHg·s/mL)。"""
    return p_ref_mmhg / max(q_mean_ml_s, 1e-6)


def _default_proximal_resistance(
    r_distal_mmhg_s_per_ml: float,
    proximal_fraction: float = 0.12,
) -> float:
    """参数 R_p 近端阻力取远端阻力的一定比例（特征阻抗量级）。"""
    return proximal_fraction * r_distal_mmhg_s_per_ml

"""
上述两个函数仅在未提供r_distal_mmhg_s_per_ml参数时使用。
"""


def _steady_windkessel_pressure(
    q_mean_ml_s: float,
    r_distal_mmhg_s_per_ml: float,
    r_proximal_mmhg_s_per_ml: float = 0.0,
) -> tuple[float, float]:
    """通过流量输入计算的压力稳态顺应性节点压 P_wk 与出口管腔压 P_out初始值。"""
    p_wk = q_mean_ml_s * r_distal_mmhg_s_per_ml
    p_out = p_wk + q_mean_ml_s * r_proximal_mmhg_s_per_ml
    return p_wk, p_out


def _integrate_windkessel2(
    t: np.ndarray,
    q: np.ndarray,
    r_distal: float,
    compliance: float,
    p_init: float | None = None,
) -> np.ndarray:
    """二元 Windkessel 数值积分（RK4），返回出口管腔压 P_out(t)。"""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if t.size < 2:
        p0 = p_init if p_init is not None else q[0] * r_distal
        return np.full_like(t, p0, dtype=float)

    if p_init is None:
        p_init, _ = _steady_windkessel_pressure(float(np.mean(q)), r_distal)

    p = np.empty_like(t)
    p[0] = float(p_init)
    rd = max(float(r_distal), 1e-9)
    c = max(float(compliance), 1e-9)

    def rhs(ti: float, pi: float) -> float:
        qi = float(np.interp(ti, t, q))  # 线性插值计算某个单元点的流量q
        return (qi - pi / rd) / c

    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        if dt <= 0.0:
            p[i] = p[i - 1]
            continue
        pi = p[i - 1]
        k1 = rhs(t[i - 1], pi)
        k2 = rhs(t[i - 1] + 0.5 * dt, pi + 0.5 * dt * k1)
        k3 = rhs(t[i - 1] + 0.5 * dt, pi + 0.5 * dt * k2)
        k4 = rhs(t[i], pi + dt * k3)
        p[i] = pi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return p


def _integrate_windkessel3(
    t: np.ndarray,
    q: np.ndarray,
    r_proximal: float,
    r_distal: float,
    compliance: float,
    p_wk_init: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """三元 Windkessel：积分 P_wk，出口管腔压 P_out = P_wk + R_p·Q。"""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    rp = float(r_proximal)

    p_wk = _integrate_windkessel2(
        t, q, r_distal, compliance, p_init=p_wk_init
    )
    p_out = p_wk + rp * q
    return p_out, p_wk


def windkessel2_outlet_pressure(
    t,
    q_in,
    r_distal_mmhg_s_per_ml: float | None = None,
    compliance_ml_per_mmhg: float = _DEFAULT_C_ML_PER_MMHG,
    p_init_mmhg: float | None = None,
):
    """
    二元 Windkessel 出口管腔压 P_out(t)（mmHg）。

    Parameters
    ----------
    t : array
        时间 (s)
    q_in : array
        驱动流量 Q(t) (mL/s)，与 t 等长
    r_distal_mmhg_s_per_ml : float, optional
        远端阻力 R_d；默认由平均流量与参考压标定
    compliance_ml_per_mmhg : float
        顺应性 C (mL/mmHg)
    p_init_mmhg : float, optional
        初始管腔压；默认取稳态值
    """
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q_in, dtype=float)
    q_mean = float(np.mean(q_arr))
    rd = (
        _default_distal_resistance(q_mean)
        if r_distal_mmhg_s_per_ml is None
        else float(r_distal_mmhg_s_per_ml)
    )
    if p_init_mmhg is None:
        p_init_mmhg, _ = _steady_windkessel_pressure(q_mean, rd)
    return _integrate_windkessel2(
        t_arr,
        q_arr,
        rd,
        compliance_ml_per_mmhg,
        p_init=p_init_mmhg,
    )


def windkessel3_outlet_pressure(
    t,
    q_in,
    r_proximal_mmhg_s_per_ml: float | None = None,
    r_distal_mmhg_s_per_ml: float | None = None,
    compliance_ml_per_mmhg: float = _DEFAULT_C_ML_PER_MMHG,
    proximal_fraction: float = 0.2,
    p_wk_init_mmhg: float | None = None,
):
    """
    三元 Windkessel 出口管腔压 P_out(t)（mmHg）。

    返回 (P_out, P_wk)，P_wk 为顺应性节点（微循环）压力。
    """
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q_in, dtype=float)
    q_mean = float(np.mean(q_arr))
    rd = (
        _default_distal_resistance(q_mean)
        if r_distal_mmhg_s_per_ml is None
        else float(r_distal_mmhg_s_per_ml)
    )
    rp = (
        _default_proximal_resistance(rd, proximal_fraction)
        if r_proximal_mmhg_s_per_ml is None
        else float(r_proximal_mmhg_s_per_ml)
    )
    if p_wk_init_mmhg is None:
        p_wk_init_mmhg, _ = _steady_windkessel_pressure(q_mean, rd)
    return _integrate_windkessel3(
        t_arr,
        q_arr,
        rp,
        rd,
        compliance_ml_per_mmhg,
        p_wk_init=p_wk_init_mmhg,
    )


def coronary_outlet_flow_distribution(Q_total, outlet_radii, method='murray',
                                     flow_fractions=None, cardiac_phase=None):
    """
    冠状动脉出口流量分配函数。
    
    基于解剖学缩放法则和生理学原理，将总冠状动脉流量分配到各个出口。
    
    参考文献：
    1. Murray CD. "The physiological principle of minimum work applied to 
       the angle of branching of arteries." J Gen Physiol, 1926;9(6):835-841.
    2. van der Giessen AG, et al. "The influence of boundary conditions on 
       wall shear stress distribution in patient specific coronary trees."
       J Biomech, 2011;44(6):1089-1095.
    3. Taylor CA, et al. "Patient-specific modeling of cardiovascular 
       mechanics." Annu Rev Biomed Eng, 2009;11:109-134.
    
    参数：
    Q_total : float or array
        总冠状动脉入口流量(mL/s)，可以是标量或时间序列
    outlet_radii : array
        各出口血管半径(cm)
    method : str
        分配方法：
        - 'murray': Murray's Law, Q ∝ r³
        - 'area': 面积比例, Q ∝ r²
        - 'custom': 自定义流量分数
    flow_fractions : array, optional
        自定义流量分数(method='custom'时使用)，应为总和为1的数组

    返回：
    Q_outs : array
        各出口分配的流量(mL/s)，形状为(len(outlet_radii), len(t))
    """
    
    # 处理输入
    Q_total_arr = np.atleast_1d(Q_total)
    n_outlets = len(outlet_radii)
    n_time = len(Q_total_arr)
    
    # 计算出口半径
    radii = np.array(outlet_radii)
    
    # 根据不同方法计算分配权重
    if method == 'murray':
        # Murray's Law: 父血管半径的立方等于子血管半径立方之和
        # Q ∝ r³
        weights = radii ** 3
        
    elif method == 'area':
        # 面积比例分配: Q ∝ r²
        weights = radii ** 2
        
    elif method == 'murray_modified':
        # 改进的Murray's Law (考虑冠脉特殊性)
        # 指数通常在2.27-3.0之间
        exponent = 2.7  # 冠状动脉常用指数
        weights = radii ** exponent
        
    elif method == 'custom':
        if flow_fractions is None:
            raise ValueError("使用'custom'方法时必须提供flow_fractions参数")
        weights = np.array(flow_fractions)
        
    else:
        raise ValueError(f"不支持的分配方法: {method}")
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 分配流量
    Q_outs = np.outer(weights, Q_total_arr).T
    
    return Q_outs.squeeze()


def coronary_outlet_pressure(
    t,
    model: Literal["2wk", "3wk", "rc", "rcr"] = "2wk",
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.03,
    q_in: np.ndarray | None = None,
    **windkessel_kwargs,
):
    """
    冠脉出口管腔压边界条件：入口傅里叶流量 + Windkessel。

    Parameters
    ----------
    t : array
        时间 (s)
    model : {'2wk', '3wk', 'rc', 'rcr'}
        '2wk'/'rc'：二元；'3wk'/'rcr'：三元
    heart_rate, cardiac_output, coronary_fraction
        用于生成 Q(t)（当未提供 q_in 时）
    q_in : array, optional
        自定义驱动流量 (mL/s)
    **windkessel_kwargs
        传给 windkessel2/3_outlet_pressure 的 R、C 等

    Returns
    -------
    2wk : ndarray, 出口管腔压 P_out(t)
    3wk : tuple (P_out, P_wk)
    """
    t_arr = np.asarray(t, dtype=float)
    if q_in is None:
        q_arr = coronary_inlet_flow(
            t_arr,
            heart_rate=heart_rate,
            cardiac_output=cardiac_output,
            coronary_fraction=coronary_fraction,
        )
    else:
        q_arr = np.asarray(q_in, dtype=float)

    m = model.lower()
    if m in ("2wk", "rc"):
        return windkessel2_outlet_pressure(t_arr, q_arr, **windkessel_kwargs)
    if m in ("3wk", "rcr"):
        return windkessel3_outlet_pressure(t_arr, q_arr, **windkessel_kwargs)
    raise ValueError(f"不支持的 Windkessel 模型: {model!r}，请使用 '2wk' 或 '3wk'")


def coronary_outlet_pressure_stats(
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.03,
    duration_cycles: float = 3.0,
    model: Literal["2wk", "3wk"] = "2wk",
    **kwargs,
) -> dict:
    """单模型出口压力统计：均值、峰谷、脉动指数等。"""
    period = 60.0 / heart_rate
    t = np.linspace(0.0, duration_cycles * period, int(2000 * duration_cycles))
    q = coronary_inlet_flow(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ("fourier_coefficients", "min_flow_fraction", "phase_offset")
        },
    )
    if model == "2wk":
        p = windkessel2_outlet_pressure(t, q, **kwargs)
        p_wk = None
    else:
        p, p_wk = windkessel3_outlet_pressure(t, q, **kwargs)

    p_mean = float(np.mean(p))
    p_max = float(np.max(p))
    p_min = float(np.min(p))
    out = {
        "model": model,
        "P_mean_mmHg": p_mean,
        "P_max_mmHg": p_max,
        "P_min_mmHg": p_min,
        "Q_mean_mL_s": float(np.mean(q)),
    }
    if p_wk is not None:
        out["P_wk_mean_mmHg"] = float(np.mean(p_wk))
        out["P_wk_max_mmHg"] = float(np.max(p_wk))
        out["P_wk_min_mmHg"] = float(np.min(p_wk))
    return out


def plot_coronary_outlet_pressure(
    heart_rate: float = 75.0,
    cardiac_output: float = 5.0,
    coronary_fraction: float = 0.03,
    duration_s: float = 4.0,
    r_distal_mmhg_s_per_ml: float | None = None,
    r_proximal_mmhg_s_per_ml: float | None = None,
    compliance_ml_per_mmhg: float = _DEFAULT_C_ML_PER_MMHG,
):
    """对比二元 / 三元 Windkessel 出口管腔压与驱动流量。"""
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

    t = np.linspace(0.0, duration_s, int(1000 * duration_s))
    q = coronary_inlet_flow(
        t,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )
    p2 = windkessel2_outlet_pressure(t, q, r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml, compliance_ml_per_mmhg=compliance_ml_per_mmhg)
    p3, p_wk3 = windkessel3_outlet_pressure(
        t,
        q,
        r_proximal_mmhg_s_per_ml=r_proximal_mmhg_s_per_ml,
        r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml,
        compliance_ml_per_mmhg=compliance_ml_per_mmhg,
    )

    stats2 = coronary_outlet_pressure_stats(
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        model="2wk",
        r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml,
        compliance_ml_per_mmhg=compliance_ml_per_mmhg,
    )
    stats3 = coronary_outlet_pressure_stats(
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
        model="3wk",
        r_proximal_mmhg_s_per_ml=r_proximal_mmhg_s_per_ml,
        r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml,
        compliance_ml_per_mmhg=compliance_ml_per_mmhg,
    )

    period = 60.0 / heart_rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax0 = axes[0, 0]
    ax0.plot(t, q, "b-", lw=1.8, label="入口流量 Q(t)")
    ax0.set_xlabel("时间 (s)")
    ax0.set_ylabel("流量 (mL/s)")
    ax0.set_title("冠脉入口驱动流量")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1 = axes[0, 1]
    ax1.plot(t, p2, "C0", lw=2, label=f"二元 WK")
    ax1.plot(t, p3, "C1", lw=2, label=f"三元 WK")
    ax1.axhline(stats2["P_mean_mmHg"], color="C0", ls="--", alpha=0.6)
    ax1.axhline(stats3["P_mean_mmHg"], color="C1", ls="--", alpha=0.6)
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("压力 (mmHg)")
    ax1.set_title("出口管腔内压力 P_out(t)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1, 0]
    t_one = np.linspace(0.0, period, 500, endpoint=False)
    q_one = coronary_inlet_flow(
        t_one,
        heart_rate=heart_rate,
        cardiac_output=cardiac_output,
        coronary_fraction=coronary_fraction,
    )
    p2_one = windkessel2_outlet_pressure(t_one, q_one, r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml,
        compliance_ml_per_mmhg=compliance_ml_per_mmhg,)
    p3_one, _ = windkessel3_outlet_pressure(
        t_one, q_one, r_proximal_mmhg_s_per_ml=r_proximal_mmhg_s_per_ml, r_distal_mmhg_s_per_ml=r_distal_mmhg_s_per_ml,
        compliance_ml_per_mmhg=compliance_ml_per_mmhg,
    )
    ax2.plot(t_one, p2_one, "C0", lw=2, label="二元 WK")
    ax2.plot(t_one, p3_one, "C1", lw=2, label="三元 WK")
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("压力 (mmHg)")
    ax2.set_title("单周期出口压力")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 1]
    ax3.plot(t, p_wk3, "C2", lw=1.8, label="三元 WK：P_wk（顺应性节点）")
    ax3.plot(t, p3 - p_wk3, "C3", lw=1.2, ls="--", label="R_p·Q 近端压降")
    ax3.set_xlabel("时间 (s)")
    ax3.set_ylabel("压力 (mmHg)")
    ax3.set_title("三元模型分解")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle(
        f"HR={heart_rate} bpm  |  Q_mean≈{stats2['Q_mean_mL_s']:.2f} mL/s  |  R_d = {r_distal_mmhg_s_per_ml} mmHg·s/mL"
        f"二元均值 {stats2['P_mean_mmHg']:.1f} mmHg  |  三元均值 {stats3['P_mean_mmHg']:.1f} mmHg",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    print("二元 Windkessel 出口压力:")
    for k, v in stats2.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n三元 Windkessel 出口压力:")
    for k, v in stats3.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    hr, co, frac = 75.0, 5.5, 0.03
    duration_s = 15
    r_d = None
    # rp, rd 可根据提供的流量进行估算，也可以直接给出
    t = np.linspace(0.0, duration_s, int(800 * duration_s))

    p2 = coronary_outlet_pressure(
        t, model="2wk", heart_rate=hr, cardiac_output=co, coronary_fraction=frac,
        r_distal_mmhg_s_per_ml=r_d,
    )
    p3, p_wk = coronary_outlet_pressure(
        t, model="3wk", heart_rate=hr, cardiac_output=co, coronary_fraction=frac,
        r_distal_mmhg_s_per_ml=r_d,
    )

    plot_coronary_outlet_pressure(
        heart_rate=hr, cardiac_output=co, coronary_fraction=frac, duration_s=duration_s,
        r_distal_mmhg_s_per_ml=r_d,
    )
