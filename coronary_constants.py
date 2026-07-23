"""
冠状动脉 1D 血流 — 模块间共享的公开全局常量

仅存放各脚本中以「非下划线」名称定义的模块级常量，供
coronary_inlet、coronary_outlet、navier_stokes_1d 等统一引用。
模块内部默认值（名称以下划线开头）仍保留在各自脚本中。
"""

from typing import Final

# 压力单位换算（mmHg ↔ cgs / dyne·cm⁻²）
MMHG_TO_DYNE_PER_CM2: Final[float] = 1333.22
DYNE_PER_CM2_TO_MMHG: Final[float] = 1.0 / MMHG_TO_DYNE_PER_CM2

# 入口/出口 tube law 与 Windkessel 阻力标定用的参考管腔压 (mmHg)
P_INLET_REF_MMHG: Final[float] = 70.0

# 最小管腔面积占参考面积比例
MINIMUM_AREA_RATIO = 0.8


def rho_for_mmhg_pressure_coupling(rho_g_per_cm3: float) -> float:
    """
    动量通量与波速公式中，与 mmHg 制管腔压配对的有效密度 ρ*。

    守恒方程在 CGS 下要求 p 以 dyne/cm² 代入 F₁ = αQ²/A + pA/ρ。
    若 tube law 输出 p 为 mmHg，等价写法为 F₁ = αQ²/A + p_mmHg·A/ρ*，
    其中 ρ* = ρ / MMHG_TO_DYNE_PER_CM2。换算在参数层完成一次即可。
    """
    return float(rho_g_per_cm3) / MMHG_TO_DYNE_PER_CM2


__all__ = [
    "MMHG_TO_DYNE_PER_CM2",
    "DYNE_PER_CM2_TO_MMHG",
    "P_INLET_REF_MMHG",
    "MINIMUM_AREA_RATIO",
    "rho_for_mmhg_pressure_coupling",
]
