"""
血流/压力单位换算（求解器内部：压力展示与 tube law 用 mmHg；动量通量项临时换算为 cgs）。
"""

import numpy as np

MMHG_TO_CGS = 1333.22  # dyne/cm² per mmHg
CGS_TO_MMHG = 1.0 / MMHG_TO_CGS

# 血管入口参考平均脉压
P_INLET_REF_MMHG = 80.0
P_VENOUS_DEFAULT_MMHG = 5.0


def pressure_mmhg_to_cgs(p_mmhg):
    return np.asarray(p_mmhg, dtype=float) * MMHG_TO_CGS


def pressure_cgs_to_mmhg(p_cgs):
    return np.asarray(p_cgs, dtype=float) * CGS_TO_MMHG
