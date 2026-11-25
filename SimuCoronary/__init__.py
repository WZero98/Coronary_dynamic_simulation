"""
SimuCoronary — 冠状动脉一维血流模拟包

模块划分
--------
- ``navier_stokes_1d``   1D 求解器主体（``NavierStokes1D``、``BloodFlowParameters`` 等）
- ``calculators``        高级接口：外环 ``q_scale`` 标定（``calibrate_q_scale_to_inlet_peak``）
- ``geometry``           近/远端参考选取、狭窄检测、虚拟支架、沿程重采样
- ``branch``             Murray 侧支划分
- ``coronary_inlet``     入口脉动流量波形（生理模板 + 傅里叶拟合）
- ``coronary_outlet``    三元 RCR Windkessel 出口压力
- ``coronary_constants`` 模块间共享的公开常量
- ``calibration_cache``  q_scale 标定缓存读写与事后绘图

常用 API 集中导出，便于 ``from SimuCoronary import NavierStokes1D``。
导出采用惰性导入（PEP 562）：访问属性时才加载对应子模块，因此
``python -m SimuCoronary.geometry`` 之类的子模块直跑不会被包提前导入干扰。
需要细分 API 时也可直接从子模块导入，例如
``from SimuCoronary.calibration_cache import load_calibration_cache``。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# 公开名 -> 所属子模块
_EXPORTS: dict[str, str] = {
    # branch
    "SideBranch": "branch",
    "diameter_from_area_cm2": "branch",
    "diameter_from_murray": "branch",
    "murray_flow_fractions": "branch",
    "prepare_side_branches": "branch",
    "scale_branches_to_murray": "branch",
    # calibration_cache
    "iteration_snapshot": "calibration_cache",
    "load_calibration_cache": "calibration_cache",
    "plot_calibration_iterations": "calibration_cache",
    "save_calibration_cache": "calibration_cache",
    "stenosis_probe_index": "calibration_cache",
    # coronary_constants
    "DEFAULT_PROXIMAL_FRACTION": "coronary_constants",
    "DYNE_PER_CM2_TO_MMHG": "coronary_constants",
    "MINIMUM_AREA_RATIO": "coronary_constants",
    "MINIMUM_INLET_FLOW_ML_S": "coronary_constants",
    "MMHG_TO_DYNE_PER_CM2": "coronary_constants",
    "P_INLET_REF_MMHG": "coronary_constants",
    "rho_for_mmhg_pressure_coupling": "coronary_constants",
    # coronary_inlet
    "coronary_inlet_flow": "coronary_inlet",
    "coronary_inlet_flow_stats": "coronary_inlet",
    "coronary_flow_template": "coronary_inlet",
    "diastolic_flow_fraction": "coronary_inlet",
    "fit_fourier_coefficients": "coronary_inlet",
    "plot_coronary_flow": "coronary_inlet",
    "template_peak_metrics": "coronary_inlet",
    # coronary_outlet
    "coronary_outlet_pressure": "coronary_outlet",
    "coronary_outlet_pressure_stats": "coronary_outlet",
    "plot_coronary_outlet_pressure": "coronary_outlet",
    "windkessel3_outlet_pressure": "coronary_outlet",
    # calculators
    "calibrate_q_scale_to_inlet_peak": "calculators",
    # geometry
    "StenosisSegment": "geometry",
    "build_stenosis_segment": "geometry",
    "build_virtual_stent": "geometry",
    "find_mla_index": "geometry",
    "resample_axial_profile": "geometry",
    "smooth_axial_area": "geometry",
    # navier_stokes_1d
    "BloodFlowParameters": "navier_stokes_1d",
    "NavierStokes1D": "navier_stokes_1d",
    "OutletWindkesselState": "navier_stokes_1d",
    "area_from_lumen_pressure_mmhg": "navier_stokes_1d",
    "integral_mean": "navier_stokes_1d",
    "lumen_pressure_mmhg": "navier_stokes_1d",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # 供 IDE / 类型检查器静态解析
    from .branch import (
        SideBranch,
        diameter_from_area_cm2,
        diameter_from_murray,
        murray_flow_fractions,
        prepare_side_branches,
        scale_branches_to_murray,
    )
    from .calibration_cache import (
        iteration_snapshot,
        load_calibration_cache,
        plot_calibration_iterations,
        save_calibration_cache,
        stenosis_probe_index,
    )
    from .coronary_constants import (
        DEFAULT_PROXIMAL_FRACTION,
        DYNE_PER_CM2_TO_MMHG,
        MINIMUM_AREA_RATIO,
        MINIMUM_INLET_FLOW_ML_S,
        MMHG_TO_DYNE_PER_CM2,
        P_INLET_REF_MMHG,
        rho_for_mmhg_pressure_coupling,
    )
    from .coronary_inlet import (
        coronary_flow_template,
        coronary_inlet_flow,
        coronary_inlet_flow_stats,
        diastolic_flow_fraction,
        fit_fourier_coefficients,
        plot_coronary_flow,
        template_peak_metrics,
    )
    from .coronary_outlet import (
        coronary_outlet_pressure,
        coronary_outlet_pressure_stats,
        plot_coronary_outlet_pressure,
        windkessel3_outlet_pressure,
    )
    from .calculators import calibrate_q_scale_to_inlet_peak
    from .geometry import (
        StenosisSegment,
        build_stenosis_segment,
        build_virtual_stent,
        find_mla_index,
        resample_axial_profile,
        smooth_axial_area,
    )
    from .navier_stokes_1d import (
        BloodFlowParameters,
        NavierStokes1D,
        OutletWindkesselState,
        area_from_lumen_pressure_mmhg,
        integral_mean,
        lumen_pressure_mmhg,
    )


def __getattr__(name: str) -> Any:
    """惰性加载 ``_EXPORTS`` 中登记的公开名字。"""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
