"""侧支 Murray 定律：直径闭合与流量分配权重。

仅保留与主支 1D 求解器耦合所需的部分：
- D_p^γ = Σ D_d^γ 残差 / 统一尺度因子闭合
- 终端流量权重 Q ∝ D^γ（默认 γ = 7/3）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# Zhou / Kassab / Molloi：冠脉树常用 γ ≈ 7/3；经典 Murray 为 3
DEFAULT_MURRAY_EXPONENT = 7.0 / 3.0
MIN_BRANCH_DIAMETER_MM = 0.1


@dataclass
class SideBranch:
    """一条侧支（均匀管腔，0D Windkessel 终端）。"""

    frame_index: int
    diameter_mm: float
    area_mm2: float
    flow_fraction: float = 0.0  # 占入口总流量的 Murray 份额
    source: str = "manual"


def diameter_from_area_cm2(area_cm2: float) -> float:
    """截面积 (cm²) → 等效直径 (mm)。"""
    a = max(float(area_cm2), 1e-12)
    return 20.0 * np.sqrt(a / np.pi)


def diameter_from_murray(
    parent_diameter_mm: float,
    daughter_diameters_mm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> float:
    """由 D_p^γ = Σ D_d^γ 求缺失子支直径。"""
    dp = float(parent_diameter_mm) ** exponent
    known = sum(float(d) ** exponent for d in daughter_diameters_mm)
    residual = dp - known
    if residual <= 0.0:
        return 0.0
    return float(residual ** (1.0 / exponent))


def scale_branches_to_murray(
    parent_diameter_mm: float,
    distal_daughter_mm: float,
    branch_diameters_mm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> np.ndarray:
    """统一尺度因子使 D_p^γ = D_distal^γ + Σ (α D_b,i)^γ。"""
    branches = np.asarray(branch_diameters_mm, dtype=float)
    if len(branches) == 0:
        return branches

    num = float(parent_diameter_mm) ** exponent - float(distal_daughter_mm) ** exponent
    den = float(np.sum(branches**exponent))
    if num <= 0.0 or den <= 0.0:
        return np.zeros_like(branches)

    alpha = (num / den) ** (1.0 / exponent)
    return alpha * branches


def murray_flow_fractions(
    diameters_mm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> np.ndarray:
    """终端流量份额：f_i = D_i^γ / Σ D_j^γ。"""
    d = np.asarray(diameters_mm, dtype=float)
    w = np.maximum(d, 0.0) ** exponent
    s = float(np.sum(w))
    if s <= 0.0:
        n = len(d)
        return np.full(n, 1.0 / n) if n else w
    return w / s


def prepare_side_branches(
    area_ref_cm2: np.ndarray,
    prox_idx: int,
    dist_idx: int,
    branch_frame_indices: Optional[Sequence[int]] = None,
    branch_diameters_mm: Optional[Sequence[float]] = None,
    *,
    murray_exponent: float = DEFAULT_MURRAY_EXPONENT,
    apply_murray_scale: bool = True,
    min_diameter_mm: float = MIN_BRANCH_DIAMETER_MM,
) -> tuple[List[SideBranch], float]:
    """构造侧支列表，并返回主支远端终端的 Murray 流量份额。

    Parameters
    ----------
    area_ref_cm2 :
        主支参考面积 (cm²)
    prox_idx, dist_idx :
        近/远端参考下标
    branch_frame_indices, branch_diameters_mm :
        侧支开口帧与直径 (mm)；均为 None 则无侧支
    apply_murray_scale :
        True 时用近端/远端参考对侧支直径做统一 Murray 缩放

    Returns
    -------
    branches, distal_flow_fraction
    """
    if branch_frame_indices is None or branch_diameters_mm is None:
        return [], 1.0

    idxs = [int(i) for i in branch_frame_indices]
    diams = [float(d) for d in branch_diameters_mm]
    if len(idxs) != len(diams):
        raise ValueError("branch_frame_indices 与 branch_diameters_mm 长度须一致")

    n = len(area_ref_cm2)
    for i in idxs:
        if not (0 <= i < n):
            raise ValueError(f"侧支帧索引 {i} 超出 [0, {n})")

    parent = diameter_from_area_cm2(area_ref_cm2[prox_idx])
    distal = diameter_from_area_cm2(area_ref_cm2[dist_idx])

    if apply_murray_scale:
        scaled = scale_branches_to_murray(parent, distal, diams, murray_exponent)
        if float(np.sum(scaled)) <= 0.0:
            scaled = np.asarray(diams, dtype=float)
    else:
        scaled = np.asarray(diams, dtype=float)

    branches: List[SideBranch] = []
    for i, d in zip(idxs, scaled):
        d = float(d)
        if d < min_diameter_mm:
            continue
        branches.append(
            SideBranch(
                frame_index=int(i),
                diameter_mm=d,
                area_mm2=float(np.pi * (d / 2.0) ** 2),
                source="manual+murray_scaled" if apply_murray_scale else "manual",
            )
        )

    if not branches:
        return [], 1.0

    # 按轴向排序，便于沿程累减流量
    branches.sort(key=lambda b: b.frame_index)

    terminal_diams = [b.diameter_mm for b in branches] + [distal]
    fracs = murray_flow_fractions(terminal_diams, murray_exponent)
    for b, f in zip(branches, fracs[:-1]):
        b.flow_fraction = float(f)
    distal_frac = float(fracs[-1])
    return branches, distal_frac
