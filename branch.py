"""侧支 Murray 定律：直径闭合与流量分配权重。

长度/面积一律用 CGS：直径 cm，面积 cm²。
- D_p^γ = Σ D_d^γ 残差 / 统一尺度因子闭合
- 终端流量权重 Q ∝ D^γ（默认 γ = 7/3）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# Zhou / Kassab / Molloi：冠脉树常用 γ ≈ 7/3；经典 Murray 为 3
DEFAULT_MURRAY_EXPONENT = 7.0 / 3.0
MIN_BRANCH_DIAMETER_CM = 0.01  # 0.1 mm
# 冠脉侧支直径若仍以 mm 误当作 cm 传入，量级通常 > 1 cm
_SUSPECT_MM_AS_CM_THRESHOLD = 1.0


@dataclass
class SideBranch:
    """一条侧支（均匀管腔，0D Windkessel 终端）。"""

    frame_index: int
    diameter_cm: float
    area_cm2: float
    flow_fraction: float = 0.0  # 占入口总流量的 Murray 份额
    source: str = "manual"


def diameter_from_area_cm2(area_cm2: float) -> float:
    """截面积 (cm²) → 等效直径 (cm)。"""
    a = max(float(area_cm2), 1e-12)
    return 2.0 * np.sqrt(a / np.pi)


def diameter_from_murray(
    parent_diameter_cm: float,
    daughter_diameters_cm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> float:
    """由 D_p^γ = Σ D_d^γ 求缺失子支直径 (cm)。"""
    dp = float(parent_diameter_cm) ** exponent
    known = sum(float(d) ** exponent for d in daughter_diameters_cm)
    residual = dp - known
    if residual <= 0.0:
        return 0.0
    return float(residual ** (1.0 / exponent))


def scale_branches_to_murray(
    parent_diameter_cm: float,
    distal_daughter_cm: float,
    branch_diameters_cm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> np.ndarray:
    """统一尺度因子使 D_p^γ = D_distal^γ + Σ (α D_b,i)^γ（直径均为 cm）。

    注意：缩放后 Σ (α D_b)^γ ≡ D_p^γ − D_distal^γ，与输入直径绝对值无关；
    仅各侧支之间的相对比例保留。因此总侧支流量份额也不随输入直径变化。
    若要用实测直径直接决定分流，请设 apply_murray_scale=False。
    """
    branches = np.asarray(branch_diameters_cm, dtype=float)
    if len(branches) == 0:
        return branches

    num = float(parent_diameter_cm) ** exponent - float(distal_daughter_cm) ** exponent
    den = float(np.sum(branches**exponent))
    if num <= 0.0 or den <= 0.0:
        return np.zeros_like(branches)

    alpha = (num / den) ** (1.0 / exponent)
    return alpha * branches


def murray_flow_fractions(
    diameters_cm: Sequence[float],
    exponent: float = DEFAULT_MURRAY_EXPONENT,
) -> np.ndarray:
    """终端流量份额：f_i = D_i^γ / Σ D_j^γ（D 为 cm）。"""
    d = np.asarray(diameters_cm, dtype=float)
    w = np.maximum(d, 0.0) ** exponent
    s = float(np.sum(w))
    if s <= 0.0:
        n = len(d)
        return np.full(n, 1.0 / n) if n else w
    return w / s


def validate_branch_diameters_cm(
    diameters_cm: Sequence[float],
    parent_diameter_cm: float,
    *,
    auto_mm_to_cm: bool = True,
) -> np.ndarray:
    """校验侧支直径单位（须为 cm），并警告大于主支的侧支。

    - 若中位数 > 1 cm 而主支直径 < 1 cm，判定为误把 mm 当作 cm；
      ``auto_mm_to_cm=True`` 时自动 /10 并告警。
    - 任一（校正后）侧支直径 > 主支直径时发出告警（Murray 闭合前）。
    """
    d = np.asarray(diameters_cm, dtype=float).copy()
    if d.size == 0:
        return d

    parent = float(parent_diameter_cm)
    med = float(np.median(d))
    if (
        auto_mm_to_cm
        and med > _SUSPECT_MM_AS_CM_THRESHOLD
        and parent < _SUSPECT_MM_AS_CM_THRESHOLD
    ):
        warnings.warn(
            f"侧支直径中位数 {med:.3f} 更像 mm 而非 cm（主支 D={parent:.3f} cm）；"
            "已自动按 mm→cm 除以 10。请确认输入单位为 cm。",
            stacklevel=3,
        )
        d = d / 10.0

    oversized = [
        (k, float(dk))
        for k, dk in enumerate(d)
        if float(dk) > parent + 1e-12
    ]
    if oversized:
        detail = ", ".join(f"#{k}={dk:.4f} cm" for k, dk in oversized[:5])
        more = "" if len(oversized) <= 5 else f" 等 {len(oversized)} 条"
        warnings.warn(
            f"有 {len(oversized)} 条侧支直径大于主支 D_prox={parent:.4f} cm "
            f"({detail}{more})。将在 apply_murray_scale=True 时被 Murray 闭合缩放；"
            "若关闭缩放，请检查分割结果或直径单位（须为 cm）。",
            stacklevel=3,
        )
    return d


def prepare_side_branches(
    area_ref_cm2: np.ndarray,
    prox_idx: int,
    dist_idx: int,
    branch_frame_indices: Optional[Sequence[int]] = None,
    branch_diameters_cm: Optional[Sequence[float]] = None,
    *,
    murray_exponent: float = DEFAULT_MURRAY_EXPONENT,
    apply_murray_scale: bool = True,
    min_diameter_cm: float = MIN_BRANCH_DIAMETER_CM,
    auto_mm_to_cm: bool = True,
) -> tuple[List[SideBranch], float]:
    """构造侧支列表，并返回主支远端终端的 Murray 流量份额。

    Parameters
    ----------
    area_ref_cm2 :
        主支参考面积 (cm²)
    prox_idx, dist_idx :
        近/远端参考下标
    branch_frame_indices, branch_diameters_cm :
        侧支开口帧与直径 (cm)；均为 None 则无侧支
    apply_murray_scale :
        True（默认）：统一 Murray 缩放以闭合近/远端；总侧支份额由 A₀ 近/远端决定，
        输入直径绝对值几乎不影响总分流（仅影响多侧支之间的相对分配）。
        False：直接用输入直径算 f_i ∝ D_i^γ（过大侧支易导致过分流）。
    auto_mm_to_cm :
        若直径量级像 mm，自动换算为 cm。

    Returns
    -------
    branches, distal_flow_fraction
    """
    if branch_frame_indices is None or branch_diameters_cm is None:
        return [], 1.0

    idxs = [int(i) for i in branch_frame_indices]
    diams_raw = [float(d) for d in branch_diameters_cm]
    if len(idxs) != len(diams_raw):
        raise ValueError(
            f"branch_frame_indices 与 branch_diameters_cm 长度须一致 "
            f"({len(idxs)} vs {len(diams_raw)})"
        )

    n = len(area_ref_cm2)
    for i in idxs:
        if not (0 <= i < n):
            raise ValueError(f"侧支帧索引 {i} 超出 [0, {n})")

    parent = diameter_from_area_cm2(area_ref_cm2[prox_idx])
    distal = diameter_from_area_cm2(area_ref_cm2[dist_idx])
    diams = validate_branch_diameters_cm(
        diams_raw, parent, auto_mm_to_cm=auto_mm_to_cm
    )

    if apply_murray_scale:
        scaled = scale_branches_to_murray(parent, distal, diams, murray_exponent)
        if float(np.sum(scaled)) <= 0.0:
            warnings.warn(
                "Murray 残差 ≤ 0（近端不大于远端+侧支），无法闭合缩放；"
                "回退为校验后的原始直径。可检查近/远端参考面积或侧支直径。",
                stacklevel=2,
            )
            scaled = np.asarray(diams, dtype=float)
        else:
            ratios = scaled[diams > 0] / diams[diams > 0]
            alpha = float(np.mean(ratios)) if ratios.size else 1.0
            print(
                f"apply_murray_scale=True: branch diameters scaled by alpha={alpha:.4f} "
                f"for Murray closure; total side-branch share from prox/dist A0 "
                f"(D_prox={parent:.4f}, D_dist={distal:.4f} cm)."
            )
    else:
        scaled = np.asarray(diams, dtype=float)

    branches: List[SideBranch] = []
    for i, d in zip(idxs, scaled):
        d = float(d)
        if d < min_diameter_cm:
            continue
        branches.append(
            SideBranch(
                frame_index=int(i),
                diameter_cm=d,
                area_cm2=float(np.pi * (d / 2.0) ** 2),
                source="manual+murray_scaled" if apply_murray_scale else "manual",
            )
        )

    if not branches:
        return [], 1.0

    branches.sort(key=lambda b: b.frame_index)

    terminal_diams = [b.diameter_cm for b in branches] + [distal]
    fracs = murray_flow_fractions(terminal_diams, murray_exponent)
    for b, f in zip(branches, fracs[:-1]):
        b.flow_fraction = float(f)
    distal_frac = float(fracs[-1])
    return branches, distal_frac
