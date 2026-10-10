from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StenosisSegment:
    """一段检测到的狭窄。"""

    mla_idx: int
    i0: int
    i1: int
    area_cm2: float



def select_reference_indices(
    areas: np.ndarray,
    frac: float = 0.25
) -> Tuple[int, int]:
    """选择近/远端参考帧。

    - 近端：前 ``ref_end_fraction``（默认 25%）区段内面积最大帧；
    - 远端：后 ``ref_end_fraction`` 区段内面积最大、且面积 ≤ 近端参考的帧；
    - 保证 proximal_ref_idx < distal_ref_idx，且近端面积 ≥ 远端面积。
    """
    a = np.asarray(areas, dtype=float)
    n = len(a)
    if n < 3:
        return 0, max(n - 1, 0)

    prox_end = max(int(np.ceil(frac * n)), 1)
    dist_start = min(int(np.floor((1.0 - frac) * n)), n - 1)
    if dist_start <= prox_end:
        dist_start = min(prox_end + 1, n - 1)

    prox_region = np.arange(0, prox_end)
    prox_idx = int(prox_region[np.argmax(a[prox_region])])

    dist_region = np.arange(dist_start, n)
    dist_idx = int(dist_region[np.argmax(a[dist_region])])

    # 若远端面积更大，改为远端区最大且 ≤ 近端参考的帧
    if a[dist_idx] > a[prox_idx] + 1e-9:
        found = False
        ordered = sorted(dist_region, key=lambda i: -a[i])
        for i in ordered:
            if a[i] <= a[prox_idx] + 1e-9:
                dist_idx = int(i)
                found = True
                break
        if not found:
            print("The distal reference area is larger than the proximal reference area."
            "Please check the input data.")
    return prox_idx, dist_idx


def resample_axial_profile(
    area_cm2: np.ndarray,
    *,
    target_nx: int | None = 100,
    branch_frame_indices: Optional[Sequence[int]] = None,
    branch_diameters_cm: Optional[Sequence[float]] = None,
    frame_spacing_cm: float = 0.02,
    length_cm: Optional[float] = None,
) -> Tuple[np.ndarray, float, int, List[int], List[float]]:
    """将沿程面积（及侧支开口）重采样到 ``target_nx`` 个节点。

    用于加速 1D 求解：物理长度保持不变，仅加粗网格。``target_nx is None``
    或 ``target_nx >= len(area)`` 时不做重采样。

    Parameters
    ----------
    area_cm2 :
        近端→远端的参考截面积 (cm²)，长度 = 原始 ``nx``
    target_nx :
        目标节点数；``None`` 表示不降采样。默认 100。
    branch_frame_indices, branch_diameters_cm :
        原始网格上的侧支开口下标与直径 (cm)；可均为 None
    frame_spacing_cm :
        原始帧间距 (cm)，默认 0.02 cm = 0.2 mm；用于在未显式给
        ``length_cm`` 时算血管长度 ``length = frame_spacing_cm * nx``
    length_cm :
        可选；若给定则覆盖由帧间距推得的长度

    Returns
    -------
    area_new, length_cm, nx_new, branch_idx_new, branch_d_new
        重采样后的面积、物理长度、节点数、侧支下标与直径。
        落到同一新下标的多条侧支合并为一条（保留较大直径）。
    """
    area = np.asarray(area_cm2, dtype=float).ravel()
    nx_old = int(area.shape[0])
    if nx_old < 2:
        raise ValueError(f"area 长度至少为 2，当前 {nx_old}")

    L = float(length_cm) if length_cm is not None else float(frame_spacing_cm) * nx_old

    idxs_old = (
        [int(i) for i in branch_frame_indices]
        if branch_frame_indices is not None
        else []
    )
    diams_old = (
        [float(d) for d in branch_diameters_cm]
        if branch_diameters_cm is not None
        else []
    )
    if bool(idxs_old) != bool(diams_old):
        raise ValueError("branch_frame_indices 与 branch_diameters_cm 须同时提供或同时为 None")
    if len(idxs_old) != len(diams_old):
        raise ValueError(
            f"侧支下标与直径长度不一致 ({len(idxs_old)} vs {len(diams_old)})"
        )
    for i in idxs_old:
        if not (0 <= i < nx_old):
            raise ValueError(f"侧支帧 {i} 超出原始网格 [0, {nx_old})")

    # 无需降采样
    if target_nx is None or int(target_nx) >= nx_old:
        return (
            area.copy(),
            L,
            nx_old,
            list(idxs_old),
            list(diams_old),
        )

    nx_new = int(target_nx)
    if nx_new < 3:
        raise ValueError(f"target_nx 至少为 3，当前 {nx_new}")

    x_old = np.linspace(0.0, 1.0, nx_old)
    x_new = np.linspace(0.0, 1.0, nx_new)
    area_new = np.interp(x_new, x_old, area)

    # 分支：按归一化轴向位置映射到新网格，同格合并取较大直径
    merged: dict[int, float] = {}
    for i_old, d in zip(idxs_old, diams_old):
        i_new = int(np.clip(np.rint(i_old * (nx_new - 1) / (nx_old - 1)), 0, nx_new - 1))
        if i_new in merged:
            merged[i_new] = max(merged[i_new], float(d))
        else:
            merged[i_new] = float(d)
    idxs_new = sorted(merged.keys())
    diams_new = [merged[i] for i in idxs_new]

    print(
        f"resample_axial_profile: nx {nx_old} → {nx_new}, "
        f"L={L:.3f} cm, dx={L / (nx_new - 1):.4f} cm, "
        f"branches {len(idxs_old)} → {len(idxs_new)}"
    )
    return area_new, L, nx_new, idxs_new, diams_new


def detect_stenoses(
    areas_cm2: np.ndarray,
    proximal_ref_idx: int,
    distal_ref_idx: int,
) -> List[StenosisSegment]:
    """在近远端参考之间检测最多 3 段狭窄。

    规则：
    1. 仅在开区间 (prox, dist) 内取全局最小面积为 MLA；
    2. 该 MLA 必须严格小于两端参考面积，否则视为无狭窄（锥形/均匀管腔）；
    3. 若次小值 A 满足 (A - MLA) ≤ 0.1 × MLA，且不落在已记录段 [i0, i1] 内，
       则记为下一段狭窄；
    4. 重复直至达到上限或不再满足条件。
    """
    a = np.asarray(areas_cm2, dtype=float)
    n = len(a)
    i_lo = int(proximal_ref_idx)
    i_hi = int(distal_ref_idx)
    if not (0 <= i_lo < n and 0 <= i_hi < n):
        raise ValueError(
            f"参考下标越界：prox={i_lo}, dist={i_hi}, n={n}"
        )
    if i_hi <= i_lo:
        raise ValueError(
            "The distal reference index is smaller than the proximal reference index."
        )

    half = max(int(0.05 * n), 5)
    max_n = 3
    tol = 0.1
    a_ref = min(float(a[i_lo]), float(a[i_hi]))

    # 只在 (i_lo, i_hi) 内搜索；已占用段置 +inf
    work = a.copy()
    work[: i_lo + 1] = np.inf
    work[i_hi:] = np.inf

    stenoses: List[StenosisSegment] = []
    primary_mla_area: Optional[float] = None

    for _ in range(max_n):
        if not np.isfinite(work).any():
            break
        mla_idx = int(np.argmin(work))
        mla_area = float(work[mla_idx])
        if not np.isfinite(mla_area):
            break

        if primary_mla_area is None:
            if mla_area >= a_ref - 1e-12:
                break
            primary_mla_area = mla_area
        elif mla_area - primary_mla_area > tol * primary_mla_area + 1e-12:
            break

        i0 = max(i_lo, mla_idx - half)
        i1 = min(i_hi, mla_idx + half)
        if i1 <= i0:
            break

        stenoses.append(
            StenosisSegment(
                mla_idx=mla_idx,
                i0=i0,
                i1=i1,
                area_cm2=mla_area,
            )
        )
        work[i0 : i1 + 1] = np.inf

    stenoses.sort(key=lambda s: s.mla_idx)
    return stenoses


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter1d
    area = np.array([
        0.08, 0.078, 0.079, 0.081, 0.082, 0.076, 0.077, 0.075, 0.074, 0.073,
        0.072, 0.071, 0.070, 0.069, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063,
        0.062, 0.065, 0.070, 0.069, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063,
        0.061, 0.062, 0.070, 0.069, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063,
        0.042, 0.041, 0.040, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033,
        0.032, 0.031, 0.030, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023,
        0.041, 0.042, 0.041, 0.041, 0.0412, 0.0422, 0.043, 0.0431, 0.0431, 0.0433,
        0.0434, 0.0435, 0.0436, 0.0437, 0.0438, 0.0439, 0.0440, 0.0441, 0.0442, 0.0443,
        0.0444, 0.0445, 0.0446, 0.0447, 0.0448, 0.0449, 0.0450, 0.0451, 0.0452, 0.0453,
        0.0454, 0.0455, 0.0456, 0.0457, 0.0458, 0.0459, 0.0460, 0.0461, 0.0462, 0.0463,
    ])
    smoothed_area = gaussian_filter1d(area, sigma=4)
    prox_idx, dist_idx = select_reference_indices(smoothed_area)
    stenoses = detect_stenoses(smoothed_area, prox_idx, dist_idx)