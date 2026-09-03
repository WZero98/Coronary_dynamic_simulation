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
    area_mm2: float



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
    a_prox = float(a[prox_idx])

    dist_region = np.arange(dist_start, n)
    feasible = dist_region[a[dist_region] <= a_prox + 1e-9]
    if len(feasible) == 0:
        # 无满足面积约束的帧：在远端区取面积最小者，尽量贴近“近>远”
        dist_idx = int(dist_region[np.argmin(a[dist_region])])
    else:
        dist_idx = int(feasible[np.argmax(a[feasible])])

    if prox_idx >= dist_idx:
        # 极端短序列：强制拉开
        prox_idx = 0
        dist_idx = n - 1

    # 若仍出现远端面积更大，交换语义上不合理，改为远端区次优可行帧
    if a[dist_idx] > a[prox_idx] + 1e-9:
        ordered = sorted(dist_region, key=lambda i: -a[i])
        for i in ordered:
            if a[i] <= a[prox_idx] + 1e-9 and i > prox_idx:
                dist_idx = int(i)
                break

    return prox_idx, dist_idx


def detect_stenoses(
    areas_mm2: np.ndarray,
    proximal_ref_idx: int,
    distal_ref_idx: int,
) -> List[StenosisSegment]:
    """在近远端参考之间检测最多 ``max_stenoses`` 段狭窄。

    规则：
    1. 取区间内全局最小面积为 MLA；
    2. 若次小值 A 满足 (A - MLA) ≤ ``mla_relative_tol`` × MLA，
       且不落在已记录狭窄段 [i0,i1] 内，则记为下一段狭窄；
    3. 重复直至达到上限或不再满足条件。
    """
    a = np.asarray(areas_mm2, dtype=float)
    n = len(a)
    i_lo = int(min(proximal_ref_idx, distal_ref_idx))
    i_hi = int(max(proximal_ref_idx, distal_ref_idx))
    if i_hi <= i_lo:
        return []

    half = max(int(0.05 * n), 5)
    max_n = 3
    tol = 0.1

    # 工作副本：已占用区间置为 +inf，避免再次选中
    work = a.copy()
    work[: i_lo + 1] = np.inf
    work[i_hi:] = np.inf

    stenoses: List[StenosisSegment] = []
    primary_mla_area: Optional[float] = None

    for _ in range(max_n):
        if not np.isfinite(work).any():
            break
        mla_idx = int(np.argmin(work))
        mla_area = float(a[mla_idx])
        if not np.isfinite(work[mla_idx]):
            break

        if primary_mla_area is None:
            primary_mla_area = mla_area
        else:
            # (A - MLA) / MLA ≤ tol  ⟺  A ≤ MLA * (1 + tol)
            if mla_area - primary_mla_area > tol * primary_mla_area + 1e-12:
                break

        i0 = max(i_lo, mla_idx - half)
        i1 = min(i_hi, mla_idx + half)
        if i1 <= i0:
            i1 = min(i_hi, i0 + 1)

        stenoses.append(
            StenosisSegment(
                mla_idx=mla_idx,
                i0=i0,
                i1=i1,
                area_mm2=mla_area,
            )
        )
        # 屏蔽本段狭窄范围（含边界）
        work[i0 : i1 + 1] = np.inf

    stenoses.sort(key=lambda s: s.mla_idx)
    return stenoses


