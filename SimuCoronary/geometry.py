from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StenosisSegment:
    """一段检测到的狭窄（仅保留主 MLA 一段）。"""

    prox_ref_idx: int  # 近端参考下标
    dist_ref_idx: int  # 远端参考下标
    mla_idx: int  # 最小管腔面积下标
    prox_area_cm2: float  # 近端参考面积
    dist_area_cm2: float  # 远端参考面积
    mean_ref_area_cm2: float  # 近远端参考面积平均值
    mla_area_cm2: float  # 最小管腔面积


_END_EXCLUDE_FRAMES = 5  # MLA搜索时，首尾各排除帧数（固定）
_MAX_REF_DISTANCE_CM = 5.0  # 参考帧距 MLA 上限 (cm)


def find_mla_index(
    areas: np.ndarray,
    end_exclude: int = _END_EXCLUDE_FRAMES,
) -> int:
    """排除首尾各 ``end_exclude`` 帧后，取面积最小帧为 MLA。"""
    a = np.asarray(areas, dtype=float).ravel()
    n = int(a.shape[0])
    if n <= 0:
        raise ValueError("areas 为空")
    if n == 1:
        return 0
    # 保证中间至少留 1 帧
    exc = int(min(max(end_exclude, 0), max((n - 1) // 2, 0)))
    if n - 2 * exc < 1:
        exc = 0
    i0 = exc
    i1 = n - exc  # slice end (exclusive)
    return int(i0 + np.argmin(a[i0:i1]))


def _vessel_length_cm(n: int, frame_spacing_cm: float) -> float:
    if n <= 1:
        return 0.0
    return float(frame_spacing_cm) * float(n - 1)


def _proximal_ref_index(
    a: np.ndarray,
    mla_idx: int,
    frame_spacing_cm: float,
    max_ref_distance_cm: float,
) -> int:
    """近端：在 MLA 近侧窗口内取面积最大帧；并列取距 MLA 最近（下标更大）。"""
    n = int(a.shape[0])
    if mla_idx <= 0:
        return 0
    L = _vessel_length_cm(n, frame_spacing_cm)
    if L < max_ref_distance_cm - 1e-15:
        lo = 0
    else:
        max_off = int(np.floor(max_ref_distance_cm / max(float(frame_spacing_cm), 1e-12)))
        lo = max(0, mla_idx - max_off)
    hi = mla_idx  # [lo, mla) ；不含 MLA
    if hi <= lo:
        return max(mla_idx - 1, 0)
    region = a[lo:hi]
    # 并列最大值取最后一个 → 距 MLA 最近
    return int(lo + (len(region) - 1 - int(np.argmax(region[::-1]))))


def _distal_ref_index(
    a: np.ndarray,
    mla_idx: int,
    frame_spacing_cm: float,
    max_ref_distance_cm: float,
) -> int:
    """远端：在 MLA 远侧窗口内取面积最大帧；并列取距 MLA 最近（下标更小）。"""
    n = int(a.shape[0])
    if mla_idx >= n - 1:
        return n - 1
    L = _vessel_length_cm(n, frame_spacing_cm)
    if L < max_ref_distance_cm - 1e-15:
        hi = n  # exclusive
    else:
        max_off = int(np.floor(max_ref_distance_cm / max(float(frame_spacing_cm), 1e-12)))
        hi = min(n, mla_idx + 1 + max_off)
    lo = mla_idx + 1  # (mla, hi)
    if hi <= lo:
        return min(mla_idx + 1, n - 1)
    region = a[lo:hi]
    # np.argmax 取第一个最大值 → 距 MLA 最近
    return int(lo + int(np.argmax(region)))


def build_stenosis_segment(
    areas: np.ndarray,
    *,
    frame_spacing_cm: float = 0.02,
    max_ref_distance_cm: float = _MAX_REF_DISTANCE_CM,
    end_exclude: int = _END_EXCLUDE_FRAMES,
) -> StenosisSegment:
    """先定位 MLA，再在近/远端 ≤10 mm（或全段）内取面积最大帧为参考。"""
    a = np.asarray(areas, dtype=float).ravel()
    n = int(a.shape[0])
    if n < 1:
        raise ValueError("areas 为空")
    dx = max(float(frame_spacing_cm), 1e-12)
    mla_idx = find_mla_index(a, end_exclude=end_exclude)
    prox_idx = _proximal_ref_index(a, mla_idx, dx, float(max_ref_distance_cm))
    dist_idx = _distal_ref_index(a, mla_idx, dx, float(max_ref_distance_cm))
    # 首尾各排除后 MLA 落在开区间，近/远侧窗口均不含 MLA ⇒ prox < mla < dist
    prox_a = float(a[prox_idx])
    dist_a = float(a[dist_idx])
    mla_a = float(a[mla_idx])
    return StenosisSegment(
        prox_ref_idx=int(prox_idx),
        dist_ref_idx=int(dist_idx),
        mla_idx=int(mla_idx),
        prox_area_cm2=prox_a,
        dist_area_cm2=dist_a,
        mean_ref_area_cm2=0.5 * (prox_a + dist_a),
        mla_area_cm2=mla_a,
    )


def smooth_axial_area(
    areas: np.ndarray,
    stenosis_segment: StenosisSegment,
    *,
    only_smooth: bool = False,
    sigma_nodes: float = 2.0,
    only_padding_max: bool = True,
) -> np.ndarray:
    """按狭窄近/远端参考分段重构管腔面积，再整段高斯平滑。

    将序列划分为三段（含端点）：
    ``[0, prox]``、``[prox, dist]``、``[dist, n-1]``。

    - 近端段：在 ``[0, prox)``（不含 prox）内取面积最大帧 ``id1``；``[0, id1]``
      常数填为 ``A[id1]``；``[id1, prox]`` 在 ``A[id1]`` 与 ``A[prox]`` 之间线性插值。
    - 中间段：保持原始面积不变。
    - 远端段：在 ``(dist, n-1]``（不含 dist）内取面积最大帧 ``id2``；``[dist, id2]``
      在 ``A[dist]`` 与 ``A[id2]`` 之间线性插值；``[id2, n-1]`` 常数填为 ``A[id2]``。

    最后对整段做 ``gaussian_filter1d(..., sigma=sigma_nodes, mode="nearest")``。
    ---
    Params:
        areas: np.ndarray, 原始血管面积序列
        stenosis_segment: StenosisSegment, 一段检测到的狭窄
        only_smooth: bool, 是否只进行高斯平滑，而不进行插值
        sigma_nodes: float, 高斯平滑的标准差
        only_padding_max: bool, 是否只把近端``[0, prox]``常数填为 ``A[prox]``，``[dist, n-1]`` 常数填为 ``A[dist]``，而不进行插值，以尽量保留原始面积曲线
    Returns:
        np.ndarray: 平滑后的面积序列
    """
    from scipy.ndimage import gaussian_filter1d

    a0 = np.asarray(areas, dtype=float).ravel()
    n = int(a0.shape[0])
    if n == 0:
        return a0.copy()

    prox = int(stenosis_segment.prox_ref_idx)
    dist = int(stenosis_segment.dist_ref_idx)
    a_prox = float(a0[prox])
    a_dist = float(a0[dist])
    if not (0 <= prox <= dist < n):
        raise ValueError(
            f"参考下标非法：prox={prox}, dist={dist}, n={n}"
        )

    out = a0.copy()

    if not only_smooth:
        # ---- 近端 [0, prox) 填补或插值 ----
        if prox > 0:
            if only_padding_max:
                out[0 : prox] = a_prox
            # ---- 近端：在 [0, prox) 搜最大
            else:
                id1 = int(np.argmax(a0[0:prox]))
                a_id1 = float(a0[id1])
                out[0 : id1 + 1] = a_id1
                x = np.arange(id1, prox + 1, dtype=float)
                out[id1 : prox + 1] = np.interp(x, [id1, prox], [a_id1, a_prox])

        # ---- 远端 (dist, n-1] 填补或插值 ----
        if dist < n - 1:
            if only_padding_max:
                out[dist: ] = a_dist
            # ---- 远端：在 (dist, n-1] 搜最大
            else:
                id2 = dist + 1 + int(np.argmax(a0[dist + 1 : n]))
                a_id2 = float(a0[id2])
                out[id2:n] = a_id2
                x = np.arange(dist, id2 + 1, dtype=float)
                out[dist : id2 + 1] = np.interp(x, [dist, id2], [a_dist, a_id2])

    return gaussian_filter1d(out, sigma=float(sigma_nodes), mode="nearest")


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


if __name__ == "__main__":
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
    seg = build_stenosis_segment(area, frame_spacing_cm=0.02)
    area_s = smooth_axial_area(area, seg, sigma_nodes=4.0)
    print(seg)
    print(f"smooth: n={len(area_s)}, min={area_s.min():.4f}, max={area_s.max():.4f}")
