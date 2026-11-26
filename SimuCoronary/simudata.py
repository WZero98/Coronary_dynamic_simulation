"""
simudata.py — 合成血管几何生成器

用一个「统计参数化」的血管类批量生成模拟数据，供 FFR 虚拟对照、
回归测试与参数敏感性分析使用。生成结果与真实影像处理的产物同构：
一维沿程面积序列（cm²）＋侧支开口下标与直径（cm）。

五段式几何
----------
::

    [0, prox]              常数 = 近端参考面积
    [prox, sten_start]     线性插值（prox → MLA）
    [sten_start, sten_end] 常数 = MLA（最小管腔面积）
    [sten_end, dist]       线性插值（MLA → dist）
    [dist, L]              常数 = 远端参考面积

狭窄率过高或参数缺失时退化为三段（无狭窄）::

    [0, prox]   常数 = 近端参考面积
    [prox, dist] 线性插值（prox → dist）
    [dist, L]   常数 = 远端参考面积

采样约定
--------
``n_cells`` 按**节点**数解释（用户确认）：返回长度为 ``n_cells`` 的数组，
节点坐标 ``x_i = i·L/(n−1)``（``i = 0..n_cells−1``），首尾正好覆盖
``[0, L]``。这样可直接传给 :class:`SimuCoronary.NavierStokes1D`
（其构造要求 ``nx ≥ 3``）。

用法
----
::

    from SimuCoronary.simudata import SimulatedVessel

    v = SimulatedVessel(
        vessel_length_cm=5.8,
        n_cells=200,
        prox_area_cm2=0.082,
        dist_area_cm2=0.046,
        prox_position=0.05,
        dist_position=0.95,
        stenosis_ratio=0.6,
        stenosis_position=0.45,   # 狭窄段中心
        stenosis_fraction=0.2,    # 占血管长度 20%
        branch_positions=[0.3, 0.6],
        branch_diameters=[0.09, 0.08],
    )
    area = v.generate_area()          # shape (200,), cm²
    idx, dia = v.generate_branch()    # 分支下标与直径 (cm)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["SimulatedVessel", "VesselGeometrySummary"]


# ---------------------------------------------------------------------------
# 校验工具
# ---------------------------------------------------------------------------
def _check_scalar(name: str, value: float, *, low: float, high: float,
                  low_open: bool = False, high_open: bool = False) -> float:
    """校验标量落在区间内，返回 ``float``；错误信息带参数名与实际值。"""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，当前为 {value!r}")
    lo_ok = v > low if low_open else v >= low
    hi_ok = v < high if high_open else v <= high
    if not (lo_ok and hi_ok):
        lbr = "(" if low_open else "["
        rbr = ")" if high_open else "]"
        raise ValueError(f"{name} 必须落在 {lbr}{low}, {high}{rbr} 内，当前为 {v}")
    return v


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} 必须为正数，当前为 {value!r}")
    return v


def _check_series(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限数值")
    return arr


# ---------------------------------------------------------------------------
# 几何摘要
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VesselGeometrySummary:
    """一段合成血管的关键位置（cm）与面积（cm²），便于日志与断言。"""

    length_cm: float
    n_cells: int
    prox_area_cm2: float
    dist_area_cm2: float
    mla_area_cm2: float
    x_prox_cm: float
    x_dist_cm: float
    x_sten_start_cm: Optional[float] = None
    x_sten_end_cm: Optional[float] = None

    @property
    def has_stenosis(self) -> bool:
        return self.x_sten_start_cm is not None

    @property
    def stenosis_length_cm(self) -> float:
        if not self.has_stenosis:
            return 0.0
        return float(self.x_sten_end_cm - self.x_sten_start_cm)

    def as_dict(self) -> dict:
        return {
            "length_cm": self.length_cm,
            "n_cells": self.n_cells,
            "prox_area_cm2": self.prox_area_cm2,
            "dist_area_cm2": self.dist_area_cm2,
            "mla_area_cm2": self.mla_area_cm2,
            "x_prox_cm": self.x_prox_cm,
            "x_dist_cm": self.x_dist_cm,
            "x_sten_start_cm": self.x_sten_start_cm,
            "x_sten_end_cm": self.x_sten_end_cm,
            "has_stenosis": self.has_stenosis,
            "stenosis_length_cm": self.stenosis_length_cm,
        }


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class SimulatedVessel:
    """参数化合成血管：五段式（含狭窄）或三段式（无狭窄）面积剖面。

    Parameters
    ----------
    vessel_length_cm :
        整段血管物理长度 (cm)，必须 > 0
    n_cells :
        离散节点数，必须 ≥ 2。返回的面积数组长度即为此值，
        节点坐标 ``x_i = i·L/(n−1)``
    prox_area_cm2 :
        近端参考面积 (cm²)，必须 > 0
    dist_area_cm2 :
        远端参考面积 (cm²)，必须 > 0
    prox_position :
        近端参考位置占血管长度的比例，``[0, 1]``
    dist_position :
        远端参考位置占血管长度的比例，``[0, 1]``，必须 > ``prox_position``
    stenosis_ratio :
        狭窄率，``[0, 1)``。``MLA = 0.5·(prox+dist)·(1 − ratio)``。
        与 ``stenosis_position`` / ``stenosis_fraction`` 任一为 ``None``
        或为 0 时视为**无狭窄**
    stenosis_position :
        狭窄段**中心**占血管长度的比例，``[0, 1]``
    stenosis_fraction :
        狭窄段长度占血管总长的比例，``[0, 1]``
    branch_positions :
        各分支位置占血管长度的比例，一维序列，``[0, 1]``
    branch_diameters :
        各分支直径 (cm)，长度须与 ``branch_positions`` 一致

    Notes
    -----
    狭窄段起止位置由中心与占比推出::

        sten_start = stenosis_position − stenosis_fraction / 2
        sten_end   = stenosis_position + stenosis_fraction / 2

    并强制校验 ``prox < sten_start < sten_end < dist``（同时也要求二者落在
    ``[0, 1]`` 内）。无狭窄时只校验 ``prox < dist``。
    """

    def __init__(
        self,
        vessel_length_cm: float,
        n_cells: int,
        prox_area_cm2: float,
        dist_area_cm2: float,
        *,
        prox_position: float,
        dist_position: float,
        stenosis_ratio: Optional[float] = None,
        stenosis_position: Optional[float] = None,
        stenosis_fraction: Optional[float] = None,
        branch_positions: Optional[Sequence[float]] = None,
        branch_diameters: Optional[Sequence[float]] = None,
    ) -> None:
        # ---- 1. 必填数值参数 ----
        self.vessel_length_cm = _check_positive("vessel_length_cm", vessel_length_cm)
        n = int(n_cells)
        if n != n_cells:
            raise ValueError(f"n_cells 必须为整数，当前为 {n_cells!r}")
        if n < 2:
            raise ValueError(
                f"n_cells 必须 ≥ 2（节点采样需首尾两个端点），当前为 {n}"
            )
        self.n_cells = n
        self.prox_area_cm2 = _check_positive("prox_area_cm2", prox_area_cm2)
        self.dist_area_cm2 = _check_positive("dist_area_cm2", dist_area_cm2)

        # ---- 2. 参考位置（必填）----
        if prox_position is None:
            raise ValueError("prox_position 不能为空，必须指定近端参考位置")
        if dist_position is None:
            raise ValueError("dist_position 不能为空，必须指定远端参考位置")
        self.prox_position = _check_scalar(
            "prox_position", prox_position, low=0.0, high=1.0
        )
        self.dist_position = _check_scalar(
            "dist_position", dist_position, low=0.0, high=1.0
        )
        if self.prox_position >= self.dist_position:
            raise ValueError(
                f"prox_position ({self.prox_position:.6f}) 必须小于 "
                f"dist_position ({self.dist_position:.6f})"
            )

        # ---- 3. 狭窄参数：任一为 None 视为无狭窄 ----
        raw = (stenosis_ratio, stenosis_position, stenosis_fraction)
        n_none = sum(v is None for v in raw)
        if n_none == len(raw):
            has_stenosis = False
        elif n_none > 0:
            has_stenosis = False
            warnings.warn(
                "stenosis_ratio / stenosis_position / stenosis_fraction 中有参数为 "
                "None，按约定视为**无狭窄**（已忽略其余已给值）。"
                "如需狭窄，请三者同时提供。",
                stacklevel=2,
            )
        else:
            # 三者齐全，再按「比率为 0」判断
            ratio = _check_scalar(
                "stenosis_ratio", stenosis_ratio, low=0.0, high=1.0,
                high_open=True,
            )
            frac = _check_scalar(
                "stenosis_fraction", stenosis_fraction, low=0.0, high=1.0
            )
            pos = _check_scalar(
                "stenosis_position", stenosis_position, low=0.0, high=1.0
            )
            has_stenosis = ratio > 0.0 and frac > 0.0

        self.has_stenosis = bool(has_stenosis)

        if self.has_stenosis:
            self.stenosis_ratio = ratio
            self.stenosis_position = pos
            self.stenosis_fraction = frac
            self.mean_ref_area_cm2 = 0.5 * (
                self.prox_area_cm2 + self.dist_area_cm2
            )
            self.mla_area_cm2 = self.mean_ref_area_cm2 * (1.0 - ratio)

            start = pos - 0.5 * frac
            end = pos + 0.5 * frac

            if start < 0.0:
                raise ValueError(
                    f"狭窄起始位置推算为 {start:.6f} < 0："
                    f"stenosis_position={pos:.6f}, stenosis_fraction={frac:.6f} "
                    "使狭窄段越过血管起始端"
                )
            if end > 1.0:
                raise ValueError(
                    f"狭窄结束位置推算为 {end:.6f} > 1："
                    f"stenosis_position={pos:.6f}, stenosis_fraction={frac:.6f} "
                    "使狭窄段越过血管末端"
                )
            if start <= self.prox_position:
                raise ValueError(
                    f"狭窄起始位置 ({start:.6f}) 必须大于近端参考位置 "
                    f"({self.prox_position:.6f})"
                )
            if end <= start:
                raise ValueError(
                    f"狭窄结束位置 ({end:.6f}) 必须大于狭窄起始位置 ({start:.6f})"
                )
            if self.dist_position <= end:
                raise ValueError(
                    f"远端参考位置 ({self.dist_position:.6f}) 必须大于狭窄结束位置 "
                    f"({end:.6f})"
                )
            self.sten_start_position = float(start)
            self.sten_end_position = float(end)

            # 参考面积不等（锥形血管）时，按公式算出的 MLA 可能不低于较细一端，
            # 此时「狭窄段」反而比参考段宽，全段最小面积落在参考段而非狭窄段。
            if self.mla_area_cm2 >= min(self.prox_area_cm2, self.dist_area_cm2):
                thinner_end = "远端" if self.dist_area_cm2 < self.prox_area_cm2 else "近端"
                need = 1.0 - min(self.prox_area_cm2, self.dist_area_cm2) / self.mean_ref_area_cm2
                warnings.warn(
                    f"stenosis_ratio={ratio:g} 偏小：按公式 MLA={self.mla_area_cm2:.6g} cm² "
                    f"不低于{thinner_end}参考面积 "
                    f"{min(self.prox_area_cm2, self.dist_area_cm2):.6g} cm²，"
                    f"生成的狭窄段实际比参考段更宽，全段最小面积不在狭窄段。"
                    f"若希望狭窄段确实最细，stenosis_ratio 需 > {need:.4f}。",
                    stacklevel=2,
                )
        else:
            # 无狭窄时这些字段置 None，避免误用
            self.stenosis_ratio = None
            self.stenosis_position = None
            self.stenosis_fraction = None
            self.mean_ref_area_cm2 = 0.5 * (
                self.prox_area_cm2 + self.dist_area_cm2
            )
            self.mla_area_cm2 = self.mean_ref_area_cm2
            self.sten_start_position = None
            self.sten_end_position = None

        # ---- 4. 分支参数 ----
        if (branch_positions is None) != (branch_diameters is None):
            raise ValueError(
                "branch_positions 与 branch_diameters 须同时提供或同时为 None"
            )
        if branch_positions is None:
            self.branch_positions = np.zeros(0, dtype=float)
            self.branch_diameters = np.zeros(0, dtype=float)
        else:
            bpos = _check_series("branch_positions", branch_positions)
            bdia = _check_series("branch_diameters", branch_diameters)
            if bpos.size != bdia.size:
                raise ValueError(
                    f"branch_positions 与 branch_diameters 长度须一致 "
                    f"({bpos.size} vs {bdia.size})"
                )
            if bpos.size and (np.any(bpos < 0.0) or np.any(bpos > 1.0)):
                bad = bpos[(bpos < 0.0) | (bpos > 1.0)]
                raise ValueError(
                    f"branch_positions 必须落在 [0, 1] 内，越界值：{bad.tolist()}"
                )
            if bdia.size and np.any(bdia <= 0.0):
                bad = bdia[bdia <= 0.0]
                raise ValueError(
                    f"branch_diameters 必须全部 > 0，非法值：{bad.tolist()}"
                )
            self.branch_positions = bpos
            self.branch_diameters = bdia

    # ------------------------------------------------------------------
    # 坐标与几何
    # ------------------------------------------------------------------
    @property
    def x_cm(self) -> np.ndarray:
        """节点轴向坐标 (cm)，``x_i = i·L/(n−1)``，长度 ``n_cells``。"""
        n = self.n_cells
        if n == 1:
            return np.zeros(1, dtype=float)
        return np.linspace(0.0, self.vessel_length_cm, n)

    @property
    def dx_cm(self) -> float:
        """节点间距 (cm)。"""
        if self.n_cells < 2:
            return 0.0
        return self.vessel_length_cm / (self.n_cells - 1)

    def _segment_breakpoints(self) -> list[tuple[float, float]]:
        """面积剖面的折点 ``(x_cm, area_cm2)``，按 x 递增排列。"""
        L = self.vessel_length_cm
        ap = self.prox_area_cm2
        ad = self.dist_area_cm2
        xp = self.prox_position * L
        xd = self.dist_position * L

        if not self.has_stenosis:
            # 三段：常数 / 线性 / 常数
            return [(0.0, ap), (xp, ap), (xd, ad), (L, ad)]

        am = self.mla_area_cm2
        xs = float(self.sten_start_position) * L
        xe = float(self.sten_end_position) * L
        # 五段：常数 / 线性 / 常数(MLA) / 线性 / 常数
        return [
            (0.0, ap),
            (xp, ap),
            (xs, am),
            (xe, am),
            (xd, ad),
            (L, ad),
        ]

    @staticmethod
    def _interp_plateau(
        x: np.ndarray, breakpoints: list[tuple[float, float]]
    ) -> np.ndarray:
        """在折点上做分段线性插值；零长度段自动合并（保留后者值）。"""
        bx: list[float] = []
        by: list[float] = []
        for xk, yk in breakpoints:
            if bx and xk <= bx[-1]:
                # 与前一个折点重合（段长为 0）：保留后者的面积
                by[-1] = yk
                bx[-1] = max(bx[-1], xk)
            else:
                bx.append(xk)
                by.append(yk)
        if len(bx) == 1:
            return np.full_like(x, by[0], dtype=float)
        return np.interp(x, np.asarray(bx), np.asarray(by))

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def generate_area(self) -> np.ndarray:
        """生成沿程参考面积 ``A₀(x)``。

        Returns
        -------
        np.ndarray
            shape ``(n_cells,)``，单位 cm²，节点坐标见 :attr:`x_cm`
        """
        area = self._interp_plateau(self.x_cm, self._segment_breakpoints())
        return np.ascontiguousarray(area, dtype=float)

    def generate_branch(self) -> tuple[np.ndarray, np.ndarray]:
        """生成分支开口的下标与直径。

        Returns
        -------
        indices : np.ndarray
            分支所在节点下标（``int``，按升序），shape ``(n_branch,)``
        diameters : np.ndarray
            对应分支直径 (cm)，shape ``(n_branch,)``

        Notes
        -----
        位置比例按 ``idx = round(pos · (n−1))`` 映射到最近节点并钳到
        ``[0, n−1]``。若两个分支落到同一节点，此处**不合并**，保持与输入
        一一对应；需要合并时由下游（如 ``NavierStokes1D``）处理。
        """
        if self.branch_positions.size == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        n = self.n_cells
        raw = self.branch_positions * (n - 1)
        idx = np.clip(np.rint(raw).astype(int), 0, n - 1)
        order = np.argsort(idx, kind="stable")
        return idx[order].astype(int), self.branch_diameters[order].astype(float)

    # ------------------------------------------------------------------
    # 摘要 / 描述
    # ------------------------------------------------------------------
    def summary(self) -> VesselGeometrySummary:
        """返回关键位置与面积的只读摘要。"""
        L = self.vessel_length_cm
        return VesselGeometrySummary(
            length_cm=L,
            n_cells=self.n_cells,
            prox_area_cm2=self.prox_area_cm2,
            dist_area_cm2=self.dist_area_cm2,
            mla_area_cm2=self.mla_area_cm2,
            x_prox_cm=self.prox_position * L,
            x_dist_cm=self.dist_position * L,
            x_sten_start_cm=(
                None if not self.has_stenosis
                else float(self.sten_start_position) * L
            ),
            x_sten_end_cm=(
                None if not self.has_stenosis
                else float(self.sten_end_position) * L
            ),
        )

    def describe(self) -> dict:
        """扁平字典，便于日志或写入 JSON。"""
        d = self.summary().as_dict()
        d["n_branch"] = int(self.branch_positions.size)
        d["branch_indices"], d["branch_diameters_cm"] = (
            (arr.tolist() for arr in self.generate_branch())
        )
        return d

    def __repr__(self) -> str:
        s = self.summary()
        core = (
            f"SimulatedVessel(L={s.length_cm:g} cm, n={s.n_cells}, "
            f"A_prox={s.prox_area_cm2:g}, A_dist={s.dist_area_cm2:g}"
        )
        if self.has_stenosis:
            core += (
                f", MLA={self.mla_area_cm2:.6g} "
                f"(ratio={self.stenosis_ratio:g}), "
                f"sten x∈[{s.x_sten_start_cm:.4g}, {s.x_sten_end_cm:.4g}] cm"
            )
        else:
            core += ", no stenosis"
        core += f", n_branch={self.branch_positions.size})"
        return core
