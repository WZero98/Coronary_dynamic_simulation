"""
simudata_experiment_different_scale.py — 面积缩放敏感性实验

与 ``simudata_experiment.py`` 使用**同一套 7 种几何**，唯一区别是把
``AREA_SCALER`` 取一系列值，观察血管整体粗细对压降 / FFR 的影响。

面积缩放的含义
--------------
``prox_area = 0.15 * AREA_SCALER``，``dist_area = 0.15 * AREA_SCALER``
（锥形例再乘 0.6）。血管长度、狭窄率、狭窄位置/占比、侧支位置与直径
**均保持不变**，因此面积缩放相当于把整条血管等比放大/缩小：

- 面积按 scaler 缩放 → 直径按 √scaler 缩放
- 侧支直径固定不变 → 缩放后侧支相对主干的比例会变化（这是有意保留的）

扫描取值：``0.3, 0.6, 0.9, 1.2, 1.5``（共 5 × 7 = 35 个算例）

运行::

    python Scripts/simudata_experiment_different_scale.py              # 4 进程
    python Scripts/simudata_experiment_different_scale.py --workers 6
    python Scripts/simudata_experiment_different_scale.py --scalers 0.3 1.5
    python Scripts/simudata_experiment_different_scale.py --dry-run

复用缓存::

    python Scripts/simudata_experiment_different_scale.py --list-cached
    python Scripts/simudata_experiment_different_scale.py --replot
    python Scripts/simudata_experiment_different_scale.py --replot scale_0.90/uniform

输出（``results/模拟数据实验2（面积大小变化）/``）::

    summary.json                       全部 35 例指标 + 扫描设置
    compare_ffr_by_scale.png           各 scaler 下 FFR(x) 对比（7 子图）
    compare_ffr_uniform.png            同一几何、不同 scaler 的 FFR 对比
    compare_ffr_min_vs_scale.png       FFR_min 随 scaler 变化（7 条曲线）
    compare_pressure_drop_vs_scale.png 压降随 scaler 变化
    compare_q_scale_vs_scale.png       标定 q_scale 随 scaler 变化
    compare_summary.png                关键指标柱状图（按 case 分组）
    scale_<s>/<case>/geometry.png      面积剖面
    scale_<s>/<case>/final_<case>.html Plotly 五图交互页
    scale_<s>/<case>/solver.pkl        solver 缓存（约 18 MB/例）

读取缓存做二次分析::

    from simudata_experiment_different_scale import load_scale_case_solver

    bundle = load_scale_case_solver(0.9, "uniform")
    solver = bundle["solver"]
    print(bundle["meta"]["area_scaler"])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence

# 本脚本位于 <root>/Scripts/，仓库根 = 上一级
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np  # noqa: E402

# 复用主脚本的几何定义、绘图与缓存逻辑，避免重复实现
import simudata_experiment as base  # noqa: E402
from SimuCoronary.utils import load_solver_bundle, plot_results  # noqa: E402

_OUT_ROOT = os.path.join(_REPO_ROOT, "results", "模拟数据实验2（面积大小变化）")

SCALERS = (0.3, 0.6, 0.9, 1.2, 1.5)
BASE_AREA_CM2 = 0.15  # AREA_SCALER = 1.0 时的参考面积


def _scaler_dir(scaler: float) -> str:
    """子目录名，固定两位小数避免 0.3 与 0.30 不一致。"""
    return f"scale_{scaler:.2f}"


# ---------------------------------------------------------------------------
# 汇总对比图
# ---------------------------------------------------------------------------
def save_comparison_figures(results: Sequence[dict], out_root: str) -> list[str]:
    """按 scaler 分组的对比图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    ok = [r for r in results if r.get("ok")]
    if not ok:
        return []

    written: list[str] = []
    scalers = sorted({r["area_scaler"] for r in ok})
    case_keys = sorted({r["key"] for r in ok})
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(scalers)))

    def _by_scaler(s: float) -> list[dict]:
        return sorted(
            [r for r in ok if r["area_scaler"] == s], key=lambda r: r["key"]
        )

    # ---- 1. FFR(x)：每个 scaler 一个子图 ----
    n = len(scalers)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False
    )
    for i, s in enumerate(scalers):
        ax = axes[i // ncol][i % ncol]
        rows = _by_scaler(s)
        c = plt.cm.tab10(np.linspace(0, 1, 10))
        for j, r in enumerate(rows):
            ax.plot(r["_x_cm"], r["_ffr"], lw=1.7, color=c[j % 10],
                    label=f'{r["key"]} ({r["ffr_min"]:.3f})')
        ax.axhline(0.80, color="k", ls=":", lw=1.1)
        ax.set_title(f"AREA_SCALER = {s:g}  "
                     f"(A_prox = {BASE_AREA_CM2 * s:.4g} cm²)", fontsize=11)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("FFR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for i in range(n, nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("FFR 沿程分布 — 各面积缩放下的 7 种几何", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_ffr_by_scale.png")
    fig.savefig(p, dpi=135)
    plt.close(fig)
    written.append(p)

    # ---- 2. 同一几何、不同 scaler（选 uniform 作代表）----
    ref = "uniform" if "uniform" in case_keys else case_keys[0]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, s in enumerate(scalers):
        rows = [r for r in ok if r["area_scaler"] == s and r["key"] == ref]
        if not rows:
            continue
        r = rows[0]
        ax.plot(r["_x_cm"], r["_ffr"], lw=2.0, color=colors[i],
                label=f'scaler={s:g}  (min={r["ffr_min"]:.4f})')
    ax.axhline(0.80, color="k", ls=":", lw=1.2, label="FFR = 0.80")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("FFR")
    ax.set_title(f"不同面积缩放下的 FFR 对比（几何 = {ref}）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_root, f"compare_ffr_{ref}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 3/4/5. 指标随 scaler 的曲线 ----
    panels = (
        ("ffr_min", "最小 FFR", "compare_ffr_min_vs_scale.png", None),
        ("p_drop_mean", "时间平均压降 p(0)−p(L) (mmHg)",
         "compare_pressure_drop_vs_scale.png", None),
        ("q_scale", "标定 q_scale", "compare_q_scale_vs_scale.png", 1.0),
    )
    for field_name, ylabel, fname, hline in panels:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        c = plt.cm.tab10(np.linspace(0, 1, 10))
        for j, key in enumerate(case_keys):
            xs, ys = [], []
            for s in scalers:
                rows = [r for r in ok
                        if r["area_scaler"] == s and r["key"] == key]
                if rows:
                    xs.append(s)
                    ys.append(rows[0][field_name])
            if xs:
                ax.plot(xs, ys, "o-", lw=1.8, ms=6, color=c[j % 10], label=key)
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=1.1)
        ax.set_xlabel("AREA_SCALER")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs 面积缩放")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_root, fname)
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    # ---- 6. 关键指标柱状图（按 case × scaler）----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    width = 0.8 / max(len(scalers), 1)
    pos = np.arange(len(case_keys))
    sub_panels = (
        ("ffr_min", "最小 FFR", axes[0, 0], True),
        ("p_in_peak_mmhg", "入口峰值压 (mmHg)", axes[0, 1], False),
        ("q_scale", "标定 q_scale", axes[1, 0], False),
        ("p_drop_mean", "时间平均压降 (mmHg)", axes[1, 1], False),
    )
    for field_name, title, ax, show_thr in sub_panels:
        for i, s in enumerate(scalers):
            rows = _by_scaler(s)
            lookup = {r["key"]: r for r in rows}
            vals = [
                lookup[k][field_name] if k in lookup else 0.0
                for k in case_keys
            ]
            ax.bar(
                pos + i * width - 0.4 + width / 2, vals, width,
                label=f"scaler={s:g}", color=colors[i],
            )
        if show_thr:
            ax.axhline(0.80, color="k", ls=":", lw=1.1)
        ax.set_xticks(pos)
        ax.set_xticklabels(case_keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, title="AREA_SCALER")
    fig.suptitle("面积缩放实验 — 关键指标对比", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_summary.png")
    fig.savefig(p, dpi=135)
    plt.close(fig)
    written.append(p)

    return written


# ---------------------------------------------------------------------------
# 读取缓存
# ---------------------------------------------------------------------------
def scale_case_dir(scaler: float, case_key: str) -> str:
    """某 (scaler, case) 的目录绝对路径。"""
    return os.path.abspath(
        os.path.join(_OUT_ROOT, _scaler_dir(scaler), case_key)
    )


def list_cached_cases(out_root: str = _OUT_ROOT) -> dict[str, str]:
    """列出所有已缓存算例，键为 ``"scale_0.90/uniform"`` 这样的相对路径。"""
    if not os.path.isdir(out_root):
        return {}
    found: dict[str, str] = {}
    for scale_dir in sorted(os.listdir(out_root)):
        sd = os.path.join(out_root, scale_dir)
        if not os.path.isdir(sd) or not scale_dir.startswith("scale_"):
            continue
        for case_dir in sorted(os.listdir(sd)):
            p = os.path.join(sd, case_dir, base.SOLVER_FILENAME)
            if os.path.isfile(p):
                found[f"{scale_dir}/{case_dir}"] = os.path.abspath(p)
    return found


def load_scale_case_solver(
    scaler: float | str,
    case_key: str,
    out_root: str = _OUT_ROOT,
) -> dict:
    """读取某个 (面积缩放, 几何) 组合的缓存。

    Parameters
    ----------
    scaler :
        面积缩放系数（如 ``0.9``），或已是目录名（如 ``"scale_0.90"``）
    case_key :
        几何名（``uniform`` / ``taper`` / ... 共 7 个）
    out_root :
        结果根目录

    Returns
    -------
    dict
        ``{"solver", "cal", "meta", "path"}``；``meta["area_scaler"]``
        即该例的缩放系数

    Examples
    --------
    ::

        bundle = load_scale_case_solver(1.2, "stenosis_center")
        solver = bundle["solver"]
        ffr = solver.ffr_ratio()
    """
    if isinstance(scaler, str):
        scale_dir = scaler if scaler.startswith("scale_") else _scaler_dir(
            float(scaler)
        )
    else:
        scale_dir = _scaler_dir(float(scaler))

    path = os.path.join(out_root, scale_dir, case_key, base.SOLVER_FILENAME)
    if not os.path.exists(path):
        avail = list_cached_cases(out_root)
        hint = (
            "可用（前 20 个）：\n  " + "\n  ".join(list(avail)[:20])
            if avail
            else f"{out_root} 下没有任何缓存，请先运行本脚本。"
        )
        raise FileNotFoundError(
            f"未找到 scaler={scaler!r}, case={case_key!r} 的缓存：{path}\n{hint}"
        )

    bundle = load_solver_bundle(path)
    bundle["path"] = path
    return bundle


def replot_from_cache(
    scaler: float | str,
    case_key: str,
    out_root: str = _OUT_ROOT,
    save_path: str | None = None,
    show: bool = False,
):
    """从缓存重新出五图交互页，不重跑仿真。"""
    bundle = load_scale_case_solver(scaler, case_key, out_root)
    solver = bundle["solver"]
    meta = bundle.get("meta", {})
    s_val = meta.get("area_scaler", scaler)
    q = bundle.get("cal", {}).get("q_scale")

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(bundle["path"]),
            f"replot_{case_key}.html",
        )

    title = f"scaler={s_val:g} {case_key}"
    if isinstance(q, float):
        title += f" (q_scale={q:.3f})"
    return plot_results(solver, title=title, show=show, save_path=save_path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _json_safe(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _run_scale_job(job: tuple) -> dict:
    """执行一个 (几何, 目录名, 额外 meta, 输出根目录) 任务。

    必须是**模块级函数**：``ProcessPoolExecutor`` 要把任务对象 pickle 后发给
    子进程，闭包/局部函数无法序列化（会报
    ``AttributeError: Can't get local object``）。
    """
    spec, dir_name, extra, out_root = job
    r = base.run_one_case(spec, out_root, dir_name, extra)
    r["area_scaler"] = extra["area_scaler"]
    if r.get("ok"):
        # 时间平均压降：入口 vs 出口（用已算好的 p_mean）
        r["p_drop_mean"] = float(r["_p_mean"][0] - r["_p_mean"][-1])
    return r


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="面积缩放敏感性实验（7 种几何 × 5 个 AREA_SCALER）"
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="并行进程数（默认 min(4, cpu_count)）",
    )
    parser.add_argument(
        "--scalers", type=float, nargs="+", default=list(SCALERS),
        help=f"要扫描的 AREA_SCALER 列表（默认 {list(SCALERS)}）",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="只跑 key 或 label 含该子串的几何",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印算例清单，不跑仿真",
    )
    parser.add_argument(
        "--out", type=str, default=_OUT_ROOT, help="输出根目录",
    )
    parser.add_argument(
        "--list-cached", action="store_true",
        help="列出已缓存的 (scaler, case) 后退出",
    )
    parser.add_argument(
        "--replot", type=str, nargs="?", const="__ALL__", default=None,
        metavar="SCALE_DIR/CASE",
        help="从缓存重新出图：形如 scale_0.90/uniform；不带参数则重画全部",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ---- 子命令 ----
    if args.list_cached:
        cached = list_cached_cases(args.out)
        if not cached:
            print(f"{args.out} 下没有 solver 缓存。")
            print("请先运行：python Scripts/"
                  "simudata_experiment_different_scale.py")
            return
        print(f"已缓存 {len(cached)} 个算例（{args.out}）：")
        for key, path in cached.items():
            mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {key:28} {mb:7.1f} MB  {path}")
        return

    if args.replot is not None:
        cached = list_cached_cases(args.out)
        if not cached:
            raise SystemExit(
                f"{args.out} 下没有 solver 缓存，无法重画。"
                "请先运行本脚本（不带 --replot）。"
            )
        targets = sorted(cached) if args.replot == "__ALL__" else [args.replot]
        print(f"从缓存重画 {len(targets)} 个算例…")
        for key in targets:
            parts = key.split("/")
            if len(parts) != 2:
                print(f"  [SKIP] {key}: 需要形如 scale_0.90/uniform")
                continue
            try:
                fig = replot_from_cache(parts[0], parts[1], args.out)
                print(f"  [OK] {key}  ({len(fig.data)} traces)")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {key}: {type(exc).__name__}: {exc}")
        return

    # ---- 构造任务清单：scalers × geometries ----
    scalers = sorted({float(s) for s in args.scalers})
    if any(s <= 0 for s in scalers):
        raise SystemExit("--scalers 必须全部为正数")

    jobs: list[tuple[base.CaseSpec, str, dict]] = []
    for s in scalers:
        specs = base.build_cases(area_scaler=s)
        if args.only:
            needle = args.only.lower()
            specs = [
                c for c in specs
                if needle in c.key.lower() or needle in c.label.lower()
            ]
        for spec in specs:
            jobs.append(
                (
                    spec,
                    f"{_scaler_dir(s)}/{spec.key}",
                    {"area_scaler": s},
                    args.out,
                )
            )

    if not jobs:
        raise SystemExit("没有匹配的算例，请检查 --only / --scalers")

    print(f"共 {len(jobs)} 个算例（{len(scalers)} 个 AREA_SCALER "
          f"× {len({j[0].key for j in jobs})} 种几何）")
    print(f"输出目录：{args.out}")
    print(f"缩放系数：{[f'{s:g}' for s in scalers]}")
    print("基准面积：0.15 cm² × scaler"
          f"  →  {[f'{BASE_AREA_CM2 * s:.4g}' for s in scalers]}")
    print("仿真设置：L={} cm, n_cells={}, duration={}s, "
          "peak∈({}, {}) mmHg, virtual_stent=False".format(
              base.LENGTH_CM, base.N_CELLS, base.DURATION_S,
              base.PEAK_LO_MMHG, base.PEAK_HI_MMHG))
    print(f"并行：{args.workers} 进程")
    print()

    if args.dry_run:
        for spec, dir_name, extra, _out in jobs:
            print(f"  {dir_name:28} scaler={extra['area_scaler']:g}  "
                  f"A_prox={spec.vessel_kwargs['prox_area_cm2']:.4g} "
                  f"A_dist={spec.vessel_kwargs['dist_area_cm2']:.4g}")
        print("\n--dry-run：未执行仿真")
        return

    os.makedirs(args.out, exist_ok=True)
    t_all = time.perf_counter()
    results: list[dict] = []

    if args.workers <= 1:
        for job in jobs:
            print(f"[start] {job[1]}", flush=True)
            r = _run_scale_job(job)
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{status}] {r.get('case_dir', job[1])}  "
                  f"{r['wall_s']:.0f}s"
                  + ("" if r["ok"] else f"  {r['error']}"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_scale_job, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {
                        "ok": False,
                        "key": job[0].key,
                        "case_dir": job[1],
                        "area_scaler": job[2]["area_scaler"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "wall_s": float("nan"),
                    }
                results.append(r)
                status = "OK" if r["ok"] else "FAIL"
                print(f"[{status}] {r.get('case_dir', job[1])}  "
                      f"{r['wall_s']:.0f}s"
                      + ("" if r["ok"] else f"  {r['error']}"), flush=True)

    elapsed = time.perf_counter() - t_all
    results.sort(key=lambda r: (r.get("area_scaler", 0), r.get("key", "")))
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    summary = {
        "n_cases": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "wall_s_total": round(elapsed, 1),
        "workers": args.workers,
        "scan": {
            "parameter": "AREA_SCALER",
            "values": scalers,
            "base_area_cm2": BASE_AREA_CM2,
            "prox_area_cm2": {
                f"{s:g}": BASE_AREA_CM2 * s for s in scalers
            },
        },
        "settings": {
            "length_cm": base.LENGTH_CM,
            "n_cells": base.N_CELLS,
            "dx_cm": base.LENGTH_CM / (base.N_CELLS - 1),
            "duration_s": base.DURATION_S,
            "record_interval_steps": base.RECORD_INTERVAL_STEPS,
            "peak_lo_mmhg": base.PEAK_LO_MMHG,
            "peak_hi_mmhg": base.PEAK_HI_MMHG,
            "max_iters": base.MAX_ITERS,
            "virtual_stent": False,
            "branch_positions": base.BRANCH_POSITIONS,
            "branch_diameters_cm": base.BRANCH_DIAMETERS,
            "note": (
                "侧支直径固定为 0.15 cm 不随 area_scaler 缩放；"
                "因此小 scaler 下侧支相对主干更粗，分流比更大。"
            ),
        },
        "cases": [_json_safe(r) for r in results],
    }

    fig_paths = save_comparison_figures(results, args.out) if ok else []
    summary["comparison_figures"] = [
        os.path.relpath(p, args.out) for p in fig_paths
    ]

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print(f"完成：{len(ok)}/{len(results)} 成功，总耗时 {elapsed:.0f}s")
    if ok:
        print(f"{'scale':>7} {'case':22} {'q_scale':>8} {'p_in':>8} "
              f"{'ffr_min':>9} {'p_drop':>8} {'iter':>5} {'wall(s)':>8}")
        for r in ok:
            print(f"{r['area_scaler']:7.2f} {r['key']:22} "
                  f"{r['q_scale']:8.4f} {r['p_in_peak_mmhg']:8.2f} "
                  f"{r['ffr_min']:9.4f} {r['p_drop_mean']:8.2f} "
                  f"{r['n_iters']:5d} {r['wall_s']:8.0f}")
    if failed:
        print()
        print("失败算例：")
        for r in failed:
            print(f"  {r.get('case_dir', r['key'])}: {r['error']}")
    print()
    print(f"summary: {summary_path}")
    for p in fig_paths:
        print(f"figure : {p}")


if __name__ == "__main__":
    main()
