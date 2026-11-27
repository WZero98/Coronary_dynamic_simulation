"""
simudata_experiment_different_stenosis_rate.py — 狭窄率敏感性实验

与 ``simudata_experiment.py`` 共享几何定义，只取**含狭窄的 4 个算例**
（``stenosis_center`` / ``taper_stenosis`` / ``stenosis_proximal`` /
``stenosis_distal``），把 ``STENOSIS_RATE`` 取一系列值，观察狭窄严重程度
对压降与 FFR 的影响。

扫描取值：``0.9, 0.8, 0.7, 0.6, 0.5``（由重到轻，共 5 × 4 = 20 个算例）

其余参数与主实验一致：``L=10 cm``、``n_cells=501``、``duration_s=3.0``、
面积缩放 ``AREA_SCALER=1.0``、3 条侧支。

狭窄率语义
----------
``MLA = 0.5·(prox+dist)·(1 − STENOSIS_RATE)``。取值越接近 1，狭窄越重；
``0.9`` 对应最小管腔面积只剩参考均值的 10%。

注意：锥形算例（``taper_stenosis``）的近/远端面积不等（0.15 / 0.09 cm²），
按上式 ``MLA = 0.12·(1−rate)``。当 rate ≤ 0.25 时 MLA 会高于较细一端，
此时「狭窄段」反而更宽（``SimulatedVessel`` 会给出 warning）。
本实验最低取 0.5，不会触发该边界。

运行::

    python Scripts/simudata_experiment_different_stenosis_rate.py           # 4 进程
    python Scripts/simudata_experiment_different_stenosis_rate.py --workers 6
    python Scripts/simudata_experiment_different_stenosis_rate.py --rates 0.9 0.5
    python Scripts/simudata_experiment_different_stenosis_rate.py --dry-run

复用缓存::

    python Scripts/simudata_experiment_different_stenosis_rate.py --list-cached
    python Scripts/simudata_experiment_different_stenosis_rate.py --replot
    python Scripts/simudata_experiment_different_stenosis_rate.py --replot rate_0.90/stenosis_center

输出（``results/模拟数据实验3（狭窄率变化）/``）::

    summary.json                              全部 20 例指标 + 扫描设置
    compare_ffr_by_rate.png                   每个狭窄率一个子图的 FFR(x)
    compare_ffr_stenosis_center.png           同一几何、不同狭窄率的 FFR 对比
    compare_ffr_min_vs_rate.png               最小 FFR 随狭窄率变化（4 条曲线）
    compare_pressure_drop_vs_rate.png         压降随狭窄率变化
    compare_q_scale_vs_rate.png               标定 q_scale 随狭窄率变化
    compare_mla_vs_rate.png                   MLA 与 FFR_min 的双轴关系
    compare_summary.png                       关键指标柱状图（按 case 分组）
    rate_<r>/<case>/geometry.png              面积剖面
    rate_<r>/<case>/final_<case>.html         Plotly 五图交互页
    rate_<r>/<case>/solver.pkl                solver 缓存（约 55–77 MB/例）

读取缓存做二次分析::

    from simudata_experiment_different_stenosis_rate import load_rate_case_solver

    bundle = load_rate_case_solver(0.9, "stenosis_center")
    solver = bundle["solver"]
    print(bundle["meta"]["stenosis_rate"])
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

_OUT_ROOT = os.path.join(_REPO_ROOT, "results", "模拟数据实验3（狭窄率变化）")

# 由重到轻：0.9 对应 MLA 只剩 10%
STENOSIS_RATES = (0.9, 0.8, 0.7, 0.6, 0.5)


def _rate_dir(rate: float) -> str:
    """子目录名，固定两位小数避免 0.9 与 0.90 不一致。"""
    return f"rate_{rate:.2f}"


def _rate_label(rate: float) -> str:
    """人类可读的狭窄程度说明。"""
    return f"狭窄率 {rate:.1f}（MLA = 参考的 {100 * (1 - rate):.0f}%）"


# ---------------------------------------------------------------------------
# 汇总对比图
# ---------------------------------------------------------------------------
def save_comparison_figures(results: Sequence[dict], out_root: str) -> list[str]:
    """按狭窄率分组的对比图。"""
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
    rates = sorted({r["stenosis_rate"] for r in ok})
    case_keys = sorted({r["key"] for r in ok})
    # 狭窄率由重到轻，配色由深到浅更直观
    colors = plt.cm.Reds(np.linspace(0.85, 0.30, len(rates)))

    def _by_rate(r: float) -> list[dict]:
        return sorted(
            [x for x in ok if x["stenosis_rate"] == r], key=lambda x: x["key"]
        )

    # ---- 1. FFR(x)：每个狭窄率一个子图 ----
    n = len(rates)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False
    )
    c = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in enumerate(rates):
        ax = axes[i // ncol][i % ncol]
        for j, row in enumerate(_by_rate(r)):
            ax.plot(row["_x_cm"], row["_ffr"], lw=1.7, color=c[j % 10],
                    label=f'{row["key"]} ({row["ffr_min"]:.3f})')
        ax.axhline(0.80, color="k", ls=":", lw=1.1)
        ax.set_title(_rate_label(r), fontsize=11)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("FFR")
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for i in range(n, nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("FFR 沿程分布 — 各狭窄率下的 4 种含狭窄几何", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_ffr_by_rate.png")
    fig.savefig(p, dpi=135)
    plt.close(fig)
    written.append(p)

    # ---- 2. 同一几何、不同狭窄率（选 stenosis_center 作代表）----
    ref = "stenosis_center" if "stenosis_center" in case_keys else case_keys[0]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, r in enumerate(rates):
        rows = [x for x in ok if x["stenosis_rate"] == r and x["key"] == ref]
        if not rows:
            continue
        row = rows[0]
        ax.plot(row["_x_cm"], row["_ffr"], lw=2.0, color=colors[i],
                label=f'rate={r:g}  (min={row["ffr_min"]:.4f}, '
                      f'MLA={row["mla_area_cm2"]:.4g} cm²)')
    ax.axhline(0.80, color="k", ls=":", lw=1.2, label="FFR = 0.80")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("FFR")
    ax.set_title(f"不同狭窄率下的 FFR 对比（几何 = {ref}）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_root, f"compare_ffr_{ref}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 3/4/5. 指标随狭窄率的曲线 ----
    panels = (
        ("ffr_min", "最小 FFR", "compare_ffr_min_vs_rate.png", 0.80),
        ("p_drop_mean", "时间平均压降 p(0)−p(L) (mmHg)",
         "compare_pressure_drop_vs_rate.png", None),
        ("q_scale", "标定 q_scale", "compare_q_scale_vs_rate.png", 1.0),
    )
    for field_name, ylabel, fname, hline in panels:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        cc = plt.cm.tab10(np.linspace(0, 1, 10))
        for j, key in enumerate(case_keys):
            xs, ys = [], []
            for r in rates:
                rows = [x for x in ok
                        if x["stenosis_rate"] == r and x["key"] == key]
                if rows:
                    xs.append(r)
                    ys.append(rows[0][field_name])
            if xs:
                # x 轴按「狭窄程度」递减排布更符合直觉，故反转
                ax.plot(xs, ys, "o-", lw=1.8, ms=6,
                        color=cc[j % 10], label=key)
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=1.1)
        ax.invert_xaxis()  # 左重右轻
        ax.set_xlabel("STENOSIS_RATE（左 = 狭窄更重）")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs 狭窄率")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_root, fname)
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    # ---- 6. MLA 与 FFR_min 的关系（双轴）----
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cc = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, key in enumerate(case_keys):
        rows = sorted(
            [x for x in ok if x["key"] == key], key=lambda x: x["mla_area_cm2"]
        )
        if not rows:
            continue
        ax.plot([x["mla_area_cm2"] for x in rows],
                [x["ffr_min"] for x in rows],
                "o-", lw=1.9, ms=6, color=cc[j % 10], label=key)
        # 标注狭窄率
        for x in rows:
            ax.annotate(f'{x["stenosis_rate"]:g}',
                        (x["mla_area_cm2"], x["ffr_min"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=cc[j % 10])
    ax.axhline(0.80, color="k", ls=":", lw=1.2, label="FFR = 0.80")
    ax.set_xlabel("MLA 最小管腔面积 (cm²)")
    ax.set_ylabel("最小 FFR")
    ax.set_title("最小 FFR 与 MLA 的关系（标注为狭窄率）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_mla_vs_rate.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # ---- 7. 关键指标柱状图（按 case × rate）----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    width = 0.8 / max(len(rates), 1)
    pos = np.arange(len(case_keys))
    sub_panels = (
        ("ffr_min", "最小 FFR", axes[0, 0], True),
        ("p_in_peak_mmhg", "入口峰值压 (mmHg)", axes[0, 1], False),
        ("q_scale", "标定 q_scale", axes[1, 0], False),
        ("p_drop_mean", "时间平均压降 (mmHg)", axes[1, 1], False),
    )
    for field_name, title, ax, show_thr in sub_panels:
        for i, r in enumerate(rates):
            lookup = {x["key"]: x for x in _by_rate(r)}
            vals = [
                lookup[k][field_name] if k in lookup else 0.0
                for k in case_keys
            ]
            ax.bar(
                pos + i * width - 0.4 + width / 2, vals, width,
                label=f"rate={r:g}", color=colors[i],
            )
        if show_thr:
            ax.axhline(0.80, color="k", ls=":", lw=1.1)
        ax.set_xticks(pos)
        ax.set_xticklabels(case_keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, title="STENOSIS_RATE")
    fig.suptitle("狭窄率实验 — 关键指标对比", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_root, "compare_summary.png")
    fig.savefig(p, dpi=135)
    plt.close(fig)
    written.append(p)

    return written


# ---------------------------------------------------------------------------
# 读取缓存
# ---------------------------------------------------------------------------
def rate_case_dir(rate: float, case_key: str) -> str:
    """某 (狭窄率, 几何) 的目录绝对路径。"""
    return os.path.abspath(
        os.path.join(_OUT_ROOT, _rate_dir(rate), case_key)
    )


def list_cached_cases(out_root: str = _OUT_ROOT) -> dict[str, str]:
    """列出所有已缓存算例，键形如 ``"rate_0.90/stenosis_center"``。"""
    if not os.path.isdir(out_root):
        return {}
    found: dict[str, str] = {}
    for rate_dir in sorted(os.listdir(out_root)):
        rd = os.path.join(out_root, rate_dir)
        if not os.path.isdir(rd) or not rate_dir.startswith("rate_"):
            continue
        for case_dir in sorted(os.listdir(rd)):
            p = os.path.join(rd, case_dir, base.SOLVER_FILENAME)
            if os.path.isfile(p):
                found[f"{rate_dir}/{case_dir}"] = os.path.abspath(p)
    return found


def load_rate_case_solver(
    rate: float | str,
    case_key: str,
    out_root: str = _OUT_ROOT,
) -> dict:
    """读取某个 (狭窄率, 几何) 组合的缓存。

    Parameters
    ----------
    rate :
        狭窄率（如 ``0.9``），或已是目录名（如 ``"rate_0.90"``）
    case_key :
        几何名，须是 4 个含狭窄算例之一：
        ``stenosis_center`` / ``taper_stenosis`` /
        ``stenosis_proximal`` / ``stenosis_distal``
    out_root :
        结果根目录

    Returns
    -------
    dict
        ``{"solver", "cal", "meta", "path"}``；``meta["stenosis_rate"]``
        即该例的狭窄率

    Examples
    --------
    ::

        bundle = load_rate_case_solver(0.9, "stenosis_center")
        solver = bundle["solver"]
        ffr = solver.ffr_ratio()
        print(bundle["meta"]["mla_area_cm2"])
    """
    if isinstance(rate, str):
        rate_dir = rate if rate.startswith("rate_") else _rate_dir(float(rate))
    else:
        rate_dir = _rate_dir(float(rate))

    path = os.path.join(out_root, rate_dir, case_key, base.SOLVER_FILENAME)
    if not os.path.exists(path):
        avail = list_cached_cases(out_root)
        hint = (
            "可用（前 20 个）：\n  " + "\n  ".join(list(avail)[:20])
            if avail
            else f"{out_root} 下没有任何缓存，请先运行本脚本。"
        )
        raise FileNotFoundError(
            f"未找到 rate={rate!r}, case={case_key!r} 的缓存：{path}\n{hint}"
        )

    bundle = load_solver_bundle(path)
    bundle["path"] = path
    return bundle


def replot_from_cache(
    rate: float | str,
    case_key: str,
    out_root: str = _OUT_ROOT,
    save_path: str | None = None,
    show: bool = False,
):
    """从缓存重新出五图交互页，不重跑仿真。"""
    bundle = load_rate_case_solver(rate, case_key, out_root)
    solver = bundle["solver"]
    meta = bundle.get("meta", {})
    r_val = meta.get("stenosis_rate", rate)
    q = bundle.get("cal", {}).get("q_scale")

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(bundle["path"]), f"replot_{case_key}.html"
        )

    title = f"rate={r_val:g} {case_key}"
    if isinstance(q, float):
        title += f" (q_scale={q:.3f})"
    return plot_results(solver, title=title, show=show, save_path=save_path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _json_safe(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _run_rate_job(job: tuple) -> dict:
    """执行一个 (几何, 目录名, 额外 meta, 输出根目录) 任务。

    必须是**模块级函数**：``ProcessPoolExecutor`` 要把任务对象 pickle 后发给
    子进程，闭包/局部函数无法序列化（会报
    ``AttributeError: Can't get local object``）。
    """
    spec, dir_name, extra, out_root = job
    r = base.run_one_case(spec, out_root, dir_name, extra)
    r["stenosis_rate"] = extra["stenosis_rate"]
    if r.get("ok"):
        # 时间平均压降：入口 vs 出口（用已算好的 p_mean）
        r["p_drop_mean"] = float(r["_p_mean"][0] - r["_p_mean"][-1])
    return r


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="狭窄率敏感性实验（4 种含狭窄几何 × 5 个 STENOSIS_RATE）"
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="并行进程数（默认 min(4, cpu_count)）",
    )
    parser.add_argument(
        "--rates", type=float, nargs="+", default=list(STENOSIS_RATES),
        help=f"要扫描的 STENOSIS_RATE 列表（默认 {list(STENOSIS_RATES)}）",
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
        help="列出已缓存的 (rate, case) 后退出",
    )
    parser.add_argument(
        "--replot", type=str, nargs="?", const="__ALL__", default=None,
        metavar="RATE_DIR/CASE",
        help="从缓存重新出图：形如 rate_0.90/stenosis_center；"
             "不带参数则重画全部",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ---- 子命令 ----
    if args.list_cached:
        cached = list_cached_cases(args.out)
        if not cached:
            print(f"{args.out} 下没有 solver 缓存。")
            print("请先运行：python Scripts/"
                  "simudata_experiment_different_stenosis_rate.py")
            return
        print(f"已缓存 {len(cached)} 个算例（{args.out}）：")
        for key, path in cached.items():
            mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {key:32} {mb:7.1f} MB  {path}")
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
                print(f"  [SKIP] {key}: 需要形如 rate_0.90/stenosis_center")
                continue
            try:
                fig = replot_from_cache(parts[0], parts[1], args.out)
                print(f"  [OK] {key}  ({len(fig.data)} traces)")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {key}: {type(exc).__name__}: {exc}")
        return

    # ---- 构造任务清单：rates × 4 个含狭窄几何 ----
    rates = sorted({float(r) for r in args.rates}, reverse=True)
    bad = [r for r in rates if not (0.0 <= r < 1.0)]
    if bad:
        raise SystemExit(f"--rates 必须全部落在 [0, 1) 内，非法值：{bad}")

    jobs: list[tuple] = []
    for r in rates:
        # only_stenosis=True：只取 4 个含狭窄的几何
        specs = base.build_cases(stenosis_rate=r, only_stenosis=True)
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
                    f"{_rate_dir(r)}/{spec.key}",
                    {"stenosis_rate": r},
                    args.out,
                )
            )

    if not jobs:
        raise SystemExit("没有匹配的算例，请检查 --only / --rates")

    n_cases = len({j[0].key for j in jobs})
    print(f"共 {len(jobs)} 个算例（{len(rates)} 个 STENOSIS_RATE "
          f"× {n_cases} 种含狭窄几何）")
    print(f"输出目录：{args.out}")
    print(f"狭窄率：{[f'{r:g}' for r in rates]}（由重到轻）")
    print("对应 MLA：0.5×(prox+dist)×(1−rate)，"
          "等截面例（A=0.15）为："
          f"{[f'{0.15 * (1 - r):.4g}' for r in rates]} cm²")
    print("仿真设置：L={} cm, n_cells={} (生成 {} → 重采样), "
          "duration={}s, AREA_SCALER=1.0, peak∈({}, {}) mmHg, "
          "virtual_stent=False".format(
              base.LENGTH_CM, base.RESAMPLE_NX, base.N_CELLS,
              base.DURATION_S, base.PEAK_LO_MMHG, base.PEAK_HI_MMHG))
    print(f"并行：{args.workers} 进程")
    print()

    if args.dry_run:
        for spec, dir_name, extra, _out in jobs:
            kw = spec.vessel_kwargs
            print(f"  {dir_name:32} rate={extra['stenosis_rate']:g}  "
                  f"A_prox={kw['prox_area_cm2']:.4g} "
                  f"A_dist={kw['dist_area_cm2']:.4g}")
        print("\n--dry-run：未执行仿真")
        return

    os.makedirs(args.out, exist_ok=True)
    t_all = time.perf_counter()
    results: list[dict] = []

    if args.workers <= 1:
        for job in jobs:
            print(f"[start] {job[1]}", flush=True)
            r = _run_rate_job(job)
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{status}] {r.get('case_dir', job[1])}  "
                  f"{r['wall_s']:.0f}s"
                  + ("" if r["ok"] else f"  {r['error']}"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_rate_job, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {
                        "ok": False,
                        "key": job[0].key,
                        "case_dir": job[1],
                        "stenosis_rate": job[2]["stenosis_rate"],
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
    results.sort(
        key=lambda r: (-r.get("stenosis_rate", 0), r.get("key", ""))
    )
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    summary = {
        "n_cases": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "wall_s_total": round(elapsed, 1),
        "workers": args.workers,
        "scan": {
            "parameter": "STENOSIS_RATE",
            "values": rates,
            "semantics": (
                "MLA = 0.5*(prox_area + dist_area)*(1 - STENOSIS_RATE)；"
                "取值越接近 1，狭窄越重"
            ),
            "geometries": [
                "stenosis_center", "taper_stenosis",
                "stenosis_proximal", "stenosis_distal",
            ],
        },
        "settings": {
            "length_cm": base.LENGTH_CM,
            "n_cells_generated": base.N_CELLS,
            "n_cells": base.RESAMPLE_NX,
            "resample_nx": base.RESAMPLE_NX,
            "dx_cm": base.LENGTH_CM / (base.RESAMPLE_NX - 1),
            "duration_s": base.DURATION_S,
            "record_interval_steps": base.RECORD_INTERVAL_STEPS,
            "peak_lo_mmhg": base.PEAK_LO_MMHG,
            "peak_hi_mmhg": base.PEAK_HI_MMHG,
            "max_iters": base.MAX_ITERS,
            "area_scaler": base.AREA_SCALER,
            "virtual_stent": False,
            "stenosis_position_center": base.STENOSIS_POSITION,
            "stenosis_fraction": base.STENOSIS_FRACTION,
            "branch_positions": base.BRANCH_POSITIONS,
            "branch_diameters_cm": base.BRANCH_DIAMETERS,
            "note": (
                "只包含 4 种含狭窄几何；uniform / taper / taper_reverse 与本扫描无关。"
                "锥形例（taper_stenosis）的近远端面积不等，MLA 按参考均值计算，"
                "因此其 MLA 与等截面例不同。"
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
    print("=" * 92)
    print(f"完成：{len(ok)}/{len(results)} 成功，总耗时 {elapsed:.0f}s")
    if ok:
        print(f"{'rate':>6} {'case':22} {'MLA':>9} {'q_scale':>8} "
              f"{'p_in':>8} {'ffr_min':>9} {'p_drop':>8} {'wall(s)':>8}")
        for r in ok:
            print(f"{r['stenosis_rate']:6.2f} {r['key']:22} "
                  f"{r['mla_area_cm2']:9.5f} {r['q_scale']:8.4f} "
                  f"{r['p_in_peak_mmhg']:8.2f} {r['ffr_min']:9.4f} "
                  f"{r['p_drop_mean']:8.2f} {r['wall_s']:8.0f}")
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
