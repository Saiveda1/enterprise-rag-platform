"""Render product-grade PNG dashboards from the real eval run (benchmarks/results.json).

Generates into assets/:
    quality_comparison.png  — Recall@k / nDCG bars across the 4 methods
    latency_vs_k.png        — mean per-query latency vs retrieval depth
    scorecard.png           — KPI-tile eval dashboard (multi-panel)
    reranker_gain.png       — per-metric lift from reranking + score separation

    python scripts/make_screenshots.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ragplatform.viztheme import (
    ACCENT,
    BAD,
    GOOD,
    GRID,
    MUTED,
    PALETTE,
    PANEL,
    TEXT,
    WARN,
    apply_theme,
    kpi,
    save_panel,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "benchmarks" / "results.json"

METHOD_COLORS = {
    "BM25": PALETTE[3],
    "Dense": PALETTE[0],
    "Hybrid": PALETTE[4],
    "Hybrid+Rerank": GOOD,
}


def _load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("Run scripts/run_eval.py first to produce benchmarks/results.json")
    data = json.loads(RESULTS.read_text())
    # JSON stringifies int dict keys — coerce recall@k keys back to int.
    for m, v in data["quality"].items():
        v["recall"] = {int(k): val for k, val in v["recall"].items()}
    return data


def chart_quality(data: dict) -> None:
    q = data["quality"]
    methods = [m for m in METHOD_COLORS if m in q]
    metrics = [("recall@1", lambda v: v["recall"][1]),
               ("recall@5", lambda v: v["recall"][5]),
               ("nDCG@10", lambda v: v["ndcg"]),
               ("MRR", lambda v: v["mrr"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    w = 0.2
    for i, m in enumerate(methods):
        vals = [f(q[m]) for _, f in metrics]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                      label=m, color=METHOD_COLORS[m], edgecolor=PANEL, linewidth=0.5)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval quality — BM25 vs Dense vs Hybrid vs Hybrid+Rerank")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    save_panel(fig, str(ASSETS / "quality_comparison.png"))


def chart_latency(data: dict) -> None:
    lat = data["latency"]
    ks = data["latency_ks"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHOD_COLORS:
        if m not in lat:
            continue
        ys = [lat[m][str(k)] for k in ks]
        ax.plot(ks, ys, marker="o", label=m, color=METHOD_COLORS[m], linewidth=2)
    ax.set_xlabel("retrieval depth k")
    ax.set_ylabel("mean latency per query (ms)")
    ax.set_title("Query latency vs retrieval depth")
    ax.legend(loc="upper right")
    ax.set_xticks(ks)
    save_panel(fig, str(ASSETS / "latency_vs_k.png"))


def chart_scorecard(data: dict) -> None:
    q = data["quality"]
    cfg = data["config"]
    best = "Hybrid+Rerank" if "Hybrid+Rerank" in q else max(q, key=lambda m: q[m]["mrr"])
    bm = q.get("BM25", q[best])
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.3,
                          height_ratios=[1, 1, 1.5])

    # KPI tiles (top two rows).
    tiles = [
        ("chunks indexed", f"{cfg['chunks']:,}", "streaming generator", ACCENT),
        ("embed dim", f"{cfg['dim']}", "TF-IDF + SVD (LSA)", PALETTE[0]),
        ("QA queries", f"{cfg['queries']}", "held-out eval", PALETTE[6]),
        ("index build", f"{cfg['build_s']:.1f}s", "offline, 1 thread", WARN),
        ("best nDCG@10", f"{q[best]['ndcg']:.3f}", best, GOOD),
        ("best MRR", f"{q[best]['mrr']:.3f}", best, GOOD),
        ("best Recall@5", f"{q[best]['recall'][5]:.3f}", best, PALETTE[4]),
        ("groundedness", f"{q[best]['grounded']:.3f}", "faithfulness proxy", PALETTE[0]),
    ]
    for i, (label, value, sub, color) in enumerate(tiles):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        kpi(ax, label, value, sub, color=color)

    # Bottom-left: nDCG per method.
    axb = fig.add_subplot(gs[2, :2])
    methods = [m for m in METHOD_COLORS if m in q]
    ndcgs = [q[m]["ndcg"] for m in methods]
    axb.barh(methods, ndcgs, color=[METHOD_COLORS[m] for m in methods],
             edgecolor=PANEL)
    for y, val in enumerate(ndcgs):
        axb.text(val + 0.01, y, f"{val:.3f}", va="center", fontsize=8, color=MUTED)
    axb.set_xlim(0, 1.08)
    axb.set_title("nDCG@10 by method")
    axb.invert_yaxis()

    # Bottom-right: recall curve per method.
    axc = fig.add_subplot(gs[2, 2:])
    ks = data["ks"]
    for m in methods:
        ys = [q[m]["recall"][k] for k in ks]
        axc.plot(ks, ys, marker="o", label=m, color=METHOD_COLORS[m], linewidth=2)
    axc.set_xlabel("k")
    axc.set_ylabel("recall@k")
    axc.set_title("Recall@k")
    axc.set_xticks(ks)
    axc.legend(fontsize=7, loc="lower right")

    save_panel(fig, str(ASSETS / "scorecard.png"),
               suptitle="Enterprise RAG Platform — Evaluation Scorecard")


def chart_reranker_gain(data: dict) -> None:
    q = data["quality"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    base = "Hybrid" if "Hybrid" in q else "BM25"
    if "Hybrid+Rerank" in q:
        metrics = ["recall@1", "recall@5", "mrr", "ndcg@10", "grounded"]
        getters = [lambda v: v["recall"][1], lambda v: v["recall"][5],
                   lambda v: v["mrr"], lambda v: v["ndcg"], lambda v: v["grounded"]]
        base_v = [g(q[base]) for g in getters]
        rr_v = [g(q["Hybrid+Rerank"]) for g in getters]
        x = np.arange(len(metrics))
        ax1.bar(x - 0.2, base_v, 0.4, label=base, color=PALETTE[4], edgecolor=PANEL)
        ax1.bar(x + 0.2, rr_v, 0.4, label="Hybrid+Rerank", color=GOOD, edgecolor=PANEL)
        for xi, (bv, rv) in enumerate(zip(base_v, rr_v)):
            delta = rv - bv
            ax1.text(xi + 0.2, rv + 0.01, f"{'+' if delta >= 0 else ''}{delta:.03f}",
                     ha="center", va="bottom", fontsize=7,
                     color=GOOD if delta >= 0 else BAD)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=20, ha="right")
        ax1.set_ylim(0, 1.08)
        ax1.set_title(f"Reranker lift over {base}")
        ax1.legend(loc="lower right", fontsize=8)

    # Right: method scores as a grouped lollipop for MRR / nDCG.
    methods = [m for m in METHOD_COLORS if m in q]
    mrrs = [q[m]["mrr"] for m in methods]
    ndcgs = [q[m]["ndcg"] for m in methods]
    y = np.arange(len(methods))
    ax2.hlines(y, [min(a, b) for a, b in zip(mrrs, ndcgs)],
               [max(a, b) for a, b in zip(mrrs, ndcgs)], color=GRID, linewidth=2)
    ax2.scatter(mrrs, y, color=ACCENT, s=70, label="MRR", zorder=3)
    ax2.scatter(ndcgs, y, color=WARN, s=70, label="nDCG@10", zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Ranking quality by method")
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    save_panel(fig, str(ASSETS / "reranker_gain.png"))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    apply_theme()
    data = _load()
    chart_quality(data)
    chart_latency(data)
    chart_scorecard(data)
    chart_reranker_gain(data)
    print(f"[screenshots] wrote 4 PNGs to {ASSETS}/")


if __name__ == "__main__":
    main()
