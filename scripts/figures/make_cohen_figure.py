#!/usr/bin/env python3
"""Companion to make_camval_figure.py: per-series Cohen's d (summer vs winter),
the effect-size half of the seasonality test. Same stuck-removal + seasonality
logic (imported), one thesis-sized horizontal bar chart coloured by class, with
the d=0.5 keep threshold and 0.2/0.8 small/large references drawn as lines.
Output: figures/fig_cohen_scatter.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_camval_figure import remove_stuck, analyse_seasonality, CSV
from thesis_figure_style import use_thesis_style, TEXT_W

OUT = Path(__file__).resolve().parents[2] / "paper/figures/fig_cohen_scatter.png"


def main():
    raw = pd.read_csv(CSV, parse_dates=["ds"])
    raw = raw[(raw["hour"] >= 8) & (raw["hour"] <= 20)]
    clean = remove_stuck(raw)
    dj = clean[clean["source"].isin(["django", "both"])].copy()
    sea = analyse_seasonality(dj).dropna(subset=["cohens_d"])
    sea = sea.sort_values("cohens_d").reset_index(drop=True)
    print("[cohen] d range:", round(sea["cohens_d"].min(), 2), "->", round(sea["cohens_d"].max(), 2))
    print("[cohen] class breakdown:", sea["cls"].value_counts().to_dict())

    use_thesis_style(8)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    palette = {"Strong": "#2ecc71", "Weak": "#f39c12", "No signal": "#e74c3c"}
    XCAP = 3.0
    fig, ax = plt.subplots(figsize=(TEXT_W, max(4.8, 0.155 * len(sea))))
    ax.barh(range(len(sea)), sea["cohens_d"].clip(upper=XCAP), color=sea["cls"].map(palette),
            edgecolor="black", linewidth=0.3, zorder=3)
    for i, d in enumerate(sea["cohens_d"]):
        if d > XCAP:
            ax.annotate(f"{d:.1f}", (XCAP, i), fontsize=6, va="center",
                        xytext=(2, 0), textcoords="offset points")
    ax.set_yticks(range(len(sea)))
    ax.set_yticklabels(sea["beach"], fontsize=8)
    ax.set_ylim(-0.6, len(sea) - 0.4)
    ax.axvline(0.0, color="0.45", ls="-", lw=0.7, alpha=0.6, zorder=2)
    ax.axvline(0.2, color="0.6", ls=":", lw=1.0, zorder=2)
    ax.axvline(0.5, color="#2ecc71", ls="--", lw=1.1, zorder=2)
    ax.axvline(0.8, color="0.6", ls=":", lw=1.0, zorder=2)
    ax.set_xlim(min(-0.3, sea["cohens_d"].min() - 0.1), XCAP + 0.5)
    ax.set_xlabel(r"Cohen's $d$ (summer vs winter, standardised by pooled SD; capped at 3)")
    handles = [Patch(facecolor=palette[c], edgecolor="black", label=l) for c, l in
               [("Strong", "Strong (kept)"), ("Weak", "Weak (flagged)"), ("No signal", "No signal (excluded)")]]
    handles += [plt.Line2D([], [], color="#2ecc71", ls="--", label=r"$d=0.5$ (medium, kept)"),
                plt.Line2D([], [], color="0.6", ls=":", label=r"$d=0.2/0.8$ (small/large)")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"[cohen] -> {OUT} ({len(sea)} series)")


if __name__ == "__main__":
    main()
