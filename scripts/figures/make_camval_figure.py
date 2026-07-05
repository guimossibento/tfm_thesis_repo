#!/usr/bin/env python3
"""One clean, thesis-sized camera-validation figure.

Replicates the seasonality test from the data-pipeline notebook
(generate_train_data.ipynb, cell 20: stuck-period removal + per-series
summer/winter ratio and Cohen's d) but renders ONE readable summer-vs-winter
scatter instead of the 4-panel, 40-label montage. Ratio thresholds (1.3x weak,
2x strong) are drawn as rays from the origin; only the flagged (excluded)
cameras are labelled. Output: figures/fig_camval_scatter.png at the 10pt thesis
size (include with width=\\columnwidth).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import use_thesis_style, COL_W

_ROOT = Path(__file__).resolve().parents[2]
CSV = str(_ROOT / "data/dataset.csv") if (_ROOT / "data/dataset.csv").exists() else str(_ROOT / "data/dataset.csv.gz")
OUT = _ROOT / "paper/figures/fig_camval_scatter.png"
summer_months, winter_months = [6, 7, 8], [12, 1, 2]
warm_months, cold_months = [5, 6, 7, 8, 9], [11, 12, 1, 2, 3]
CV_WINDOW, CV_THRESHOLD = 48, 0.05


def remove_stuck(subset):
    parts = []
    for (_, _), grp in subset.groupby(["unique_id", "source"]):
        grp = grp.sort_values("ds").copy()
        if len(grp) < CV_WINDOW:
            parts.append(grp); continue
        rm = grp["y"].rolling(CV_WINDOW, center=True, min_periods=12).mean()
        rs = grp["y"].rolling(CV_WINDOW, center=True, min_periods=12).std()
        cv = (rs / rm).where(rm > 0, np.nan)
        parts.append(grp[~(cv < CV_THRESHOLD).fillna(False)])
    return pd.concat(parts, ignore_index=True)


def analyse_seasonality(subset):
    rows = []
    for beach in sorted(subset["unique_id"].unique()):
        b = subset[subset["unique_id"] == beach]
        s = b[b["month"].isin(summer_months)]["y"]
        w = b[b["month"].isin(winter_months)]["y"]
        if len(s) < 10:
            s = b[b["month"].isin(warm_months)]["y"]
        if len(w) < 10:
            w = b[b["month"].isin(cold_months)]["y"]
        if len(s) < 10 or len(w) < 10:
            continue
        ratio = s.mean() / w.mean() if w.mean() > 0 else np.nan
        pooled = np.sqrt((s.std() ** 2 + w.std() ** 2) / 2)
        d = (s.mean() - w.mean()) / pooled if pooled > 0 else 0.0
        cls = "Strong" if (ratio > 2 or d > 0.5) else ("Weak" if ratio > 1.3 else "No signal")
        rows.append({"beach": beach, "summer_mean": s.mean(), "winter_mean": w.mean(),
                     "ratio": ratio, "cohens_d": d, "cls": cls})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(CSV, parse_dates=["ds"])
    raw = raw[(raw["hour"] >= 8) & (raw["hour"] <= 20)]
    clean = remove_stuck(raw)
    dj = clean[clean["source"].isin(["django", "both"])].copy()
    sea = analyse_seasonality(dj).dropna(subset=["summer_mean", "winter_mean", "ratio"])
    print("[camval] class breakdown:", sea["cls"].value_counts().to_dict())
    print("[camval] No-signal (excluded):", sorted(sea[sea["cls"] == "No signal"]["beach"]))
    print("[camval] Weak (borderline):   ", sorted(sea[sea["cls"] == "Weak"]["beach"]))

    use_thesis_style(8)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from thesis_figure_style import TEXT_W
    palette = {"Strong": "#2ecc71", "Weak": "#f39c12", "No signal": "#e74c3c"}
    sea = sea.sort_values("ratio").reset_index(drop=True)
    XCAP = 12.0                                            # cap so the 1.3x/2x region stays readable
    fig, ax = plt.subplots(figsize=(TEXT_W, max(4.8, 0.155 * len(sea))))
    ax.barh(range(len(sea)), sea["ratio"].clip(upper=XCAP), color=sea["cls"].map(palette),
            edgecolor="black", linewidth=0.3, zorder=3)
    for i, r in enumerate(sea["ratio"]):                  # annotate the few bars past the cap
        if r > XCAP:
            ax.annotate(f"{r:.0f}×", (XCAP, i), fontsize=6, va="center",
                        xytext=(2, 0), textcoords="offset points")
    ax.set_yticks(range(len(sea)))
    ax.set_yticklabels(sea["beach"], fontsize=8)
    ax.set_ylim(-0.6, len(sea) - 0.4)
    ax.axvline(1.0, color="0.45", ls="-", lw=0.7, alpha=0.6, zorder=2)
    ax.axvline(1.3, color="#f39c12", ls="--", lw=1.1, zorder=2)
    ax.axvline(2.0, color="#2ecc71", ls="--", lw=1.1, zorder=2)
    ax.set_xlim(0, XCAP + 0.6)
    ax.set_xlabel(r"Summer / winter mean ratio (bars capped at 12$\times$)")
    handles = [Patch(facecolor=palette[c], edgecolor="black", label=l) for c, l in
               [("Strong", "Strong (kept)"), ("Weak", "Weak (flagged)"), ("No signal", "No signal (excluded)")]]
    handles += [plt.Line2D([], [], color="0.45", ls="-", label=r"1$\times$ (no seasonality)"),
                plt.Line2D([], [], color="#2ecc71", ls="--", label=r"2$\times$ (strong)"),
                plt.Line2D([], [], color="#f39c12", ls="--", label=r"1.3$\times$ (weak)")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"[camval] -> {OUT} ({len(sea)} series)")


if __name__ == "__main__":
    main()
