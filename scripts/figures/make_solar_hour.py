#!/usr/bin/env python3
"""Reproduce fig_solar_vs_crowd_by_hour.png (fig:solar_hour) --- the daytime-filter figure.

1x2 layout: crowd count (left) and solar radiation (right) by hour, summer (Jun-Aug)
and winter (Dec-Feb) overlaid on each panel. Grey bands + dashed lines mark the hours
excluded by the 8:00-20:00 filter. Reads the raw 24-hour panel (dataset.csv.gz, which
keeps night hours and the om_ weather columns). Exported at IEEE \textwidth, serif 8 pt.

Run: python make_solar_hour.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data/dataset.csv.gz"
OUT = Path(__file__).resolve().parents[2] / "paper/figures/fig_solar_vs_crowd_by_hour.png"
SOLAR = "om_shortwave_radiation"
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "axes.linewidth": 0.6, "lines.linewidth": 1.2}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df["hour"] = df["ds"].dt.hour
    df["month"] = df["ds"].dt.month
    summer = df[df["month"].isin([6, 7, 8])]
    winter = df[df["month"].isin([12, 1, 2])]

    with plt.rc_context(RC):
        fig, (axc, axs) = plt.subplots(1, 2, figsize=(7.139, 2.7))
        for sub, lab, col in [(summer, "Summer (Jun-Aug)", "#e67e22"),
                              (winter, "Winter (Dec-Feb)", "#2980b9")]:
            hc = sub.groupby("hour")["y"].mean()
            axc.plot(hc.index, hc.values, "-o", ms=2.5, color=col, label=lab)
            hs = sub.groupby("hour")[SOLAR].mean()
            axs.plot(hs.index, hs.values, "-o", ms=2.5, color=col, label=lab)
        for ax, ttl, yl in [(axc, "Crowd count", "Mean crowd count"),
                            (axs, "Solar radiation", r"Shortwave (W/m$^2$)")]:
            ax.axvspan(-0.5, 7.5, color="0.5", alpha=0.13)
            ax.axvspan(20.5, 23.5, color="0.5", alpha=0.13)
            ax.axvline(7.5, ls="--", lw=0.7, color="0.35")
            ax.axvline(20.5, ls="--", lw=0.7, color="0.35")
            ax.set_xlim(-0.5, 23.5)
            ax.set_xticks([0, 4, 8, 12, 16, 20])
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(yl)
            ax.set_title(ttl)
            ax.text(2.0, ax.get_ylim()[1] * 0.5, "excluded", fontsize=6,
                    color="0.4", ha="center", rotation=90, style="italic")
        axc.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
