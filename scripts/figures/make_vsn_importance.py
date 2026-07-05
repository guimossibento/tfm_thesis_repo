"""Build the TFT Variable Selection Network feature-importance figure for the thesis.

Reads the per-horizon VSN gating weights saved by the final-deploy run
(tft_grid_results_v6/{3d,10d,15d}/fi_final_deploy.csv) and renders a 2x3 grid:
top row = known-future stream, bottom row = past-observed stream, one column per
horizon. Weights within each stream sum to 1 (softmax gating).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "server_files/weather_data/tft_grid_results_v6"
OUT_FIG = Path(__file__).resolve().parents[2] / "paper/figures/fig_tft_vsn_importance.png"
OUT_CSV = Path(__file__).resolve().parent / "vsn_importance_consolidated.csv"

HORIZONS = ["3d", "10d", "15d"]
FUTURE = "Future variable importance over time"
PAST = "Past variable importance over time"

PRETTY = {
    "hour": "Hour",
    "day_of_week": "Day of week",
    "month": "Month",
    "is_weekend": "Is weekend",
    "is_summer": "Is summer",
    "is_holiday": "Is holiday",
    "observed_target": "Observed target",
    "om_temperature_2m": "2m temperature",
    "om_apparent_temperature": "Apparent temp.",
    "om_shortwave_radiation": "Shortwave rad.",
    "om_wind_speed_10m": "Wind speed 10m",
}
C_FUTURE = "#2c6fbb"
C_PAST = "#d9822b"


def load():
    frames = []
    for h in HORIZONS:
        d = pd.read_csv(SRC / h / "fi_final_deploy.csv")
        d["horizon"] = h
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def stream_order(df, stream):
    ref = df[(df["group"] == stream) & (df["horizon"] == "3d")]
    return ref.sort_values("importance")["feature"].tolist()


def main():
    df = pd.read_csv(OUT_CSV)

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
        "mathtext.fontset": "stix",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.139, 4.0), sharex="row")

    for row, (stream, color, label) in enumerate([
        (FUTURE, C_FUTURE, "Known-future stream"),
        (PAST, C_PAST, "Past-observed stream"),
    ]):
        order = stream_order(df, stream)
        ypos = np.arange(len(order))
        for col, h in enumerate(HORIZONS):
            ax = axes[row, col]
            sub = df[(df["group"] == stream) & (df["horizon"] == h)].set_index("feature")
            vals = [sub.loc[f, "importance"] if f in sub.index else 0.0 for f in order]
            ax.barh(ypos, vals, color=color, edgecolor="black", linewidth=0.4)
            ax.set_yticks(ypos)
            ax.set_yticklabels([PRETTY.get(f, f) for f in order] if col == 0 else [])
            ax.set_xlim(0, max(0.7, max(vals) * 1.15))
            ax.grid(axis="x", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            for y, v in zip(ypos, vals):
                if v > 0.01:
                    ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=6)
            if row == 0:
                ax.set_title(f"{h} horizon", fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontweight="bold")
            if row == 1:
                ax.set_xlabel("VSN selection weight")

    fig.suptitle(
        "TFT Variable Selection Network: per-feature gating weights by stream and horizon\n"
        "(static covariates: per-series mean 0.85, per-series CV 0.15)",
        fontsize=8.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
