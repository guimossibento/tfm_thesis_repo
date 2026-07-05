#!/usr/bin/env python3
"""TFT-only fANOVA hyperparameter importance figure (15-day horizon).
Reads the existing param_importance_fanova.csv and plots just the TFT rows."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))
import thesis_figure_style  # noqa: F401 — sets thesis rcParams on import
import matplotlib.pyplot as plt
import pandas as pd

CSV = Path(__file__).resolve().parents[2] / "data/hp_methodology/param_importance_fanova.csv"
OUT = Path(__file__).resolve().parents[2] / "paper/figures/fig_tft_fanova_importance.png"
LABELS = {
    "scaler": "scaler", "lr": "learning rate", "batch_size": "batch size",
    "dropout": "dropout", "n_head": "n\\_head", "hidden_size": "hidden size",
    "attn_dropout": "attn dropout", "input_size": "input size",
    "early_stop_patience": "early-stop patience",
}

df = pd.read_csv(CSV)
g = df[(df["model"] == "tft") & (df["horizon"] == "15d")].sort_values("importance")
labels = [LABELS.get(p, p).replace("\\_", "_") for p in g["param"]]

plt.rcParams.update({"font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7})
fig, ax = plt.subplots(figsize=(3.21, 2.5))
# searched axes (top-3) in a stronger colour, frozen ones muted
colors = ["#1f77b4" if v >= 0.10 else "#9bb9d6" for v in g["importance"]]
bars = ax.barh(labels, g["importance"], color=colors)
for b, v in zip(bars, g["importance"]):
    ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=6)
ax.set_xlabel("fANOVA importance")
ax.set_xlim(0, max(g["importance"]) * 1.18)
ax.margins(y=0.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
