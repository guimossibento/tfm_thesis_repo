"""Season/summer per-series-P90 relMAE + Diebold-Mariano for a walk-forward horizon.
Reproduces the thesis 3d/10d numbers EXACTLY; reusable for 15d when its run finishes.

  relMAE: mean of per-series (MAE_season / P90_train) over the per_beach beaches (those
          with a valid train P90). Reproduces 3d 14.6/22.1/26.3 and 10d 19.2/23.8/26.9.
  DM:     one pooled concatenated series over ALL identical_rows beaches, season rows,
          HAC (Newey-West, bw=floor(4*(n/100)^(2/9))) + HLN, normal reference.
          Reproduces 10d vs XGB -4.16 / vs LSTM -32.68 and 3d -8.48 / -17.39.

Usage: python wf_aggregate.py <identical_rows.csv> <per_beach_with_P90.csv> <label>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

MODELS = ["TFT", "LSTM", "XGB"]
SEASON, SUMMER = [4, 5, 6, 7, 8, 9], [6, 7, 8]


def per_series_relmae(df, p90, months):
    sub = df[df["ds"].dt.month.isin(months)]
    out = {}
    for m in MODELS:
        per = []
        for uid, g in sub.groupby("unique_id"):
            cap = p90.get(uid)
            if cap is None or pd.isna(cap) or cap <= 0:
                continue
            per.append((g["y_true"] - g[m]).abs().mean() / cap)
        out[m] = 100 * float(np.mean(per)) if per else float("nan")
    return out, len(per)


def dm_pooled(df, a, b, months):
    """One pooled HAC Diebold-Mariano (MAE loss, HLN correction) over all beaches."""
    sub = df[df["ds"].dt.month.isin(months)].sort_values(["unique_id", "ds"])
    d = (sub["y_true"] - sub[a]).abs().values - (sub["y_true"] - sub[b]).abs().values
    n = len(d)
    dbar = d.mean()
    h = int(np.floor(4 * (n / 100) ** (2 / 9))) or 1
    g0 = np.mean((d - dbar) ** 2)
    var = g0 + 2 * sum((1 - k / (h + 1)) * np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, h + 1))
    dm = dbar / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * norm.sf(abs(dm)), n


def main():
    ir, pb_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    df = pd.read_csv(ir, parse_dates=["ds"])
    pb = pd.read_csv(pb_path)
    p90 = dict(zip(pb["unique_id"], pb["P90"]))
    print(f"\n=== {label} | {len(df)} rows, {df['unique_id'].nunique()} beaches (relMAE over {len(p90)} with P90) ===")
    for name, months in [("season", SEASON), ("summer", SUMMER)]:
        rel, n = per_series_relmae(df, p90, months)
        print(f"[{name:6}] relMAE %: " + "  ".join(f"{m} {rel[m]:.1f}" for m in MODELS) + f"   (n_series={n})")
    for a, b in [("TFT", "XGB"), ("TFT", "LSTM")]:
        dm, p, n = dm_pooled(df, a, b, SEASON)
        print(f"[DM season] {a} vs {b}: DM={dm:.2f}, p={p:.2e}, n={n}")


if __name__ == "__main__":
    main()
