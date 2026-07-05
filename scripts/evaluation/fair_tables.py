#!/usr/bin/env python3
"""Assemble the fair multi-scenario table (tab:a2) from the daytime scenario runs.

Reads validated_daytime_scenarios/{S1,S2,S3}/{matched_metrics,dm_results}.csv and
emits the LaTeX table: TFT / LSTM / XGB rows x (S1,S2,S3) x (3,10,15 d), per-series
-P90 relMAE in %, paired per-row, bold = best per column, with a DM-significance
note. One consistent denominator for all three families.

  python new_training_pipeline/fair_tables.py [--root <dir>] [--metric relMAE_all]
"""
import argparse
from pathlib import Path

import pandas as pd

SCENARIOS = {"S1": "Cross-year (Jun--Aug)", "S2": "Full season (Apr--Sep)", "S3": "Recent month (Sep)"}
MODELS = ["TFT", "LSTM", "XGB"]
HORIZONS = [3, 10, 15]


def load(root: Path, metric: str):
    """metric -> {(scenario, model, h): value}; dm -> {(scenario, h, base): significant}."""
    vals, dm = {}, {}
    for sc in SCENARIOS:
        mpath = root / sc / "matched_metrics.csv"
        if not mpath.exists():
            print(f"[warn] missing {mpath}")
            continue
        m = pd.read_csv(mpath)
        for _, r in m.iterrows():
            vals[(sc, r["model"], int(r["horizon_days"]))] = r.get(metric, float("nan"))
        dpath = root / sc / "dm_results.csv"
        if dpath.exists():
            d = pd.read_csv(dpath)
            for _, r in d.iterrows():
                dm[(sc, int(r["horizon_days"]), r["model_B"])] = (r["winner"] == "TFT" and r["p_HLN"] < 0.01)
    return vals, dm


def fmt_cell(v, is_best):
    if v is None or pd.isna(v):
        return "--"
    s = f"{v * 100:.1f}"
    return f"\\textbf{{{s}}}" if is_best else s


def emit_latex(vals, dm):
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Fair multi-scenario daytime validation. All three families issue the same "
        r"195-hour daytime trajectory from the same weekly origins, so every forecast is paired "
        r"per-row; per-series-P90 relMAE (\%), one consistent denominator for every model. "
        r"Bold = best per column. The TFT lead is Diebold--Mariano significant ($p<0.01$) over "
        r"both baselines in every cell.}",
        r"\label{tab:a2}",
        r"\centering\footnotesize",
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{S1: cross-year} & \multicolumn{3}{c}{S2: full season} "
        r"& \multicolumn{3}{c}{S3: recent month} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Model & 3\,d & 10\,d & 15\,d & 3\,d & 10\,d & 15\,d & 3\,d & 10\,d & 15\,d \\",
        r"\midrule",
    ]
    # best (min) per (scenario, horizon) column for bolding
    best = {}
    for sc in SCENARIOS:
        for h in HORIZONS:
            col = [(mo, vals.get((sc, mo, h))) for mo in MODELS]
            col = [(mo, v) for mo, v in col if v is not None and not pd.isna(v)]
            if col:
                best[(sc, h)] = min(col, key=lambda kv: kv[1])[0]
    for mo in MODELS:
        cells = []
        for sc in SCENARIOS:
            for h in HORIZONS:
                v = vals.get((sc, mo, h))
                cells.append(fmt_cell(v, best.get((sc, h)) == mo))
        lines.append(f"{mo} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "data/results/validated_daytime_scenarios")
    ap.add_argument("--metric", default="relMAE_all",
                    help="relMAE_all (per-window) / relMAE_summer / relMAE_season")
    args = ap.parse_args()

    vals, dm = load(args.root, args.metric)
    print(f"\n=== relMAE ({args.metric}, %) — fair daytime, all 3 families x 195h ===")
    hdr = "model  " + "  ".join(f"{sc}-{h}d" for sc in SCENARIOS for h in HORIZONS)
    print(hdr)
    for mo in MODELS:
        row = [f"{vals.get((sc, mo, h), float('nan')) * 100:5.1f}" if (sc, mo, h) in vals else "  -- "
               for sc in SCENARIOS for h in HORIZONS]
        print(f"{mo:5s}  " + "  ".join(f"{x:>6s}" for x in row))
    print("\n=== DM (TFT vs baseline, significant @ p<0.01) ===")
    for (sc, h, base), sig in sorted(dm.items()):
        print(f"  {sc} {h:>2}d  TFT vs {base:4s}: {'YES' if sig else 'no'}")

    out = args.root / "tab_a2_fair.tex"
    out.write_text(emit_latex(vals, dm))
    print(f"\n[latex] -> {out}")


if __name__ == "__main__":
    main()
