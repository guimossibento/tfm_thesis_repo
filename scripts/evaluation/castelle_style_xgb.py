#!/usr/bin/env python3
"""Castelle-style XGB (option 1) — ONE regression model, full hourly trajectory.

Castelle et al. (2025) do NOT do a multi-horizon forecast: they regress hourly
attendance on (hour of day, weekday, weather, tide), so a SINGLE model covers
every hour. The trajectory for the next 15 days is just the model queried for
each hour with that hour's (forecast) weather — no autoregressive lag, no
per-horizon target, no 180 models.

This script builds exactly that and compares it, PER FORECAST LEAD, against the
TFT trajectory (which uses recent occupancy context and so degrades as the lead
grows). The expected picture: the TFT wins at short lead (fresh context); the
Castelle-XGB is flat across lead (it always sees the target hour's weather), so
it can catch up at long lead. This is the apples-to-apples trajectory comparison.

Usage:
  python new_training_pipeline/castelle_style_xgb.py \
    --panel data/all_clean.csv \
    --validated-run new_training_pipeline_server_1906/validated_run --tft-proto walkforward \
    --test-start 2025-06-01 --test-end 2026-02-28 --out new_training_pipeline/castelle_xgb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CASTELLE_WEATHER = ["om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
                    "om_wind_direction_10m", "om_shortwave_radiation"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}
SEED = 42


def _slugify_uid(name: str) -> str:
    """Match the retrain's unique_id normalisation (display names -> slugs)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"[-_]\d+$", "", s)
    return s


def load_panel(path: Path, cap_k: float = 1.5) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()
    df["unique_id"] = df["unique_id"].map(_slugify_uid)
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # per-series cap of the real y above k*P90 (daytime positives), vectorised
    day = df[df["hour"].between(8, 20) & (df["y"] > 0)]
    cap = (cap_k * day.groupby("unique_id")["y"].quantile(0.9)).to_dict()
    capv = df["unique_id"].map(cap)
    m = capv.notna() & (df["y"] > capv)
    df.loc[m, "y"] = capv[m]
    return df


def per_series_relmae(d: pd.DataFrame, cap: dict) -> float:
    rel = []
    for uid, g in d.groupby("unique_id"):
        c = cap.get(uid)
        if not c or c != c or c <= 0:
            continue
        rel.append(float((g["y_pred"] - g["y_true"]).abs().mean()) / c)
    return float(np.mean(rel)) if rel else float("nan")


def train_castelle_xgb(panel: pd.DataFrame, test_start: str, test_end: str,
                       trials: int, train_era: str = "all") -> tuple[pd.DataFrame, dict, list[str]]:
    import optuna
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base = CALENDAR + [c for c in CASTELLE_WEATHER if c in panel.columns]
    day = panel[panel["hour"].between(8, 20)].dropna(subset=base + ["y"]).copy()
    # Static per-beach level features (train only) — the multi-beach analogue of
    # Castelle's single-beach setup; without them a global model can't set each
    # beach's scale. This is the static-covariate role the TFT's encoder plays.
    pretest = day[day["ds"] < pd.Timestamp(test_start)]      # static level + capacity source
    bp90 = pretest.groupby("unique_id")["y"].quantile(0.9)
    bmean = pretest.groupby("unique_id")["y"].mean()
    day["beach_p90"] = day["unique_id"].map(bp90)
    day["beach_mean"] = day["unique_id"].map(bmean)
    feat_cols = base + ["beach_p90", "beach_mean"]
    day = day.dropna(subset=feat_cols)
    # XGB fit data — match the NF models' training era (cache2022 trains on 2022 only)
    if train_era == "2022":
        train = day[day["ds"].dt.year == 2022]
    else:
        train = day[day["ds"] < pd.Timestamp(test_start)]
    test = day[(day["ds"] >= pd.Timestamp(test_start)) & (day["ds"] <= pd.Timestamp(test_end))]
    print(f"[castelle] train_era={train_era}  features={feat_cols}")
    print(f"[castelle] train rows={len(train):,}  test rows={len(test):,}  beaches={test['unique_id'].nunique()}")

    # inner temporal split of TRAIN for tuning (never the test window)
    cut = train["ds"].quantile(0.82)
    itr, iva = train[train["ds"] < cut], train[train["ds"] >= cut]
    Xa, ya = itr[feat_cols].values, itr["y"].values
    Xb, yb = iva[feat_cols].values, iva["y"].values

    def objective(t):
        hp = dict(n_estimators=t.suggest_int("n_estimators", 200, 1200, step=100),
                  max_depth=t.suggest_int("max_depth", 3, 10),
                  learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                  subsample=t.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                  min_child_weight=t.suggest_int("min_child_weight", 1, 10))
        m = XGBRegressor(**hp, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
        m.fit(Xa, ya, verbose=False)
        return float(np.mean(np.abs(m.predict(Xb) - yb)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=min(15, trials)))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_params
    print(f"[castelle] best HP: {best}")

    model = XGBRegressor(**best, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
    model.fit(train[feat_cols].values, train["y"].values, verbose=False)
    out = test[["unique_id", "ds", "month"]].copy()
    out["y_pred"] = model.predict(test[feat_cols].values).clip(0, None)
    out["y_true"] = test["y"].values
    cap = pretest.groupby("unique_id")["y"].quantile(0.9).clip(lower=1).to_dict()
    return out, cap, feat_cols


def load_nf_traj(vr: Path, proto: str, model: str, h_days: int) -> pd.DataFrame:
    """Rolling-origin trajectory of a NF model (tft/lstm): each row is a target
    predicted from an earlier origin (issue_date), lead = target - origin."""
    f = vr / proto / f"per_row_predictions_{model}_{h_days}d.csv"
    d = pd.read_csv(f)
    d["ds"] = pd.to_datetime(d["ds"])
    if "is_padded" in d.columns:
        d = d[~d["is_padded"].fillna(False)]
    d["hour"] = d["ds"].dt.hour
    d = d[d["hour"].between(8, 20)].copy()
    d["lead_h"] = (d["ds"] - pd.to_datetime(d["issue_date"])).dt.total_seconds() / 3600
    return d[["unique_id", "ds", "y_true", "lead_h"]].assign(**{f"pred_{model.upper()}": d["y_pred"].values})


def load_cv_traj(cv_dir: Path, model: str, h_days: int) -> pd.DataFrame:
    """In-distribution single-origin 15-day trajectory from the all_data campaign's
    cv_predictions (models trained on data up to the test-window start). lead is
    measured in hours from the window start (the forecast issuance)."""
    f = cv_dir / f"{model}_{h_days}d" / "cv_predictions.csv"
    d = pd.read_csv(f)
    d["ds"] = pd.to_datetime(d["ds_real"])
    d["hour"] = d["ds"].dt.hour
    d = d[d["hour"].between(8, 20)].copy()
    origin = d["ds"].min().normalize()
    d["lead_h"] = (d["ds"] - origin).dt.total_seconds() / 3600
    return d[["unique_id", "ds", "y_true", "lead_h"]].assign(**{f"pred_{model.upper()}": d["y_pred"].values})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--validated-run", required=True, type=Path)
    ap.add_argument("--proto", default="cache2022",
                    help="protocol whose per_row trajectories to compare (cache2022 has summer coverage)")
    ap.add_argument("--cv-dir", type=Path, default=None,
                    help="in-distribution mode: alldata campaign dir; use its cv_predictions "
                         "(models trained up to the test-window start) instead of --proto per_row")
    ap.add_argument("--h-days", type=int, default=15, help="TFT trajectory horizon to compare against")
    ap.add_argument("--test-start", default="2025-06-01")
    ap.add_argument("--test-end", default="2026-02-28")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default="new_training_pipeline/castelle_xgb", type=Path)
    args = ap.parse_args()

    panel = load_panel(args.panel)
    era = "all" if args.cv_dir else ("2022" if args.proto == "cache2022" else "all")
    cx, cap, feat = train_castelle_xgb(panel, args.test_start, args.test_end, args.trials, train_era=era)
    args.out.mkdir(parents=True, exist_ok=True)
    cx.to_csv(args.out / "castelle_xgb_predictions.csv", index=False)

    # Castelle-XGB standalone accuracy (full trajectory = every test hour)
    print("\n" + "=" * 64 + "\nCASTELLE-STYLE XGB (one model, full trajectory)\n" + "=" * 64)
    for lbl, mset in [("all", None), ("season", SEASON_MONTHS), ("summer", SUMMER_MONTHS)]:
        sub = cx if mset is None else cx[cx["month"].isin(mset)]
        print(f"  relMAE_{lbl:<7} (per-series P90): {per_series_relmae(sub, cap):.4f}")

    # 3-way rolling-origin simulation: TFT vs LSTM vs Castelle-XGB, per lead,
    # overall + summer, on the IDENTICAL (series, target) rows.
    if args.cv_dir:
        tft = load_cv_traj(args.cv_dir, "tft", args.h_days)
        lstm = load_cv_traj(args.cv_dir, "lstm", args.h_days)
        args.proto = "in-dist (alldata cv)"
    else:
        tft = load_nf_traj(args.validated_run, args.proto, "tft", args.h_days)
        lstm = load_nf_traj(args.validated_run, args.proto, "lstm", args.h_days)
    base = tft.merge(lstm[["unique_id", "ds", "pred_LSTM"]], on=["unique_id", "ds"], how="inner")
    base = base.merge(cx[["unique_id", "ds", "y_pred", "month"]].rename(columns={"y_pred": "pred_CASTELLE"}),
                      on=["unique_id", "ds"], how="inner")
    if base.empty:
        print("\n[warn] no overlapping rows — check proto/panel/window")
        return
    base["lead_d"] = (base["lead_h"] / 24).round().clip(0, args.h_days).astype(int)
    MODELS = {"TFT": "pred_TFT", "LSTM": "pred_LSTM", "Castelle": "pred_CASTELLE"}
    buckets = [(0, 1), (1, 3), (3, 7), (7, args.h_days + 1)]

    def score(g: pd.DataFrame, col: str, kind: str) -> float:
        if kind == "mae":                       # raw MAE in users — no denominator
            return float((g[col] - g["y_true"]).abs().mean())
        rel = []                                # per-series P90-normalised relMAE
        for uid, gg in g.groupby("unique_id"):
            c = cap.get(uid)
            if c and c > 0:
                rel.append(float((gg[col] - gg["y_true"]).abs().mean()) / c)
        return float(np.mean(rel)) if rel else float("nan")

    rows = []
    for season, sub in [("overall", base), ("summer", base[base["month"].isin(SUMMER_MONTHS)])]:
        if sub.empty:
            continue
        for kind, label, dec in [("p90", "relMAE per-series P90", 4), ("mae", "MAE raw (users)", 2)]:
            print("\n" + "=" * 78)
            print(f"SIMULATION — {season.upper()} — {label} by forecast lead (proto={args.proto})")
            print("=" * 78)
            print(f"  {'lead (d)':<9}{'n':>7}{'TFT':>11}{'LSTM':>11}{'Castelle':>12}   winner")
            for lo, hi in [(0, args.h_days + 1)] + buckets:
                g = sub[(sub["lead_d"] >= lo) & (sub["lead_d"] < hi)]
                if g.empty:
                    continue
                vals = {k: score(g, c, kind) for k, c in MODELS.items()}
                win = min(vals, key=lambda k: (vals[k] if vals[k] == vals[k] else 9e9))
                rng = "all" if (lo, hi) == (0, args.h_days + 1) else (
                    f"{lo}-{hi-1}" if hi <= args.h_days else f"{lo}-{args.h_days}")
                print(f"  {rng:<9}{len(g):>7}{vals['TFT']:>11.{dec}f}{vals['LSTM']:>11.{dec}f}"
                      f"{vals['Castelle']:>12.{dec}f}   {win}")
                rows.append({"season": season, "metric": kind, "lead_d": rng, "n_rows": len(g),
                             **{k: round(v, 4) for k, v in vals.items()}, "winner": win})
    pd.DataFrame(rows).to_csv(args.out / "simulation_tft_lstm_castelle.csv", index=False)
    print(f"\n[done] {args.out}/ (castelle_xgb_predictions.csv, simulation_tft_lstm_castelle.csv)")


if __name__ == "__main__":
    main()
