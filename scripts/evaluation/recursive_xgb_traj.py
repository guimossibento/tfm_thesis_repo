#!/usr/bin/env python3
"""Recursive XGB — a FAIR trajectory baseline that predicts the full ~180-hour
hourly curve (not 3 isolated points like the direct h-step XGB).

ONE 1-step model: predict the next daytime hour y[t+1] from the current occupancy
(autoregressive lag + rolling means), the target hour's known weather/calendar,
and the per-beach static level. To forecast 15 days it iterates hour-by-hour from
a forecast origin, feeding each prediction back as the next lag — so it produces
all daytime hours ahead with a single model and uses recent occupancy (like the
TFT), unlike the lag-free Castelle regression.

To stay apples-to-apples, it reuses the SAME forecast origins as the NF models
(the per_row `issue_date`) and is matched on (beach, ds, issue_date); the lead is
ds - issue_date. It sees only the real past at each origin (no leakage).

Usage:
  python new_training_pipeline/recursive_xgb_traj.py \
    --panel beachcamweb/apps/prediction/pipeline_workspace/clean_dataset_backup/all_clean.csv \
    --validated-run new_training_pipeline_server_20260626_FINAL/validated_run --proto cache2022 \
    --trials 40 --out new_training_pipeline/recursive_xgb
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CASTELLE_WEATHER = ["om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
                    "om_wind_direction_10m", "om_shortwave_radiation"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]
SUMMER_MONTHS = {6, 7, 8}
SEED = 42


def _slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return re.sub(r"[-_]\d+$", "", s)


def load_panel(path: Path, cap_k: float = 1.5) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()
    df["unique_id"] = df["unique_id"].map(_slug)
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = df[df["hour"].between(8, 20)].copy()
    day = df[df["y"] > 0]
    cap = (cap_k * day.groupby("unique_id")["y"].quantile(0.9)).to_dict()
    capv = df["unique_id"].map(cap)
    m = capv.notna() & (df["y"] > capv)
    df.loc[m, "y"] = capv[m]
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def train_one_step(panel: pd.DataFrame, test_start: str, train_era: str, trials: int):
    import optuna
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    wcn = [c for c in CASTELLE_WEATHER if c in panel.columns] + CALENDAR

    d = panel.copy()
    pretest = d[d["ds"] < pd.Timestamp(test_start)]
    bp90 = pretest.groupby("unique_id")["y"].quantile(0.9)
    bmean = pretest.groupby("unique_id")["y"].mean()
    d["beach_p90"] = d["unique_id"].map(bp90)
    d["beach_mean"] = d["unique_id"].map(bmean)
    g = d.groupby("unique_id")
    d["y_lag1"] = d["y"]                                            # current occupancy
    d["y_roll12"] = g["y"].transform(lambda s: s.rolling(12, min_periods=1).mean())
    d["y_roll36"] = g["y"].transform(lambda s: s.rolling(36, min_periods=1).mean())
    nxt = g[wcn + ["y"]].shift(-1)                                  # next daytime hour
    for c in wcn:
        d[c + "_n"] = nxt[c]
    d["y_target"] = nxt["y"]
    feat = ["y_lag1", "y_roll12", "y_roll36", "beach_p90", "beach_mean"] + [c + "_n" for c in wcn]
    d = d.dropna(subset=feat + ["y_target"])
    train = d[d["ds"].dt.year == 2022] if train_era == "2022" else d[d["ds"] < pd.Timestamp(test_start)]
    print(f"[recursive] train_era={train_era}  1-step train rows={len(train):,}  features={len(feat)}")

    cut = train["ds"].quantile(0.82)
    a, b = train[train["ds"] < cut], train[train["ds"] >= cut]
    Xa, ya, Xb, yb = a[feat].values, a["y_target"].values, b[feat].values, b["y_target"].values

    def obj(t):
        hp = dict(n_estimators=t.suggest_int("n_estimators", 200, 1000, step=100),
                  max_depth=t.suggest_int("max_depth", 3, 10),
                  learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                  subsample=t.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                  min_child_weight=t.suggest_int("min_child_weight", 1, 10))
        m = XGBRegressor(**hp, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
        m.fit(Xa, ya, verbose=False)
        return float(np.mean(np.abs(m.predict(Xb) - yb)))

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=min(12, trials)))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    model = XGBRegressor(**st.best_params, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
    model.fit(train[feat].values, train["y_target"].values, verbose=False)
    cap = pretest.groupby("unique_id")["y"].quantile(0.9).clip(lower=1).to_dict()
    return model, wcn, dict(bp90=bp90.to_dict(), bmean=bmean.to_dict()), cap


def recursive_predict(panel: pd.DataFrame, model, wcn, stat, grid: pd.DataFrame) -> pd.DataFrame:
    """Iterate the 1-step model forward from each (beach, issue_date) origin in
    `grid` (the NF models' origins), recording predictions at the grid's targets."""
    recs = []
    by_uid = {uid: g.reset_index(drop=True) for uid, g in panel.groupby("unique_id")}
    npred = 0
    for (uid, issue), gg in grid.groupby(["unique_id", "issue_date"]):
        g = by_uid.get(uid)
        if g is None:
            continue
        ds = g["ds"].to_numpy()
        y = g["y"].to_numpy(dtype=float)
        W = g[wcn].to_numpy(dtype=float)
        bp90, bmean = stat["bp90"].get(uid, np.nan), stat["bmean"].get(uid, np.nan)
        if bp90 != bp90:
            continue
        tgt = pd.to_datetime(gg["ds"])
        first, last = np.datetime64(tgt.min()), np.datetime64(tgt.max())
        targets = set(tgt.to_numpy())
        p = int(np.searchsorted(ds, first, side="left")) - 1        # last real row before first target
        if p < 36 or p + 1 >= len(y):
            continue
        buf = list(y[p - 36:p + 1])
        cur = y[p]
        k = 0
        while p + k + 1 < len(y) and ds[p + k] < last:
            k += 1
            ti = p + k
            row = [cur, float(np.mean(buf[-12:])), float(np.mean(buf[-36:])), bp90, bmean] + list(W[ti])
            yp = max(float(model.predict(np.asarray(row, dtype=float)[None, :])[0]), 0.0)
            npred += 1
            if ds[ti] in targets:
                recs.append((uid, pd.Timestamp(issue), pd.Timestamp(ds[ti]), y[ti], yp))
            buf.append(yp)
            cur = yp
    print(f"[recursive] {npred:,} 1-step iterations -> {len(recs):,} matched target rows")
    return pd.DataFrame(recs, columns=["unique_id", "issue_date", "ds", "y_true", "pred_RECXGB"])


def load_nf(vr: Path, proto: str, model: str, h: int) -> pd.DataFrame:
    d = pd.read_csv(vr / proto / f"per_row_predictions_{model}_{h}d.csv")
    d["ds"] = pd.to_datetime(d["ds"])
    d["issue_date"] = pd.to_datetime(d["issue_date"])
    if "is_padded" in d.columns:
        d = d[~d["is_padded"].fillna(False)]
    d = d[d["ds"].dt.hour.between(8, 20)]
    keep = {"y_pred": f"pred_{model.upper()}"}
    cols = ["unique_id", "ds", "issue_date", "y_pred"] + (["y_true"] if model == "tft" else [])
    return d[cols].rename(columns=keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--validated-run", required=True, type=Path)
    ap.add_argument("--proto", default="cache2022")
    ap.add_argument("--h-days", type=int, default=15)
    ap.add_argument("--test-start", default="2025-06-01")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="new_training_pipeline/recursive_xgb", type=Path)
    args = ap.parse_args()

    panel = load_panel(args.panel)
    era = "2022" if args.proto == "cache2022" else "all"
    model, wcn, stat, cap = train_one_step(panel, args.test_start, era, args.trials)

    tft = load_nf(args.validated_run, args.proto, "tft", args.h_days)
    lstm = load_nf(args.validated_run, args.proto, "lstm", args.h_days)
    grid = tft[["unique_id", "issue_date", "ds"]].drop_duplicates()
    rec = recursive_predict(panel, model, wcn, stat, grid)
    args.out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(args.out / "recursive_xgb_predictions.csv", index=False)

    j = (tft.merge(lstm, on=["unique_id", "ds", "issue_date"], how="inner")
            .merge(rec[["unique_id", "ds", "issue_date", "pred_RECXGB"]],
                   on=["unique_id", "ds", "issue_date"], how="inner"))
    if j.empty:
        print("[warn] no overlap with TFT/LSTM trajectories"); return
    j["lead_d"] = (j["ds"] - j["issue_date"]).dt.days.clip(lower=0)
    j["month"] = j["ds"].dt.month
    print(f"[recursive] matched {len(j):,} rows over {j['unique_id'].nunique()} beaches, lead 0-{int(j['lead_d'].max())}d")

    MODELS = {"TFT": "pred_TFT", "LSTM": "pred_LSTM", "RecXGB": "pred_RECXGB"}
    buckets = [(0, 1), (1, 3), (3, 7), (7, args.h_days + 1)]

    def score(g, col, kind):
        if kind == "mae":
            return float((g[col] - g["y_true"]).abs().mean())
        rel = [float((gg[col] - gg["y_true"]).abs().mean()) / cap[uid]
               for uid, gg in g.groupby("unique_id") if cap.get(uid, 0) > 0]
        return float(np.mean(rel)) if rel else float("nan")

    rows = []
    for season, sub in [("overall", j), ("summer", j[j["month"].isin(SUMMER_MONTHS)])]:
        if sub.empty:
            continue
        for kind, lab, dec in [("p90", "relMAE per-series P90", 4), ("mae", "MAE raw (users)", 2)]:
            print("\n" + "=" * 72)
            print(f"180h TRAJECTORY — {season.upper()} — {lab} by lead (proto={args.proto})")
            print("=" * 72)
            print(f"  {'lead':<8}{'n':>7}{'TFT':>11}{'LSTM':>11}{'RecXGB':>11}   winner")
            for lo, hi in [(0, args.h_days + 1)] + buckets:
                g = sub[(sub["lead_d"] >= lo) & (sub["lead_d"] < hi)]
                if g.empty:
                    continue
                v = {k: score(g, c, kind) for k, c in MODELS.items()}
                win = min(v, key=lambda k: v[k] if v[k] == v[k] else 9e9)
                rng = "all" if (lo, hi) == (0, args.h_days + 1) else f"{lo}-{min(hi-1, args.h_days)}d"
                print(f"  {rng:<8}{len(g):>7}{v['TFT']:>11.{dec}f}{v['LSTM']:>11.{dec}f}{v['RecXGB']:>11.{dec}f}   {win}")
                rows.append({"season": season, "metric": kind, "lead": rng, "n": len(g),
                             **{k: round(v[k], 4) for k in MODELS}, "winner": win})
    pd.DataFrame(rows).to_csv(args.out / "recursive_comparison.csv", index=False)
    print(f"\n[done] {args.out}/  (recursive_xgb_predictions.csv, recursive_comparison.csv)")


if __name__ == "__main__":
    main()
