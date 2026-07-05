#!/usr/bin/env python3
"""Hyperparameter-search methodology artefacts from the Optuna studies.

The thesis HP search is persisted as Optuna SQLite studies named
`<campaign>__<model>__<Nd>` (e.g. cy_capped_backup_20260623_115630__tft__15d).
Several attempts exist with different ranges/trial budgets:
  - jun/18  cy/alldata_capped_20260618*   : 70 trials  (earlier, two-pass space)
  - jun/23  cy_capped_backup_20260623*     : 150 trials (final unified full-union)
  - jun/24  alldata_capped_backup_20260624 : 150 trials (all-data)

This reads any cross_year.db, catalogs its studies, and for a chosen campaign
produces the methodology tables/figures: best config per (model, horizon),
fANOVA parameter importance, search convergence, and the param-vs-objective
spread — so the methods chapter can show WHY each config was picked and how the
search behaved.

Usage:
  python new_training_pipeline/hp_methodology.py \
    --db new_training_pipeline_server_20260626_FINAL/optuna/cross_year.db \
    --campaign cy_capped_backup_20260623_115630 \
    --out new_training_pipeline/hp_methodology
  # omit --campaign to just print the catalog of every db passed.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def _storage(db: Path) -> str:
    return f"sqlite:///{db}"


def catalog(db: Path) -> pd.DataFrame:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    rows = []
    for s in optuna.get_all_study_summaries(storage=_storage(db)):
        parts = s.study_name.rsplit("__", 2)
        camp = parts[0] if len(parts) == 3 else s.study_name
        model = parts[1] if len(parts) == 3 else ""
        hz = parts[2] if len(parts) == 3 else ""
        label = db.parent.parent.name or db.name      # distinguish extraction folders
        rows.append({"db": label, "campaign": camp, "model": model, "horizon": hz,
                     "study": s.study_name, "n_trials": s.n_trials,
                     "best_value": (s.best_trial.value if s.best_trial else None)})
    return pd.DataFrame(rows)


def analyze_campaign(db: Path, campaign: str, out: Path) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = _storage(db)
    names = [s.study_name for s in optuna.get_all_study_summaries(storage=storage)
             if s.study_name.startswith(campaign + "__")]
    if not names:
        print(f"[hp] no studies for campaign '{campaign}' in {db.name}")
        return
    out.mkdir(parents=True, exist_ok=True)
    best_rows, imp_rows, trial_frames = [], [], []
    for name in sorted(names):
        _, model, hz = name.rsplit("__", 2)
        study = optuna.load_study(study_name=name, storage=storage)
        tdf = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
        tdf.insert(0, "model", model); tdf.insert(1, "horizon", hz)
        trial_frames.append(tdf)
        bt = study.best_trial
        best_rows.append({"model": model, "horizon": hz, "n_trials": len(study.trials),
                          "best_value": bt.value, **bt.params})
        try:
            for p, v in optuna.importance.get_param_importances(study).items():
                imp_rows.append({"model": model, "horizon": hz, "param": p, "importance": round(v, 4)})
        except Exception as e:
            print(f"  [warn] fANOVA failed for {name}: {e}")
    all_trials = pd.concat(trial_frames, ignore_index=True)
    all_trials.to_csv(out / "all_trials.csv", index=False)
    best = pd.DataFrame(best_rows).sort_values(["horizon", "model"])
    best.to_csv(out / "best_params.csv", index=False)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(out / "param_importance_fanova.csv", index=False)

    print(f"\n[hp] campaign {campaign}  ({len(names)} studies, {len(all_trials)} trials)")
    print("\n=== best config per (model, horizon) ===")
    show = [c for c in ["model", "horizon", "n_trials", "best_value", "hidden_size",
                        "input_size", "lr", "dropout", "batch_size"] if c in best.columns]
    print(best[show].to_string(index=False))
    if not imp.empty:
        print("\n=== top-3 fANOVA param importance per (model, horizon) ===")
        for (m, h), g in imp.groupby(["model", "horizon"]):
            top = g.nlargest(3, "importance")
            print(f"  {m:5s} {h:>3s}: " + ", ".join(f"{r.param}={r.importance:.2f}" for r in top.itertuples()))
    _figures(all_trials, imp, out)
    print(f"\n[hp] -> {out}/  (all_trials.csv, best_params.csv, param_importance_fanova.csv, *.png)")


def _figures(all_trials: pd.DataFrame, imp: pd.DataFrame, out: Path) -> None:
    """Full-width (figure*) figures at the thesis 10pt size — include with
    width=\\textwidth (scale 1.0)."""
    try:
        import matplotlib.pyplot as plt
        from thesis_figure_style import use_thesis_style, text_fig
        use_thesis_style()                                  # 10pt, IEEEtran serif, dpi 300
    except Exception as e:
        print(f"  [warn] figure style unavailable ({e}) -> skipping figures")
        return
    # Convergence: running-min objective vs trial, per model (15d).
    sub = all_trials[(all_trials["horizon"] == "15d") & all_trials["value"].notna()]
    if len(sub):
        fig, ax = text_fig(height=2.8)
        for m, g in sub.groupby("model"):
            g = g.sort_values("number")
            ax.plot(g["number"], g["value"].cummin(), label=m.upper())
        ax.set_xlabel("trial"); ax.set_ylabel("best inner relMAE")
        ax.set_title("HP search convergence (15 d)"); ax.legend()
        fig.savefig(out / "hp_convergence_15d.png"); plt.close(fig)
    # fANOVA importance — ONE panel per model (params differ across families, so a
    # shared axis is unreadable). Stacked full-width, each sorted by importance.
    if not imp.empty:
        from thesis_figure_style import TEXT_W
        i15 = imp[imp["horizon"] == "15d"]
        models = [m for m in ["tft", "lstm", "xgb"] if m in set(i15["model"])]
        colors = {"tft": "#1f77b4", "lstm": "#ff7f0e", "xgb": "#2ca02c"}
        if models:
            fig, axes = plt.subplots(len(models), 1, figsize=(TEXT_W, 1.9 * len(models)), squeeze=False)
            for ax, m in zip(axes[:, 0], models):
                g = i15[i15["model"] == m].sort_values("importance").tail(10)
                ax.barh(g["param"], g["importance"], color=colors.get(m, "#444"))
                ax.set_title(f"{m.upper()}", loc="left", fontweight="bold")
                ax.set_xlabel("fANOVA importance")
                ax.margins(x=0.02)
            fig.savefig(out / "hp_param_importance_15d.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path, nargs="+", help="one or more cross_year.db")
    ap.add_argument("--campaign", default=None, help="campaign prefix to analyze (omit = catalog only)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data/hp_methodology")
    args = ap.parse_args()

    cats = [catalog(db) for db in args.db if db.exists()]
    cat = pd.concat(cats, ignore_index=True) if cats else pd.DataFrame()
    if not cat.empty:
        args.out.mkdir(parents=True, exist_ok=True)
        cat.to_csv(args.out / "study_catalog.csv", index=False)
        print("=== study catalog (campaigns x trial budgets) ===")
        summ = (cat.groupby(["db", "campaign"])
                .agg(studies=("study", "nunique"), trials=("n_trials", "sum"),
                     trials_each=("n_trials", "max")).reset_index())
        print(summ.to_string(index=False))

    if args.campaign:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        target = next((db for db in args.db if db.exists() and any(
            s.study_name.startswith(args.campaign + "__")
            for s in optuna.get_all_study_summaries(storage=_storage(db)))), None)
        if target:
            analyze_campaign(target, args.campaign, args.out)
        else:
            print(f"[hp] campaign '{args.campaign}' not found in any provided db")


if __name__ == "__main__":
    main()
