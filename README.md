# TFM — Beach Occupancy Forecasting (thesis sources)

Manuscript, figure/evaluation scripts, and data for the master's thesis
**"Current and Future Predictions of Beach Occupancy"** (MUSI, Data Science, UIB).

This repository holds the **offline thesis artifacts** — everything used to produce the
paper and its numbers. The deployed production system (the BeachCamWeb Django app that
ingests webcam images, serves forecasts, and retrains on a schedule) is a **separate
repository** and is not included here.

## Structure

```
paper/                    LaTeX manuscript + figures
  document.tex            the manuscript
  titlepage.tex           cover / license front matter
  bibliography.bib
  figures/                the 9 figures used in the paper (.png / .pdf)

scripts/
  figures/                figure generators
    thesis_figure_style.py    shared IEEE 8 pt / \textwidth sizing
    make_camval_figure.py     camera-validation seasonal ratio (fig_camval_scatter)
    make_cohen_figure.py      per-beach Cohen's d (fig_cohen_scatter)
    make_vsn_importance.py    TFT variable-selection weights (fig_tft_vsn_importance)
    plot_tft_fanova.py        TFT fANOVA HP importance (fig_tft_fanova_importance)
    make_seasonal_grid.py     2x4 seasonal webcam grid (fig_seasonal_grid) — fetches from media server
  evaluation/             results / tables
    retrain_3models_validated.py   cross-year + walk-forward TFT/LSTM/XGB (matched rows, P90, DM)
    retrain_daytime.py             daytime headline (tab:daytime) + scenarios
    wf_aggregate.py                walk-forward season/summer relMAE + Diebold-Mariano
    fair_tables.py                 multi-scenario fair table (tab:a2)
    castelle_style_xgb.py          Castelle-style XGBoost replication (tab:c2)
    hp_methodology.py              HP tables + fANOVA (ovat / bestcounts / hp_perrun)
    recursive_xgb_traj.py          recursive multi-step XGBoost trajectory baseline
  notebooks/
    generate_train_data.ipynb          data pipeline — builds the panel, applies the daytime
                                        filter + seasonality screen, and emits the data/filter
                                        figures (solar, camval, cohen, day-type)
    early_baselines_exploration.ipynb  the initial model-behaviour study (Lasso/RF/GB/XGB/LSTM,
                                        source of tab:early_baselines)

data/
  dataset.csv.gz          unified hourly panel (26 series). read directly by pandas
  all_clean.csv.gz        curated walk-forward panel
  hp_methodology/         Optuna trial summaries + fANOVA importance CSVs
  results/                evaluation outputs per protocol:
    validated_daytime/            cross-year daytime (tab:daytime)
    validated_walkforward_820/    walk-forward 3d/10d
    validated_walkforward_820_15d/ walk-forward 15d
    validated_daytime_scenarios/  multi-scenario S1/S2/S3 (tab:a2)

optuna/                   Optuna study DBs — HP-search history (see optuna/README.md)
  cross_year_200trial.db      the Phase-3 search (1,052 completed TFT trials)
  cross_year_150trial_final.db, cross_year_70trial.db, cross_year_deployed.db
  tft_optuna_v7.db, tft_optuna_v12.db, lstm_optuna.db, optuna_summer.db, optuna_xgb_per_beach.db

analysis/
  aemet/                  weather-source investigation (AEMET drop): notebooks + rendered HTMLs
```

## Reproduce

All script paths are **repo-relative** (resolved from each file's location), so run
from anywhere after cloning. Panels are read straight from the `.gz` (no manual gunzip).

```sh
# figures  (read data/, write paper/figures/)
python scripts/figures/make_camval_figure.py     # camera-validation ratio
python scripts/figures/make_cohen_figure.py       # per-beach Cohen's d
python scripts/figures/make_vsn_importance.py      # TFT VSN weights
python scripts/figures/plot_tft_fanova.py          # TFT fANOVA importance
python scripts/figures/make_seasonal_grid.py       # 2x4 seasonal grid (needs network)

# tables / results  (CLI-driven; heavy, GPU for the TFT)
python scripts/evaluation/retrain_daytime.py       # cross-year daytime (defaults to data/)
python scripts/evaluation/fair_tables.py           # multi-scenario table
python scripts/evaluation/wf_aggregate.py \
    data/results/validated_walkforward_820/walkforward/identical_rows_10d.csv \
    data/results/validated_walkforward_820/walkforward/per_beach_tft_10d.csv "10d"

# paper
cd paper && latexmk -pdf document.tex
```

## Notes

- The **data/filter figures** (camval, cohen, solar, day-type) are the reproducible output
  of `notebooks/generate_train_data.ipynb`; the matching `make_*.py` are standalone twins.
- `results/` holds CSV / JSON only; model checkpoints and Lightning logs were left out
  (re-run the training scripts to regenerate). The Optuna `.db` search history **is** kept
  under `optuna/`; the raw ~1.7 GB AEMET station archive is **not** (too large,
  kept outside version control).
- Data covers 2022 + the 2025–2026 deployment era; counts are Bayesian VGG-19 estimates,
  not human ground truth.
# tfm_thesis_repo
