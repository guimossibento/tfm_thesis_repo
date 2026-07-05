# Optuna hyperparameter-search studies

SQLite databases holding the full Optuna trial history behind the
hyperparameter and feature-selection experiments (Methodology,
*Feature and Hyperparameter Selection*). Each study is named
`<campaign>__<model>__<Nd>` — model in {tft, lstm, xgb}, horizon in {3d, 10d, 15d}.

| File | Campaigns | Trials/cell | Role |
|---|---|---|---|
| `cross_year_200trial.db` | `cross_year_backup_20260601`, `cross_year_210526` | 200 | **Phase-3 "independent search run"** — the campaign the thesis describes. 1,052 completed TFT trials (527 + 525). Per-cell optima land at hidden ∈ [96,256], lr ∈ [3.6e-5, 5.3e-4]; the FIXED-HP triple (h64 / nh4 / lr≈1e-3) is never sampled jointly. Search space: lr ∈ [1e-5, 1e-2] log-uniform, hidden ∈ {32…512}, n_head ∈ {1,2,4,8,16,32}. |
| `cross_year_150trial_final.db` | `cy_capped_20260623`, `alldata_capped_20260624` | 150 | Final unified full-union search (fixed the earlier two-pass flaw); source of the fANOVA importances (scaler dominates LSTM, lr dominates XGB). |
| `cross_year_70trial.db` | `cy_capped_20260618`, `alldata_20260618` | 70 | Earlier two-pass attempt. |
| `cross_year_deployed.db` | beachcamweb | — | The deployed system's study store. |
| `tft_optuna_v7.db`, `tft_optuna_v12.db` | legacy TFT | 60 | Legacy per-model TFT searches (the "best-of-run" tallies, Table `tab:bestcounts`). |
| `lstm_optuna.db` | legacy LSTM | — | Legacy LSTM search. |
| `optuna_summer.db` | summer-emphasis | — | Summer-only / season-emphasis trials. |
| `optuna_xgb_per_beach.db` | per-beach XGB | — | Per-beach XGBoost search. |

Read with the Optuna API:

```python
import optuna
optuna.get_all_study_summaries("sqlite:///cross_year_200trial.db")   # list studies + trial counts
st = optuna.load_study(study_name="cross_year_backup_20260601_125143__tft__15d",
                       storage="sqlite:///cross_year_200trial.db")
```

The catalog + fANOVA tool is `../scripts/evaluation/hp_methodology.py`.

`.gitignore` normally excludes `*.db` (regenerable training artifacts); these are
un-ignored on purpose (`!optuna/*.db`) because they are the search history and are
not cheaply regenerable.
