# Walk-forward (in-distribution) result — 13-bucket frame, backup panel
Killed at 15d (MPS too slow: hidden=256 tft_15d stuck 3h/fold). 3d + 10d COMPLETE.
Source: identical_rows_{h}.csv (matched TFT/LSTM/XGB rows, 4 expanding folds, 2025-06-01..2026-02-28, 13 beaches).
per_row_predictions_* are per-fold-OVERWRITTEN (winter only) -- do NOT use them; use identical_rows.

## Per-series P90 relMAE (%), season (Apr-Sep) / summer (Jun-Aug), 13 beaches
| h  | TFT season | XGB season | LSTM season | TFT summer | XGB summer | LSTM summer |
|----|-----------|-----------|------------|-----------|-----------|------------|
| 3d | 14.6      | 22.1      | 26.3       | 14.5      | 23.2      | 28.3       |
| 10d| 19.2      | 23.8      | 26.9       | 19.6      | 24.7      | 27.8       |

## Diebold-Mariano (season, per-series HAC + HLN), TFT vs baseline
| h  | TFT vs XGB        | TFT vs LSTM        |
|----|-------------------|--------------------|
| 3d | DM -8.48, p=2.5e-17 | DM -17.39, p=1.4e-66 |
| 10d| DM -4.16, p=3.3e-05 | DM -32.68, p<1e-229  |

CONCLUSION: TFT WINS in-distribution at 3d & 10d, season & summer, DM-significant over BOTH baselines.
The old memory claim "XGB >= TFT in-distribution" (12-bucket, Jun-25) does NOT reproduce on the current 13-bucket frame.
Thesis "TFT strongest" narrative holds cross-year (OOD) AND in-distribution. 15d NOT run (would need efficient re-run).
