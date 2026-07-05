# AEMET / weather-source investigation

Backs the *Weather Source Selection* negative result (Data and Pipeline): AEMET
was evaluated and dropped in favour of Open-Meteo. These are the analyses behind
that decision (defense material — not part of the thesis PDF).

## Files

- `aemet_complete_analysis.ipynb` / `aemet_complete_analysis_preview.html` —
  AEMET Balearic analysis: variable availability, per-station availability,
  correlation analysis (temperatures at r≈0.99), geographic maps per island.
  43 stations, 191 records, 2022.
- `interpolation_explorer.ipynb` / `interpolation_explorer_STATIC.ipynb` /
  `interpolation_maps_preview.html` / `interpolation_explorer_preview.html` —
  the spatial interpolation built and tested to reach beach coordinates without a
  nearby station: **three methods — convex-hull/multipoint, IDW (power = 2), and
  KNN (k = 5/10)** — with per-coordinate and per-beach queries.

## Why AEMET was dropped

AEMET's station network is sparse relative to the 22 beaches, so reaching sites
without a nearby station needs spatial interpolation. All three methods above were
tested, but an interpolation accurate enough to trust would be a modelling project
in its own right, whereas Open-Meteo already provides per-beach interpolation out
of the box. So AEMET was dropped for effort-vs-marginal-gain, not impossibility.

## Raw data (not included here)

The ~1.7 GB raw AEMET 2022 station archive is **not** included in this repo (too
large); it is kept outside version control.
