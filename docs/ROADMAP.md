# Roadmap & ML learning path

Each milestone is a self-contained PR and teaches one skill.

| # | Milestone | What you build | ML skill |
|---|-----------|----------------|----------|
| 0 | **Synthetic MVP** (done) | Simulator, GBM downscaler, router, eval harness | Spatial CV, baselines, oracle bounds |
| 1 | **Real road graph** | `data/osm.py` using OSMnx for your city; replace the grid | Geospatial data, graph handling |
| 2 | **Real air quality** | `data/openaq.py` (CPCB/OpenAQ stations) + Open-Meteo/ERA5 weather | Data cleaning, missing data, leakage |
| 3 | **Better covariates** | Land use, NDVI, building density, Sentinel-5P NO2 | Feature engineering, SHAP |
| 4 | **Uncertainty** | Quantile GBM or conformal intervals; route on risk, not mean | Calibration, probabilistic forecasts |
| 5 | **Temporal forecasting** | 1-6 h ahead: lag features, then LSTM / graph neural net on the road graph | Time series, GNNs (PyTorch Geometric) |
| 6 | **Heat stress** | Add temperature / land-surface-temperature, heat index penalty | Multi-target modelling |
| 7 | **Multi-stop VRP** | OR-Tools with exposure-aware cost matrix | Combinatorial optimisation |
| 8 | **Dispatch timing** | Shift departure times, not just paths. Likely a bigger lever than path choice | Decision-focused learning |
| 9 | **Demo app** | Streamlit or FastAPI + MapLibre | Productionising |

## Key research question

On real data, how much exposure can path choice alone remove at <5% extra time? If the Pareto
front is flat, milestone 8 (timing) becomes the main contribution.

## Repo hygiene checklist

- [x] src layout, `pyproject.toml`, MIT license
- [x] CI (lint + tests + demo smoke run)
- [x] pre-commit with ruff
- [ ] Add repo badges once CI has run
- [ ] Tag `v0.1.0` and write a release note
- [ ] Add a notebook walkthrough under `notebooks/`
- [ ] Open issues from the milestone table with `good first issue` labels
