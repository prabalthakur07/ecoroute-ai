"""End-to-end experiment: forecast quality AND real routing benefit.

Key design choice: plan with *predicted* PM2.5, but score routes against the *true* field.
An oracle planner (plans with truth) gives the upper bound on what better forecasting could buy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ecoroute.data.synthetic import (
    build_city,
    make_station_table,
    pick_stations,
    true_pm25,
)
from ecoroute.forecast.downscale import PM25Downscaler, leave_one_station_out
from ecoroute.routing.planner import Planner

TRAIN_DAYS, TOTAL_DAYS = 11, 14


@dataclass
class Experiment:
    G: object
    stations: list[int]
    forecast_cv: pd.DataFrame
    pm_true: dict[int, float]
    pm_pred: dict[int, float]
    sweep: pd.DataFrame
    summary: pd.DataFrame
    example: dict = field(default_factory=dict)


def sample_trips(G, n: int, rng: np.random.Generator, min_hops: int = 10):
    nodes = list(G.nodes)
    trips = []
    while len(trips) < n:
        a, b = rng.choice(nodes, size=2, replace=False)
        (i1, j1), (i2, j2) = G.nodes[a]["ij"], G.nodes[b]["ij"]
        if abs(i1 - i2) + abs(j1 - j2) >= min_hops:
            trips.append((int(a), int(b)))
    return trips


def run(
    seed: int = 0,
    trips: int = 300,
    hour: int = 18,
    lam: float = 0.5,
    n_stations: int = 25,
    grid: int = 20,
) -> Experiment:
    rng = np.random.default_rng(seed)
    G = build_city(n=grid)
    stations = pick_stations(G, n_stations, rng)
    df = make_station_table(G, stations, days=TOTAL_DAYS, rng=rng)

    cv = leave_one_station_out(df, test_days_from=TRAIN_DAYS, seed=seed)

    test_day = TOTAL_DAYS - 1
    model = PM25Downscaler(seed).fit(df[df["day"] < TRAIN_DAYS])
    slot = df[(df["day"] == test_day) & (df["hour"] == hour)]
    pm_pred = model.predict_nodes(G, hour, regional=float(slot["pm25"].mean()))
    pm_true = true_pm25(G, hour, df.attrs["day_factors"][test_day])

    planner = Planner(G, hour)
    od = sample_trips(G, trips, rng)
    fastest = {t: planner.evaluate(planner.plan(*t, 0.0, pm_pred), pm_true) for t in od}

    def compare(lam_val: float, pm_plan) -> dict:
        dt, dx, changed = [], [], 0
        for t in od:
            r = planner.evaluate(planner.plan(*t, lam_val, pm_plan), pm_true)
            f = fastest[t]
            dt.append(100 * (r.time_min - f.time_min) / f.time_min)
            dx.append(100 * (r.exposure - f.exposure) / f.exposure)
            changed += r.path != f.path
        return {
            "extra_time_%": np.mean(dt),
            "exposure_change_%": np.mean(dx),
            "routes_changed_%": 100 * changed / len(od),
        }

    lams = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    sweep = pd.DataFrame([{"lam": v, **compare(v, pm_pred)} for v in lams])
    summary = pd.DataFrame(
        [
            {"planner": "eco (predicted PM2.5)", **compare(lam, pm_pred)},
            {"planner": "eco (oracle PM2.5)", **compare(lam, pm_true)},
        ]
    )
    t0 = od[0]
    example = {
        "fastest": fastest[t0].path,
        "eco": planner.plan(*t0, lam, pm_pred),
    }
    return Experiment(G, stations, cv, pm_true, pm_pred, sweep, summary, example)
