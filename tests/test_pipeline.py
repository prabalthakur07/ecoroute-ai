import numpy as np

from ecoroute.data.synthetic import build_city, make_station_table, pick_stations, true_pm25
from ecoroute.forecast.downscale import PM25Downscaler, leave_one_station_out
from ecoroute.routing.planner import Planner


def test_city_structure():
    G = build_city(n=10)
    assert G.number_of_nodes() == 100
    assert {e["road_class"] for *_, e in G.edges(data=True)} == {"arterial", "local"}


def test_truth_peaks_at_rush_hour():
    G = build_city(n=10)
    assert np.mean(list(true_pm25(G, 9).values())) > np.mean(list(true_pm25(G, 3).values()))


def test_downscaler_beats_regional_mean():
    rng = np.random.default_rng(1)
    G = build_city(n=12)
    df = make_station_table(G, pick_stations(G, 15, rng), days=8, rng=rng)
    cv = leave_one_station_out(df, test_days_from=6).groupby("model")["rmse"].mean()
    assert cv["gbm"] < cv["regional_mean"]


def test_lambda_zero_is_fastest_and_exposure_weight_helps_under_oracle():
    G = build_city(n=12)
    pm = true_pm25(G, 18)
    p = Planner(G, hour=18)
    src, dst = 0, G.number_of_nodes() - 1
    fast = p.evaluate(p.plan(src, dst, 0.0, pm), pm)
    eco = p.evaluate(p.plan(src, dst, 1.0, pm), pm)
    assert fast.time_min <= eco.time_min + 1e-9
    assert eco.exposure <= fast.exposure + 1e-9


def test_fit_predict_covers_all_nodes():
    rng = np.random.default_rng(2)
    G = build_city(n=8)
    df = make_station_table(G, pick_stations(G, 8, rng), days=3, rng=rng)
    pred = PM25Downscaler().fit(df).predict_nodes(G, hour=8, regional=float(df["pm25"].mean()))
    assert set(pred) == set(G.nodes)
