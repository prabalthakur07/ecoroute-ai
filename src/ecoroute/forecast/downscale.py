"""Spatial downscaling: sparse station readings -> PM2.5 at every road-graph node.

Model: gradient boosting on geographic covariates + time-of-day + a real-time "regional level"
(mean of all stations in the same hour). Baselines: regional mean and inverse-distance weighting.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

GEO = ["x", "y", "dist_arterial", "dist_center", "green"]
FEATURES = [*GEO, "hour_sin", "hour_cos", "regional"]


def _time_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_regional(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of all stations in the same (day, hour) slot: what you'd observe live."""
    df = df.copy()
    df["regional"] = df.groupby(["day", "hour"])["pm25"].transform("mean")
    return df


class PM25Downscaler:
    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_depth=4, l2_regularization=1.0, random_state=seed
        )

    def fit(self, station_df: pd.DataFrame) -> PM25Downscaler:
        df = _time_feats(add_regional(station_df))
        self.model.fit(df[FEATURES], df["pm25"])
        return self

    def predict_nodes(self, G: nx.Graph, hour: int, regional: float) -> dict[int, float]:
        nodes = list(G.nodes)
        X = pd.DataFrame([{k: G.nodes[n][k] for k in GEO} for n in nodes])
        X["hour"] = hour
        X["regional"] = regional
        X = _time_feats(X)[FEATURES]
        return dict(zip(nodes, self.model.predict(X).tolist(), strict=True))


def idw_predict(train_xy: np.ndarray, train_val: np.ndarray, query_xy: np.ndarray, p: float = 2.0):
    d = np.linalg.norm(query_xy[:, None, :] - train_xy[None, :, :], axis=2) + 1e-6
    w = 1.0 / d**p
    return (w * train_val[None, :]).sum(1) / w.sum(1)


def leave_one_station_out(df: pd.DataFrame, test_days_from: int, seed: int = 0) -> pd.DataFrame:
    """Hold out each station entirely (spatial generalisation) and the last days (temporal)."""
    rows = []
    for s in sorted(df["station"].unique()):
        train = df[(df["station"] != s) & (df["day"] < test_days_from)]
        test = add_regional(df[df["station"] != s])  # regional uses only *other* stations
        test = test[["day", "hour", "regional"]].drop_duplicates()
        held = df[(df["station"] == s) & (df["day"] >= test_days_from)].merge(
            test, on=["day", "hour"]
        )
        model = PM25Downscaler(seed).fit(train)
        pred_gbm = model.model.predict(_time_feats(held)[FEATURES])
        pred_reg = held["regional"].to_numpy()
        pred_idw = np.empty(len(held))
        for (day, hour), idx in held.groupby(["day", "hour"]).groups.items():
            obs = df[(df["station"] != s) & (df["day"] == day) & (df["hour"] == hour)]
            pred_idw[held.index.get_indexer(idx)] = idw_predict(
                obs[["x", "y"]].to_numpy(), obs["pm25"].to_numpy(), held.loc[idx, ["x", "y"]].to_numpy()
            )
        y = held["pm25"].to_numpy()
        for name, p in [("gbm", pred_gbm), ("idw", pred_idw), ("regional_mean", pred_reg)]:
            rows.append({"station": s, "model": name, "rmse": float(np.sqrt(np.mean((y - p) ** 2)))})
    return pd.DataFrame(rows)
