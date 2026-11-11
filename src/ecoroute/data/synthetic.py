"""Synthetic city with a known ground-truth PM2.5 field.

Why synthetic first? Real block-level PM2.5 has no ground truth, so you cannot tell whether a
model is good or merely plausible. Here the truth is known, which lets us measure (a) forecast
error and (b) the *real* exposure benefit of routing on predictions. Swap this module for an
OSMnx + OpenAQ loader once the pipeline is trusted (see docs/ROADMAP.md).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

ARTERIAL_EVERY = 5
SPEED_KMH = {"arterial": 40.0, "local": 25.0}


def _gauss(x: float, mu: float, sd: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / sd) ** 2))


def rush_intensity(hour: float) -> float:
    """0..1 congestion pressure with morning and evening peaks."""
    return max(_gauss(hour, 9, 1.5), _gauss(hour, 19, 2.0))


def build_city(n: int = 20, spacing_m: float = 250.0) -> nx.Graph:
    """An n x n grid; every ARTERIAL_EVERY-th row/column is a faster, busier arterial."""
    G = nx.grid_2d_graph(n, n)
    G = nx.convert_node_labels_to_integers(G, label_attribute="ij")
    extent = (n - 1) * spacing_m
    art_coords = np.arange(0, n, ARTERIAL_EVERY) * spacing_m
    for _node, d in G.nodes(data=True):
        i, j = d["ij"]
        x, y = i * spacing_m, j * spacing_m
        d["x"], d["y"] = x, y
        d["dist_arterial"] = float(min(np.abs(art_coords - x).min(), np.abs(art_coords - y).min()))
        d["dist_center"] = float(np.hypot(x - extent / 2, y - extent / 2))
        d["green"] = 0.5 + 0.5 * np.sin(x / 900.0) * np.cos(y / 700.0)  # NDVI-like proxy
    for u, v, e in G.edges(data=True):
        (i1, j1), (i2, j2) = G.nodes[u]["ij"], G.nodes[v]["ij"]
        on_arterial = (i1 == i2 and i1 % ARTERIAL_EVERY == 0) or (
            j1 == j2 and j1 % ARTERIAL_EVERY == 0
        )
        e["road_class"] = "arterial" if on_arterial else "local"
        e["length_m"] = spacing_m
    G.graph["extent"] = extent
    return G


def true_pm25(G: nx.Graph, hour: int, day_factor: float = 1.0) -> dict[int, float]:
    """Ground-truth PM2.5 (ug/m3) per node. Contains a hidden industrial plume that is NOT a
    model feature, so the downscaler has to infer it from sparse stations."""
    ext = G.graph["extent"]
    bx, by = 0.7 * ext, 0.3 * ext
    diurnal = 45 + 25 * (_gauss(hour, 9, 2) + _gauss(hour, 19, 2.5))
    out = {}
    for node, d in G.nodes(data=True):
        near_road = 1 + 0.6 * np.exp(-d["dist_arterial"] / 120.0)
        plume = 40 * np.exp(-((d["x"] - bx) ** 2 + (d["y"] - by) ** 2) / (2 * 600.0**2))
        out[node] = float((diurnal * near_road + plume - 15 * d["green"]) * day_factor)
    return out


def pick_stations(G: nx.Graph, k: int, rng: np.random.Generator) -> list[int]:
    return sorted(rng.choice(list(G.nodes), size=k, replace=False).tolist())


def make_station_table(
    G: nx.Graph,
    stations: list[int],
    days: int,
    rng: np.random.Generator,
    noise_sd: float = 4.0,
) -> pd.DataFrame:
    """Hourly noisy readings at sparse stations. `day_factor` mimics weather/regional shifts."""
    day_factors = rng.lognormal(mean=0.0, sigma=0.2, size=days)
    rows = []
    for day in range(days):
        for hour in range(24):
            truth = true_pm25(G, hour, day_factors[day])
            for s in stations:
                d = G.nodes[s]
                rows.append(
                    {
                        "station": s,
                        "day": day,
                        "hour": hour,
                        "x": d["x"],
                        "y": d["y"],
                        "dist_arterial": d["dist_arterial"],
                        "dist_center": d["dist_center"],
                        "green": d["green"],
                        "pm25": truth[s] + rng.normal(0, noise_sd),
                    }
                )
    df = pd.DataFrame(rows)
    df.attrs["day_factors"] = day_factors.tolist()
    return df
