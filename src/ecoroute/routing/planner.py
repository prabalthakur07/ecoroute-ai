"""Multi-objective routing on a road graph.

Edge exposure = PM2.5 (ug/m3) x time spent on edge (h) x breathing-rate factor.
Scalarised cost: (1 - lam) * time/T0 + lam * exposure/E0, with T0/E0 normalising each term.
Sweeping lam in [0, 1] traces an approximate Pareto front (supported solutions only).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ecoroute.data.synthetic import SPEED_KMH, rush_intensity

CONGESTION_LOSS = {"arterial": 0.45, "local": 0.15}  # max speed loss at peak rush
BREATHING_FACTOR = 1.0  # placeholder: scale by activity (cycling/riding ~2-3x resting)


def edge_time_s(e: dict, hour: float) -> float:
    speed = SPEED_KMH[e["road_class"]] * (
        1 - CONGESTION_LOSS[e["road_class"]] * rush_intensity(hour)
    )
    return e["length_m"] / (speed / 3.6)


def edge_exposure(e: dict, pm: dict[int, float], u: int, v: int, hour: float) -> float:
    pm_edge = 0.5 * (pm[u] + pm[v])
    return pm_edge * edge_time_s(e, hour) / 3600.0 * BREATHING_FACTOR


@dataclass(frozen=True)
class RouteStats:
    path: tuple[int, ...]
    time_min: float
    exposure: float  # ug.h/m3, measured against whatever pm field you pass in


class Planner:
    def __init__(self, G: nx.Graph, hour: float):
        self.G, self.hour = G, hour
        self._t0 = float(np.mean([edge_time_s(e, hour) for *_, e in G.edges(data=True)]))
        self._e0_cache: dict[int, float] = {}

    def _e0(self, pm: dict[int, float]) -> float:
        key = id(pm)
        if key not in self._e0_cache:
            self._e0_cache[key] = float(
                np.mean(
                    [edge_exposure(e, pm, u, v, self.hour) for u, v, e in self.G.edges(data=True)]
                )
            )
        return self._e0_cache[key]

    def plan(self, src: int, dst: int, lam: float, pm_plan: dict[int, float]) -> tuple[int, ...]:
        e0 = self._e0(pm_plan)

        def weight(u: int, v: int, e: dict) -> float:
            return (1 - lam) * edge_time_s(e, self.hour) / self._t0 + lam * edge_exposure(
                e, pm_plan, u, v, self.hour
            ) / e0

        return tuple(nx.shortest_path(self.G, src, dst, weight=weight))

    def evaluate(self, path: tuple[int, ...], pm_eval: dict[int, float]) -> RouteStats:
        t = x = 0.0
        for u, v in zip(path[:-1], path[1:], strict=True):
            e = self.G[u][v]
            t += edge_time_s(e, self.hour)
            x += edge_exposure(e, pm_eval, u, v, self.hour)
        return RouteStats(path, t / 60.0, x)
