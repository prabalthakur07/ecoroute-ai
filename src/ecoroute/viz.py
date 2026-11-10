from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _field_map(ax, G, pm, title, stations=None, routes=None):
    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]
    sc = ax.scatter(xs, ys, c=[pm[n] for n in G.nodes], cmap="magma_r", s=60, marker="s")
    if stations:
        ax.scatter([G.nodes[s]["x"] for s in stations], [G.nodes[s]["y"] for s in stations],
                   facecolors="none", edgecolors="cyan", s=90, label="stations")
    for path, color, label in routes or []:
        ax.plot([G.nodes[n]["x"] for n in path], [G.nodes[n]["y"] for n in path],
                color=color, lw=2.5, label=label)
    ax.set_title(title)
    ax.set_aspect("equal")
    if stations or routes:
        ax.legend(loc="upper left", fontsize=7)
    return sc


def plot_experiment(exp, path: str, hour: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    routes = [(exp.example["fastest"], "tab:blue", "fastest"), (exp.example["eco"], "tab:green", "eco")]
    sc = _field_map(axes[0], exp.G, exp.pm_true, f"True PM2.5 @ {hour}:00 + routes", exp.stations, routes)
    _field_map(axes[1], exp.G, exp.pm_pred, "Predicted PM2.5 (GBM downscaler)")
    fig.colorbar(sc, ax=axes[:2], shrink=0.7, label="ug/m3")
    s = exp.sweep
    axes[2].plot(s["extra_time_%"], -s["exposure_change_%"], "o-")
    for _, r in s.iterrows():
        axes[2].annotate(f"lam={r['lam']}", (r["extra_time_%"], -r["exposure_change_%"]), fontsize=8)
    axes[2].set_xlabel("extra travel time vs fastest (%)")
    axes[2].set_ylabel("exposure reduction vs fastest (%)")
    axes[2].set_title("Time-vs-exposure trade-off")
    axes[2].grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
