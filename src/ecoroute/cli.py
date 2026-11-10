from __future__ import annotations

import argparse
from pathlib import Path

from ecoroute.evaluate import run
from ecoroute.viz import plot_experiment


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ecoroute")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the synthetic-city end-to-end experiment")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--trips", type=int, default=300)
    d.add_argument("--hour", type=int, default=18)
    d.add_argument("--lam", type=float, default=0.5, help="exposure weight in [0,1]")
    d.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    exp = run(seed=args.seed, trips=args.trips, hour=args.hour, lam=args.lam)
    print("\n== Forecast: leave-one-station-out RMSE (ug/m3, lower is better) ==")
    print(exp.forecast_cv.groupby("model")["rmse"].agg(["mean", "std"]).round(2).to_string())
    print(f"\n== Routing summary (lam={args.lam}, scored on TRUE PM2.5, vs fastest route) ==")
    print(exp.summary.round(2).to_string(index=False))
    print("\n== Pareto sweep (planning with predicted PM2.5) ==")
    print(exp.sweep.round(2).to_string(index=False))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    plot_experiment(exp, str(out / "demo.png"), args.hour)
    print(f"\nSaved figure -> {out / 'demo.png'}")


if __name__ == "__main__":
    main()
