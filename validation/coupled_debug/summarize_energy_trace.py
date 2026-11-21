"""Summarize energy_stages.jsonl from a coupled aquaplanet trace.

Usage:
 python -I validation/coupled_debug/summarize_energy_trace.py RUN/energy_stages.jsonl

No PyAM21, model state, radiation tables or NumPy required.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def analyze(path):
    stages = defaultdict(lambda: [0, 0.0, 0.0, None])
    top = []
    accepted = []
    incomplete = None
    for line in Path(path).open(encoding="utf-8"):
        item = json.loads(line)
        if item["event"] == "begin":
            incomplete = item["interval_start_seconds"]
        elif item["event"] == "stage":
            name = item["stage"]
            residual = item["stage_numerical_residual_J_m2"]
            bucket = stages[name]
            bucket[0] += 1
            bucket[1] += residual
            bucket[2] += abs(residual)
            if bucket[3] is None or abs(residual) > abs(bucket[3]["residual"]):
                bucket[3] = {
                    "residual": residual,
                    "day": item["interval_start_seconds"] / 86400,
                    "ocean_edge": item.get("max_ocean_edge"),
                    "ocean_speed": item.get("max_ocean_speed_m_s"),
                }
            top.append((abs(residual), name, residual, item["interval_start_seconds"] / 86400,
                        item.get("max_ocean_edge"), item.get("max_ocean_speed_m_s")))
        elif item["event"] == "commit":
            incomplete = None
            accepted.append(item)
    print(f"Accepted intervals: {len(accepted)}")
    if accepted:
        last = accepted[-1]
        print(f"Last accepted day: {last['seconds'] / 86400:.6f}")
        print(f"Cumulative energy residual: {last['campaign_cumulative_numerical_residual_J_m2']:+.9g} J/m²")
    if incomplete is not None:
        print(f"INCOMPLETE interval beginning at day {incomplete / 86400:.6f}; see last stage for failure localization")
    print("\nStage                          Count     Signed Σ J/m²      Absolute Σ J/m²     Worst |Δ| J/m²")
    for name, (count, signed, absolute, worst) in sorted(
        stages.items(), key=lambda kv: -kv[1][2]
    ):
        print(f"{name:30s} {count:5d} {signed:+18.6g} {absolute:20.6g} {abs(worst['residual']):17.6g}")
    print("\nTen largest individual stage residuals (time = interval start):")
    for _, name, value, day, edge, speed in sorted(top, reverse=True)[:10]:
        print(f"  day {day:9.5f}  {name:30s}  {value:+15.7g} J/m²"
              f"  ocean_edge={edge} speed={speed}")
    return bool(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    analyze(args.jsonl)


if __name__ == "__main__":
    main()
