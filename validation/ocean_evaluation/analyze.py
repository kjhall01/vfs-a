"""Summarize saved evaluations and plot diagnostics; no reference truth implied."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    runs = []
    for path in sorted(a.root.glob("**/summary.json")):
        d = json.loads(path.read_text())
        row = d.get("last", d.get("initial", {}))
        if not row:
            continue
        entry = dict(
            run=str(path.parent.relative_to(a.root)),
            case=d["config"]["case"],
            variant=d["variant"],
            cells=d["config"]["cells"],
            layers=d["config"]["layers"],
            dt=d["config"]["dt"],
            status=d["status"],
            days=d.get("elapsed_seconds", 0) / 86400,
            **{
                k: row[k]
                for k in (
                    "energy_error_J_kg",
                    "kinetic_J_kg",
                    "eddy_J_kg",
                    "max_speed_m_s",
                    "volume_relative_error",
                    "salt_relative_error",
                    "tracer_relative_error",
                    "min_h_m",
                    "min_eddy_J_kg",
                    "min_potential_N2_s2",
                    "nonzonal_ke_proxy_J_kg",
                    "max_closure_residual_J_kg",
                    "max_backscatter_power_residual_W_kg",
                )
            },
        )
        snapshot = path.parent / "last_accepted.npz"
        if snapshot.exists():
            with np.load(snapshot) as x:
                q = (x["tracer_h"][..., 0] / x["h"])[x["wet_cell"]]
                entry.update(min_tracer=float(q.min()), max_tracer=float(q.max()))
        if d.get("error"):
            entry["error"] = d["error"]
        runs.append(entry)
    (a.root / "metrics.json").write_text(json.dumps(runs, indent=2) + "\n")
    for case in ("channel", "forced", "rest"):
        paths = [
            p
            for p in a.root.glob("**/series.csv")
            if json.loads(p.with_name("summary.json").read_text())["config"]["case"]
            == case
        ]
        if not paths:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        fields = [
            ("kinetic_J_kg", "Resolved KE (J/kg)"),
            ("energy_error_J_kg", "H change minus external work (J/kg)"),
            ("eddy_J_kg", "Eddy reservoir (J/kg)"),
            ("max_speed_m_s", "Maximum edge speed (m/s)"),
        ]
        for path in sorted(paths):
            with path.open() as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            label = str(path.parent.relative_to(a.root))
            x = [float(r["time_days"]) for r in rows]
            for ax, (field, title) in zip(axes.flat, fields):
                ax.plot(x, [float(r[field]) for r in rows], label=label, linewidth=1.3)
                ax.set(xlabel="Model days", ylabel=title)
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"VFS {case}: idealized integration diagnostics")
        fig.savefig(a.root / f"{case}.png", dpi=160)
        plt.close(fig)
    for path in sorted(a.root.glob("**/last_accepted.npz")):
        status = json.loads(path.with_name("summary.json").read_text())
        if status["status"] != "completed" or status["variant"] != "backscatter":
            continue
        with np.load(path) as data:
            xyz, wet = data["cell_xyz"], data["wet_cell"]
            lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
            lat = np.degrees(np.arcsin(xyz[:, 2]))
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
            fields = [
                (data["eta"][:, 0], "Free surface (m)"),
                (data["CT"][:, 0], "Surface CT (°C)"),
                (
                    data["eddy_h"].sum(-1) / data["h"].sum(-1),
                    "Column eddy energy (J/kg)",
                ),
            ]
            for ax, (field, label) in zip(axes, fields):
                points = ax.scatter(
                    lon[wet], lat[wet], c=field[wet], s=22, cmap="viridis"
                )
                ax.set(xlabel="Longitude", ylabel="Latitude", title=label)
                fig.colorbar(points, ax=ax, shrink=0.8)
            name = str(path.parent.relative_to(a.root)).replace("/", "-")
            fig.suptitle(name + " — native cell samples")
            fig.savefig(a.root / (name + "-fields.png"), dpi=160)
            plt.close(fig)
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
