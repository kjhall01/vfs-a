"""Compare wave/trade-wind campaigns; reconstructed winds are diagnostic only."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vfs.geometry.mesh import VoronoiMesh


def series(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    runs = []
    for path in sorted(a.root.glob("*/*/summary.json")):
        info = json.loads(path.read_text())
        if info["status"] != "completed":
            continue
        config = info["config"]
        data = np.load(path.with_name("last_accepted.npz"))
        initial = np.load(path.with_name("initial.npz"))
        mesh = VoronoiMesh(
            n_cells=config["cells"],
            radius=config["radius"] or 6371220.0,
            generator_xyz=data["cell_xyz"],
            lloyd_iters=0,
            rotation_scheme="compatible",
            verbose=False,
        )
        area = mesh.cell_area.numpy()
        # Recompute shape-safe elevation diagnostics from saved native states.
        # Early driver CSVs broadcast (cell,1) eta against (cell,) area. Preserve
        # those raw records; this analysis is authoritative for these metrics.
        history = series(path.with_name("series.csv"))
        eta0 = initial["eta"].reshape(-1)
        norm0 = np.sum(area * eta0**2)
        for j, day in enumerate(history["time_days"]):
            snap = path.with_name(
                "initial.npz" if day == 0 else f"state-{int(round(day*86400)):09d}s.npz"
            )
            if abs(day - info["last"]["time_days"]) < 1e-9:
                snap = path.with_name("last_accepted.npz")
            if not snap.exists():
                raise FileNotFoundError(
                    f"Need --snapshots to correct elevation diagnostics: {snap}"
                )
            with np.load(snap) as saved:
                eta = saved["eta"].reshape(-1)
                history["eta_rms_m"][j] = np.sqrt(np.sum(area * eta**2) / area.sum())
                history["initial_eta_projection"][j] = (
                    np.sum(area * eta * eta0) / norm0
                    if config["case"] == "wave"
                    else 0.0
                )
        xyz = mesh.cell_xyz
        east = torch.stack((-xyz[:, 1], xyz[:, 0], torch.zeros_like(xyz[:, 0])), -1)
        east /= torch.linalg.vector_norm(east, dim=-1, keepdim=True)
        north = torch.linalg.cross(xyz, east)
        velocity = mesh.cell_wind_vector(torch.from_numpy(data["u_e"]))
        u = (velocity * east[:, None]).sum(-1).numpy()
        v = (velocity * north[:, None]).sum(-1).numpy()
        lat = np.rad2deg(np.arcsin(data["cell_xyz"][:, 2]))
        bins = np.linspace(-90, 90, 19)
        index = np.clip(np.digitize(lat, bins) - 1, 0, 17)

        def zonal(value):
            w = np.bincount(index, weights=area, minlength=18)
            total = np.bincount(index, weights=area * value, minlength=18)
            return np.divide(total, w, out=np.full(18, np.nan), where=w > 0)

        h = data["h"]
        # Exact overlap with upper 100 m of each discrete layer, not a fixed
        # layer index; this is transport per unit zonal distance [m^2/s].
        top = h.cumsum(-1) - h
        overlap = np.maximum(0, np.minimum(h, 100 - top))
        label = (
            "GM–Redi"
            if info["variant"] == "gmredi"
            else (
                "+ backscatter"
                if config["biharmonic_viscosity"] == 0
                else "+ backscatter + sink"
            )
        )
        if config["case"] == "wave":
            label += f' (dt={config["dt"]:g}s)'
        runs.append(
            dict(
                name=str(path.parent.relative_to(a.root)),
                info=info,
                data=data,
                initial=initial,
                mesh=mesh,
                area=area,
                lat=lat,
                u=u,
                v=v,
                profiles=(zonal(u[:, 0]), zonal((overlap * v).sum(-1))),
                series=history,
                label=label,
            )
        )
    trades = [r for r in runs if r["info"]["config"]["case"] == "tradewinds"]
    waves = [r for r in runs if r["info"]["config"]["case"] == "wave"]
    if trades:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        latitude = np.linspace(-90, 90, 361)
        amplitude = trades[0]["info"]["config"]["wind"]
        stress = amplitude * (
            -np.exp(-((latitude / 20) ** 2))
            + 0.5 * np.exp(-(((abs(latitude) - 45) / 15) ** 2))
        )
        axs[0].plot(stress, latitude, color="black")
        for r in trades:
            for ax, value in zip(axs[1:], r["profiles"]):
                ax.plot(value, np.arange(-85, 90, 10), label=r["label"])
        for ax, title in zip(
            axs,
            [
                "Prescribed eastward stress (Pa)",
                "Surface eastward current (m/s)",
                "Upper-100 m northward transport (m²/s)",
            ],
        ):
            ax.set_xlabel(title)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_ylim(-90, 90)
            ax.grid(alpha=0.2)
        axs[0].set_ylabel("Latitude (degrees)")
        axs[1].legend(fontsize=8)
        days = sorted({r["info"]["last"]["time_days"] for r in trades})
        duration = f"{days[0]:g}-day" if len(days) == 1 else "Endpoint"
        fig.suptitle(f"{duration} trade-wind aquaplanet: reconstructed zonal means")
        fig.savefig(a.root / "trade-wind-profiles.png", dpi=170)
        plt.close(fig)
        fig, axs = plt.subplots(
            2,
            len(trades),
            figsize=(4 * len(trades), 6),
            layout="constrained",
            squeeze=False,
        )
        limits = [
            max(np.max(np.abs(r["u"][:, 0])) for r in trades),
            max(np.max(np.abs(r["data"]["eta"])) for r in trades),
        ]
        for j, r in enumerate(trades):
            lon = np.rad2deg(
                np.arctan2(r["data"]["cell_xyz"][:, 1], r["data"]["cell_xyz"][:, 0])
            )
            for i, values in enumerate([r["u"][:, 0], r["data"]["eta"]]):
                im = axs[i, j].scatter(
                    lon,
                    r["lat"],
                    c=values,
                    s=20,
                    cmap="RdBu_r",
                    vmin=-limits[i],
                    vmax=limits[i],
                )
                axs[i, j].set(
                    xlim=(-180, 180),
                    ylim=(-90, 90),
                    xlabel="Longitude",
                    ylabel="Latitude",
                )
                if i == 0:
                    axs[i, j].set_title(r["label"], fontsize=10)
                if j == len(trades) - 1:
                    fig.colorbar(
                        im,
                        ax=axs[i, :],
                        label=(
                            "Surface eastward current (m/s)"
                            if i == 0
                            else "Surface elevation (m)"
                        ),
                    )
        fig.savefig(a.root / "trade-wind-fields.png", dpi=170)
        plt.close(fig)
    if waves:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for r in waves:
            t = r["series"]
            for ax, key in zip(
                axs, ["eta_rms_m", "initial_eta_projection", "energy_error_J_kg"]
            ):
                ax.plot(t["time_days"], t[key], label=r["label"], lw=1.4)
        for ax, title in zip(
            axs,
            [
                "Surface elevation RMS (m)",
                "Projection onto initial surface bump",
                "Full-H energy error (J/kg)",
            ],
        ):
            ax.set(xlabel="Model days", ylabel=title)
            ax.grid(alpha=0.2)
        axs[0].legend(fontsize=8)
        fig.suptitle("Unforced rotating free-surface wave")
        fig.savefig(a.root / "wave-evolution.png", dpi=170)
        plt.close(fig)
    # Endpoint comparisons use the finer-dt GM–Redi wave as a temporal
    # reference, not analytic truth or a spatially converged reference.
    output = []
    ref = min(
        (r for r in waves if r["info"]["variant"] == "gmredi"),
        key=lambda r: r["info"]["config"]["dt"],
        default=None,
    )
    for r in runs:
        d = r["data"]
        area = r["area"]
        row = dict(run=r["name"], **r["info"]["last"])
        for key in ("eta_rms_m", "initial_eta_projection"):
            row[key] = float(r["series"][key][-1])
        if r["info"]["config"]["case"] == "wave" and ref is not None:
            eta = d["eta"].reshape(-1)
            target = ref["data"]["eta"].reshape(-1)
            row["eta_rms_difference_from_fine_dt_m"] = float(
                np.sqrt(np.sum(area * (eta - target) ** 2) / area.sum())
            )
            row["eta_relative_L2_difference_from_fine_dt"] = float(
                np.sqrt(np.sum(area * (eta - target) ** 2) / np.sum(area * target**2))
            )
            row["eta_pattern_correlation_with_fine_dt"] = float(
                np.sum(area * eta * target)
                / np.sqrt(np.sum(area * eta**2) * np.sum(area * target**2))
            )
        output.append(row)
    (a.root / "wave_trade_metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
