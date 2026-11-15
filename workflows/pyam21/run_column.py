"""Run actual AM2 physics on native VFS columns and retain audit time series."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch

# Import workflows only. VFS and PyAM21 must be separately installed packages.
sys.path.append(str(Path(__file__).resolve().parents[2]))


def run(args, *, coupled=False):
    from workflows.pyam21.setup import example
    from workflows.pyam21.dynamics import step

    adapter, state = example(
        args.source,
        cells=args.cells,
        levels=args.levels,
        aquaplanet=coupled,
        energy_mode=args.energy_mode,
        radiation_data=args.radiation_data,
        radiation_step=args.radiation_step,
        climatology_width_degrees=args.climatology_width_degrees,
    )
    model = adapter.model
    area = model.mesh.cell_area
    area_total = area.sum()
    initial = model.diagnostics(state.atmosphere)
    initial_dry = initial["mass"] - initial["species"].sum()
    initial_water = initial["species"].sum()
    energy0 = initial["energy"]
    supplied_water = torch.zeros_like(area)
    physics_energy = torch.zeros_like(area)
    declared_heat = torch.zeros_like(area)
    convention_change = torch.zeros_like(area)
    representation_change = torch.zeros_like(area)
    external_energy = torch.zeros_like(area)
    upstream_residual = torch.zeros_like(area)
    transfer_residual = torch.zeros_like(area)
    rows = []
    start = time.perf_counter()
    seconds = args.hours * 3600
    with torch.no_grad():
        while state.time_seconds < seconds:
            dt = min(args.dt, seconds - state.time_seconds)
            if coupled:
                state, report = step(adapter, state, dt, dynamics_dt=args.dynamics_dt)
            else:
                state, report = adapter.apply(state, dt)
            supplied_water += dt * (
                report.fluxes["flux_q"]
                - report.fluxes["lprec"]
                - report.fluxes["fprec"]
            )
            physics_energy += report.energy_changes["vfs_full_h"]
            declared_heat += report.energy_changes["declared_thermal_latent_forcing"]
            convention_change += report.energy_changes[
                "thermodynamic_convention_difference"
            ]
            representation_change += report.energy_changes[
                "mapping_and_carrier_difference"
            ]
            external_energy += report.energy_changes["external_energy"]
            upstream_residual += report.energy_changes["am2_energy_residual"]
            transfer_residual += report.energy_changes["transfer_residual"]
            now = model.diagnostics(state.atmosphere)
            water = now["species"].sum()
            dry = now["mass"] - water

            def mean(x):
                return float((area * x).sum() / area_total)

            row = dict(
                seconds=state.time_seconds,
                dry_air_relative=float((dry - initial_dry) / initial_dry),
                water_flux_error_kg_m2=float(
                    (water - initial_water - (area * supplied_water).sum()) / area_total
                ),
                energy_change_J_m2=float((now["energy"] - energy0) / area_total),
                dynamics_energy_residual_J_m2=float(
                    (now["energy"] - energy0) / area_total
                )
                - mean(physics_energy),
                water_mapping_max_kg_m2=float(
                    report.water_mapping_residual.abs().max()
                ),
                temperature_min_K=float(
                    model.physical_fields(state.atmosphere)["temperature"].min()
                ),
                temperature_max_K=float(
                    model.physical_fields(state.atmosphere)["temperature"].max()
                ),
                transfer_layer_max_J_m2=(
                    0.0
                    if report.energy_transfer is None
                    else float(report.energy_transfer["layer_residual"].abs().max())
                ),
                transfer_log_temperature_max=(
                    0.0
                    if report.energy_transfer is None
                    else float(
                        report.energy_transfer["log_temperature_increment"].abs().max()
                    )
                ),
                transfer_iterations=(
                    0
                    if report.energy_transfer is None
                    else report.energy_transfer["iterations"]
                ),
                cloud_fraction_max=float(state.cloud_fraction.max()),
                max_wind_m_s=float(state.atmosphere.u_e.abs().max()),
                precipitation_kg_m2_s=mean(
                    report.fluxes["lprec"] + report.fluxes["fprec"]
                ),
            )
            row.update({k + "_J_m2": mean(v) for k, v in report.energy_changes.items()})
            row.update(
                cumulative_declared_heat_J_m2=mean(declared_heat),
                cumulative_convention_change_J_m2=mean(convention_change),
                cumulative_mapping_carrier_change_J_m2=mean(representation_change),
                cumulative_external_energy_J_m2=mean(external_energy),
                cumulative_am2_residual_J_m2=mean(upstream_residual),
                cumulative_transfer_residual_J_m2=mean(transfer_residual),
                cumulative_full_h_minus_external_J_m2=float(
                    (now["energy"] - energy0) / area_total
                )
                - mean(external_energy),
                vfs_minus_AM2_heat_ledger_J_m2=float(
                    (now["energy"] - energy0) / area_total
                )
                - mean(declared_heat),
            )
            rows.append(row)
            if args.verbose and (len(rows) % 10 == 0 or state.time_seconds == seconds):
                print(
                    f"{state.time_seconds / 3600:.3f} h: water residual {row['water_flux_error_kg_m2']:.3g} kg/m2; "
                    f"Tmin {row['temperature_min_K']:.2f} K; max wind {row['max_wind_m_s']:.3g} m/s",
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "series.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        coupled_dynamics=coupled,
        radiation=adapter.radiation_manifest,
        steps=len(rows),
        wall_seconds=time.perf_counter() - start,
        restart_fingerprint=adapter.fingerprint,
        final=rows[-1],
        peaks={
            k: max(abs(r[k]) for r in rows)
            for k in (
                "dry_air_relative",
                "water_flux_error_kg_m2",
                "water_mapping_max_kg_m2",
                "dynamics_energy_residual_J_m2",
                "transfer_layer_max_J_m2",
                "transfer_log_temperature_max",
                "transfer_iterations",
            )
        },
        torch=torch.__version__,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2)
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="checkout of the pinned PyAM21 revision, for its example namelist",
    )
    p.add_argument(
        "--radiation-data",
        type=Path,
        help="extracted GFDL sea_esf directory; enables actual radiation",
    )
    p.add_argument("--radiation-step", type=int, default=3600)
    p.add_argument("--climatology-width-degrees", type=float, default=1.0)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--levels", type=int, default=24)
    p.add_argument("--hours", type=float, default=1)
    p.add_argument("--dt", type=int, default=60, help="physics seconds")
    p.add_argument("--dynamics-dt", type=float, default=15)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--energy-mode", choices=("conservative", "legacy"), default="conservative"
    )
    args = p.parse_args()
    if (
        args.radiation_step < args.dt
        or args.radiation_step % args.dt != 0
        or args.cells < 16
        or args.levels < 4
        or args.dt <= 0
        or not all(math.isfinite(x) and x > 0 for x in (args.hours, args.dynamics_dt))
        or not float(args.hours * 3600).is_integer()
    ):
        p.error(
            "cells>=16, levels>=4, positive steps and whole-number duration seconds required"
        )
    return args


if __name__ == "__main__":
    run(parser())
