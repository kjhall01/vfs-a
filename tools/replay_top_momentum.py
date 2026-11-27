#!/usr/bin/env python3
"""Compare first-interface momentum fluxes from a coupled NetCDF, offline.

This diagnostic advances only VFS atmospheric dynamics using the coupled
SSPRK(4,3) entry point. It does not run column physics, alter the checkpoint, or
grant production restart compatibility. Output is one JSON object per sample.
"""
import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfs import BoundedSSPRK3, CharneyPhillipsColumn, MoistAtmosphere, VoronoiMesh
from vfs.atmosphere import MoistState
from workflows.pyam21.dynamics import advance_dynamics


@dataclass(frozen=True)
class ReplayAir:
    atmosphere: MoistState
    cloud_fraction: torch.Tensor


def load_atmosphere(checkpoint, mesh_file, *, donor, gravity):
    """Rebuild and verify native geometry before reading the prognostics."""
    encoded = mesh_file.read_bytes()
    generators = np.load(io.BytesIO(encoded), allow_pickle=False)
    with Dataset(checkpoint) as source:
        if source.vfs_format != "coupled_restart_v1":
            raise ValueError("need a coupled NetCDF restart")
        if hashlib.sha256(encoded).hexdigest() != source.mesh_sha256:
            raise ValueError("ordered .npy mesh digest does not match checkpoint")

        def read(key):
            value = source[key][:]
            if np.any(np.ma.getmaskarray(value)):
                raise ValueError(f"missing data: {key}")
            return torch.from_numpy(np.array(value))

        mesh = VoronoiMesh(
            n_cells=len(generators), lloyd_iters=0, generator_xyz=generators,
            radius=float(source.sphere_radius_m), rotation_scheme=source.rotation_scheme,
        )
        if not torch.equal(mesh.cellsOnEdge, read("cells_on_edge")):
            raise ValueError("reconstructed native edges do not match checkpoint")
        torch.testing.assert_close(mesh.cell_xyz, read("cell_xyz"), rtol=0., atol=1e-14)
        column = CharneyPhillipsColumn(
            len(source.dimensions["air_level"]),
            read("air_hybrid_a"), read("air_hybrid_b"),
            phi_s=gravity * read("surface_height_m"),
            phi_full_mode="direct_u_gl2_adjoint",
        )
        model = MoistAtmosphere(
            mesh, column, stepper=BoundedSSPRK3(top_momentum_upwind=donor)
        )
        model.dynamics.kernel.gravity = gravity
        fields = ("u_e", "ps", "entropy_content", "water_content")
        atmosphere = MoistState(*(read("air_atmosphere_" + f) for f in fields), mesh, column)
        return int(source["time"].getValue()), model, ReplayAir(atmosphere, read("air_cloud_fraction"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mesh", type=Path, required=True, help="Original ordered .npy generators")
    parser.add_argument("--scheme", choices=("centered", "donor"), required=True)
    parser.add_argument("--dt", type=int, default=600)
    parser.add_argument("--seconds", type=int, default=14400)
    parser.add_argument("--sample-seconds", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gravity", type=float, default=9.8, help="Gravity used by the checkpoint")
    args = parser.parse_args()
    if min(args.dt, args.seconds, args.sample_seconds, args.threads) <= 0:
        parser.error("times and threads must be positive")
    if args.seconds % args.dt or args.sample_seconds % args.dt:
        parser.error("horizon and sample intervals must be multiples of dt")
    if not np.isfinite(args.gravity) or args.gravity <= 0:
        parser.error("gravity must be positive and finite")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    start, model, air = load_atmosphere(
        args.checkpoint, args.mesh, donor=args.scheme == "donor", gravity=args.gravity
    )

    def check(a):
        model.dynamics.transport_state(a.atmosphere)
        q = a.cloud_fraction
        if (not torch.isfinite(a.atmosphere.u_e).all() or not torch.isfinite(q).all()
                or (q < -1e-12).any() or (q > 1 + 1e-12).any()):
            raise ValueError("inadmissible wind or cloud fraction")

    def energy(a):
        return float(model.dynamics.energy_terms(a.atmosphere)["total"] / model.mesh.cell_area.sum())

    driver = SimpleNamespace(model=model, check=check)
    with torch.no_grad():
        check(air)
        initial_energy = energy(air)
        for t in range(0, args.seconds + 1, args.dt):
            if t:
                air = advance_dynamics(driver, air, args.dt, integrator="ssprk43")
            if t % args.sample_seconds == 0 or t == args.seconds:
                wind = air.atmosphere.u_e.abs()
                index = int(wind.argmax())
                edge, level = divmod(index, wind.shape[-1])
                print(json.dumps(dict(
                    scheme=args.scheme, dt_seconds=args.dt, elapsed_seconds=t,
                    model_seconds=start + t, peak_wind_m_s=float(wind[edge, level]),
                    peak_edge=edge, peak_level=level,
                    top_p99_m_s=float(torch.quantile(wind[:, 0], 0.99)),
                    energy_drift_J_m2=energy(air) - initial_energy,
                )), flush=True)


if __name__ == "__main__":
    main()
