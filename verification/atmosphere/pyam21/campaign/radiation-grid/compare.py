"""Reproduce analytic native-grid mapping errors without external optical data.

Run with installed VFS/PyAM21 from an external copy of tests/workflows/validation
and this verification directory, as for the installed-wheel regression suite.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tests"))
from test_radiation_mapping import (
    CP_AIR, GRAV, RadiationDriver, RadiationDriverConfig, HostRadiationDriver,
    NativeGridRadiationPackage, radiation_inputs, flux_convergence, model,
    set_date, set_time,
)
import torch
import vfs


def compare():
    host = model.__wrapped__()
    rows = []
    for levels in (12, 24, 40):
        config = RadiationDriverConfig(rad_time_step=3600, renormalize_sw_fluxes=True)
        original = RadiationDriver(config, NativeGridRadiationPackage())
        mapped = HostRadiationDriver(config, NativeGridRadiationPackage())
        for step in range(5):
            inputs = radiation_inputs(host, step, levels)
            time = set_date(2001, 6, 21, 12) + set_time(step * 1200)
            raw = original.step(time, 1200, **inputs)
            new = mapped.step(time, 1200, **inputs)
            boundary = flux_convergence(new).sum(-1)
            host_mass_cp = inputs["phalf"].diff() * CP_AIR / GRAV
            native_mass_cp = new["radiation_pflux"].diff() * CP_AIR / GRAV
            native_heat = new["native_tdtsw"] + new["native_tdtlw"]
            rows.append(dict(
                levels=levels, elapsed_seconds=step * 1200,
                fresh_radiation=new["do_rad"],
                surface_pressure_Pa=float(inputs["phalf"][0, -1]),
                old_max_residual_W_m2=float(((host_mass_cp * raw["tdt_rad"]).sum(-1) - boundary).abs().max()),
                mapped_max_residual_W_m2=float(((host_mass_cp * new["tdt_rad"]).sum(-1) - boundary).abs().max()),
                native_max_residual_W_m2=float(((native_mass_cp * native_heat).sum(-1) - boundary).abs().max()),
                max_boundary_flux_change_W_m2=max(float((new[k] - raw[k]).abs().max()) for k in ("dfsw", "ufsw", "flxnet")),
            ))
    return dict(
        kind="analytic native-grid radiation; no GFDL optical tables",
        vfs_source=vfs.__file__, torch=str(torch.__version__),
        physics_dt=1200, radiation_dt=3600, host_lid_Pa=1000, rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
