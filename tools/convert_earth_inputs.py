#!/usr/bin/env python3
"""Convert existing mesh generators and optionally their paired geography to NetCDF4."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.coupled.grid_io import read_geography, read_mesh, write_geography, write_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-output", type=Path, required=True)
    parser.add_argument("--geography", type=Path)
    parser.add_argument("--geography-output", type=Path)
    args = parser.parse_args()
    if bool(args.geography) != bool(args.geography_output):
        parser.error("supply --geography and --geography-output together")
    outputs = [args.mesh_output] + ([args.geography_output] if args.geography else [])
    if len({p.resolve() for p in outputs}) < len(outputs) or any(p.suffix != ".nc" or p.exists() for p in outputs):
        parser.error("choose distinct, new .nc output files")
    xyz, identity = read_mesh(args.mesh)
    if args.geography:
        geo = read_geography(args.geography, mesh_sha256=identity, xyz=xyz)
    for p in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
    write_mesh(args.mesh_output, xyz, mesh_sha256=identity)
    if args.geography:
        write_geography(args.geography_output, geo, xyz, identity)


if __name__ == "__main__":
    main()
