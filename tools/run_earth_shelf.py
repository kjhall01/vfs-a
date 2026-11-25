#!/usr/bin/env python3
"""Run the coupled 4,096-cell Earth shelf campaign from a VFS checkout."""

import os
from pathlib import Path
import sys


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG") or ":4096:8"

    from workflows.coupled import campaign

    args = list(sys.argv[1:] if argv is None else argv)
    data = repo / "input-data"

    def default_input(stem, legacy_suffix):
        nc = data / (stem + ".nc")
        legacy = data / (stem + legacy_suffix)
        return legacy if legacy.is_file() and not nc.is_file() else nc

    defaults = [
        "--config", str(repo / "workflows/coupled/earth_shelf_4096.yaml"),
        "--source", os.environ.get("PYAM21_SOURCE") or str(repo.parent / "PyAM21"),
        "--mesh", os.environ.get("VFS_MESH") or str(default_input("mesh_4096_exact_3000", ".npy")),
        "--geography", os.environ.get("VFS_GEOGRAPHY") or str(default_input("earth_4096_geography", ".npz")),
        "--verbose",
    ]
    if "--smoke-no-radiation" not in args:
        defaults.extend(["--radiation-data", os.environ.get("PYAM21_RADIATION_DATA") or str(data / "sea_esf")])
    return campaign.main(defaults + args)


if __name__ == "__main__":
    raise SystemExit(main())
