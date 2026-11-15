"""Construct the real upstream SeaEsfRad from explicitly supplied GFDL data.

No downloads, guessed tables, synthetic fallback, or changes to PyAM21 sources.
The manifest binds input bytes, sampling footprints and reference pressures.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from pyam21.radiation.driver import RadiationDriver
from pyam21.radiation.sea_esf import SeaEsfRad

DATA_URL = "ftp://ftp.gfdl.noaa.gov/perm/pjp/SCM/input_data.tar.gz"


@dataclass(frozen=True)
class RadiationSetup:
    driver: RadiationDriver
    identity: str
    manifest: dict


def climatology_boxes(mesh, width_degrees):
    """Local rectangular sampling, NOT conservative Voronoi remapping.

    The explicit width is recorded. Periodic longitude is unwrapped; polar
    boxes are truncated at the pole. This samples prescribed climatology only.
    """
    if not math.isfinite(width_degrees) or not 0 < width_degrees <= 180:
        raise ValueError("climatology footprint width must be in (0,180] degrees")
    half = math.radians(width_degrees) * 0.5
    xyz = mesh.cell_xyz.detach().cpu()
    lon = torch.atan2(xyz[:, 1], xyz[:, 0]).tolist()
    lat = torch.asin(xyz[:, 2]).tolist()
    return [
        (
            (lo - half, lo + half),
            (max(-math.pi / 2, la - half), min(math.pi / 2, la + half)),
        )
        for lo, la in zip(lon, lat)
    ]


def reference_pressures(column):
    ps = torch.tensor([[101325.0], [81060.0]], dtype=torch.float64)
    ph = column.interface_pressure(ps)
    dp = ph.diff()
    if (ph <= 0).any() or (dp <= 0).any():
        raise ValueError("positive increasing reference pressures required")
    r = dp / ph[:, :-1]
    full = ph[:, 1:] * torch.exp(-1 + torch.log1p(r) / r)
    return torch.cat((full, ps), -1).T.numpy()


def build_radiation(namelist, model, data_dir, *, width_degrees=1.0):
    root = Path(data_dir).resolve()
    required = ["stdlvls", "h2ocoeff_ckd_speccombwidebds_hi00", "h2ockd2.1_data"]
    missing = [f for f in required if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SeaEsfRad requires the extracted sea_esf directory; missing {missing} in {root}. Obtain the GFDL SCM bundle from {DATA_URL}; no fallback is used."
        )
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            files[str(p.relative_to(root))] = dict(
                bytes=p.stat().st_size, sha256=h.hexdigest()
            )
    boxes = climatology_boxes(model.mesh, width_degrees)
    pref = reference_pressures(model.column)
    manifest = dict(
        schema=1,
        package="pyam21.radiation.sea_esf.SeaEsfRad",
        source=DATA_URL,
        files=files,
        climatology_mapping="local rectangular samples, not mesh-cell averages",
        width_degrees=width_degrees,
        boxes=boxes,
        reference_pressures_Pa=pref.tolist(),
        namelist=namelist,
    )
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    package = SeaEsfRad(
        namelist,
        str(root),
        boxes,
        np.asarray(pref),
        do_totcld_forcing=bool(
            namelist.get("radiation_driver_nml", {}).get("do_clear_sky_pass", False)
        ),
    )
    return RadiationSetup(
        RadiationDriver.from_nml(namelist, package), identity, manifest
    )
