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

DATA_URL = "ftp://ftp.gfdl.noaa.gov/perm/GFDL_pubrelease/SCM/input_data.tar.gz"


@dataclass(frozen=True)
class RadiationSetup:
    driver: RadiationDriver
    identity: str
    manifest: dict


def remap_heating(heating, radiation_pflux, host_phalf):
    """Conserve layer-integrated heating between two fractional-pressure grids.

    heating is [column, radiation layer, band] in K/s. Both grids share surface
    pressure on a fresh call; fractional pressure keeps the entire cached
    column when host pressure subsequently changes. cp/g cancels in the map.

    SeaEsfRad extends to p=0 while VFS has a positive lid. Explicitly include
    absorption above that lid in the top host layer; divide by the actual host
    layer mass. This lumped cap keeps true TOA and surface fluxes as boundaries.
    It is not a resolved stratosphere above the lid.
    """
    if (
        heating.ndim != 3
        or radiation_pflux.shape != (heating.shape[0], heating.shape[1] + 1)
        or host_phalf.ndim != 2
        or host_phalf.shape[0] != heating.shape[0]
        or host_phalf.shape[1] < 2
    ):
        raise ValueError("radiation/host heating grid shapes disagree")
    dp_rad, dp_host = radiation_pflux.diff(), host_phalf.diff()
    if (
        not torch.isfinite(heating).all()
        or not torch.isfinite(radiation_pflux).all()
        or not torch.isfinite(host_phalf).all()
        or (dp_rad <= 0).any()
        or (dp_host <= 0).any()
        or (radiation_pflux[:, 0] != 0).any()
        or (host_phalf[:, 0] < 0).any()
    ):
        raise ValueError("finite increasing radiation/host pressures required; radiation top must be zero")
    cumulative = torch.cat(
        (torch.zeros_like(heating[:, :1]), (dp_rad[..., None] * heating).cumsum(1)),
        dim=1,
    )
    source = (radiation_pflux / radiation_pflux[:, -1:]).contiguous()
    target = (host_phalf[:, 1:-1] / host_phalf[:, -1:]).contiguous()
    lower = torch.searchsorted(source, target, right=True) - 1
    # Interior target interfaces are strictly inside (0,1); no extrapolation.
    x0, x1 = source.gather(1, lower), source.gather(1, lower + 1)
    fraction = (target - x0) / (x1 - x0)
    ids = lower[..., None].expand(-1, -1, heating.shape[-1])
    q0, q1 = cumulative.gather(1, ids), cumulative.gather(1, ids + 1)
    interior = q0 + fraction[..., None] * (q1 - q0)
    mapped = torch.cat((cumulative[:, :1], interior, cumulative[:, -1:]), dim=1)
    return mapped.diff(dim=1) / dp_host[..., None]


class HostRadiationDriver(RadiationDriver):
    """Keep upstream optics/cache native; deliver conservative host-layer heat."""

    def step(self, time, dt, lat, lon, pfull, phalf, t, q, t_surf, surface, extra=None):
        result = super().step(
            time, dt, lat, lon, pfull, phalf, t, q, t_surf, surface, extra
        )
        if result["do_rad"]:
            # saved is already part of the adapter's transactional restart state.
            self.saved["vfs_radiation_pflux"] = result["atmos"]["pflux"].clone()
        pflux = self.saved["vfs_radiation_pflux"]
        native = torch.stack((result["tdtsw"], result["tdtlw"]), dim=-1)
        mapped = remap_heating(native, pflux, phalf)
        # Return new tensors, never replace the upstream cached native heating.
        # In particular renormalized SW must not be remapped twice on cache hits.
        return {
            **result,
            "tdtsw": mapped[..., 0],
            "tdtlw": mapped[..., 1],
            "tdt_rad": mapped.sum(-1),
            "native_tdtsw": native[..., 0],
            "native_tdtlw": native[..., 1],
            "radiation_pflux": pflux,
        }


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
        schema=2,
        package="pyam21.radiation.sea_esf.SeaEsfRad",
        source=DATA_URL,
        files=files,
        climatology_mapping="local rectangular samples, not mesh-cell averages",
        heating_mapping="conservative fractional-pressure overlap v1",
        radiation_lid_policy="above-lid absorption included in top host layer",
        radiation_flux_grid="native SeaEsfRad interfaces; TOA/surface unchanged",
        radiation_cache="native heating and pressure interfaces; remap at each physics call",
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
    # Upstream SeaEsfRad does not forward the fixed solar constant to its SW
    # constructor. Configure the owned package explicitly without source edits.
    fixed_solar = namelist.get("radiation_driver_nml", {}).get("solar_constant")
    if fixed_solar is not None:
        fixed_solar = float(fixed_solar)
        if not math.isfinite(fixed_solar) or fixed_solar <= 0:
            raise ValueError("positive finite solar constant required")
        if namelist["radiation_driver_nml"].get(
            "solar_dataset_entry", [1, 1, 1, 0, 0, 0]
        ) != [1, 1, 1, 0, 0, 0]:
            raise ValueError("choose fixed solar constant or spectral time series")
        package.sw.solar_constant = fixed_solar
    return RadiationSetup(
        HostRadiationDriver.from_nml(namelist, package), identity, manifest
    )
