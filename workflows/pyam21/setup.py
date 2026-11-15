"""Explicit example configuration; no data download or package-default changes."""

from pathlib import Path
import torch
from pyam21.constants import GRAV
from pyam21.namelist import read_namelist
from pyam21.sat_vapor_pres import get_sat_vapor_pres
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)
from .adapter import AM2Adapter


def example(source, *, cells=32, levels=24, aquaplanet=False):
    """Analytic moist profile with BOMEX physics options, not the BOMEX case.

    There is no SCM large-scale forcing or radiation in this example. Surface
    fluxes, RAS convection, stratiform clouds and mixing are calculated by AM2.
    Positive upper-air vapor and a positive pressure lid are explicit initial
    conditions required by VFS, not hidden modifications to a reference case.
    """
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    nml = read_namelist(str(Path(source) / "reference/input/gcss_bomex/input.nml"))
    mesh = VoronoiMesh(n_cells=cells, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(levels, p_top=1000.0)
    column = CharneyPhillipsColumn(levels, A, B, phi_full_mode="direct_u_gl2_adjoint")
    model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
    # Explicit workflow configuration BEFORE any prognostic state exists. The
    # native dynamics and every mass/energy diagnostic use this same parameter.
    model.dynamics.kernel.gravity = GRAV
    adapter = AM2Adapter(model, nml)
    p = model.interface_pressure(100000.0)
    sine_lat = mesh.cell_xyz[:, 2]
    surface = 300.0 - (25 * sine_lat.square() if aquaplanet else 0 * sine_lat)
    T = 205 + (surface[:, None] - 205) * (p / 100000.0) ** 0.7
    vapor = 0.7 * get_sat_vapor_pres().compute_qs(T, p)
    atmosphere = model.state_from_fields(
        temperature=T, surface_pressure=100000.0, normal_wind=0.0, vapor=vapor
    )
    state = adapter.initialize(atmosphere, sst=surface + 1.0)
    return adapter, state
