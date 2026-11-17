"""Analytic radiation-grid budgets; no optical tables or physical-flux claims."""

from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

pytest.importorskip("pyam21")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pyam21.constants import CP_AIR, GRAV
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig
from pyam21.time_manager import set_date, set_time
from workflows.pyam21.radiation import HostRadiationDriver, remap_heating
from test_pyam21_workflow import model, initial, same_tree, energy_close
from vfs import CharneyPhillipsColumn, make_hybrid_coefficients


class NativeGridRadiationPackage:
    """Analytic interface fluxes on the actual upstream pflux geometry.

    Optional deliberately inconsistent LW heating tests that the host transfer
    does not repair or hide a radiation-package energy defect.
    """

    def __init__(self, defect_W_m2=0.0):
        self.defect = defect_W_m2

    def __call__(self, atmos, surface, astro, time, do_lw, do_sw, extra):
        p = atmos["pflux"]
        x = p / p[:, -1:]
        solar = astro["solar"][:, None]
        down = solar * 650 * torch.exp(-0.8 * x)
        up = solar * (35 + 25 * x.square())
        lw = 260 - 180 * x.square()
        factor = GRAV / CP_AIR * 86400 / p.diff()
        heat_lw = lw.diff() * factor
        defect = torch.zeros_like(heat_lw)
        defect[:, 0] = self.defect * factor[:, 0]
        return dict(
            hsw=-(down - up).diff() * factor,
            heatra=heat_lw + defect,
            dfsw=down,
            ufsw=up,
            flxnet=lw,
        )


def radiation_inputs(model, step=0, levels=24):
    a, b = make_hybrid_coefficients(levels, p_top=1000.)
    column = CharneyPhillipsColumn(levels, a, b)
    phalf = column.interface_pressure(torch.full((model.mesh.nCells, 1), 1e5))
    # Change carrier masses and interior locations during cache hits.
    phalf = phalf[:, :1] + (phalf - phalf[:, :1]) * (1 + step * 0.01)
    dp = phalf.diff()
    ratio = dp / phalf[:, :-1]
    pfull = phalf[:, 1:] * torch.exp(-1 + torch.log1p(ratio) / ratio)
    temp = 205 + 95 * (pfull / phalf[:, -1:]) ** 0.7
    xyz = model.mesh.cell_xyz
    return dict(
        lat=torch.asin(xyz[:, 2]),
        lon=torch.atan2(xyz[:, 1], xyz[:, 0]),
        pfull=pfull,
        phalf=phalf,
        t=temp,
        q=torch.full_like(temp, 0.005),
        t_surf=torch.full_like(temp[:, 0], 300),
        surface={},
    )


def flux_convergence(result):
    sw = result["dfsw"] - result["ufsw"]
    lw = -result["flxnet"]
    return torch.stack((sw[:, 0] - sw[:, -1], lw[:, 0] - lw[:, -1]), dim=-1)


def test_overlap_matches_independent_layer_intersections_and_lid():
    # Two columns with different cached and current surface pressures; signed
    # heating and independently integrated overlap amounts, not endpoint-only.
    source = torch.tensor([[0., 2e4, 6e4, 1e5], [0., 1.6e4, 4.8e4, 8e4]], dtype=torch.float64)
    target = torch.tensor([[1e3, 1e4, 4e4, 1e5], [2e3, 1.2e4, 4.8e4, 1.2e5]], dtype=torch.float64)
    heating = torch.tensor([[[1., -3.], [-2., 2.], [3., -1.]]] * 2, dtype=torch.float64) * 1e-5
    expected = torch.zeros_like(heating)
    for c in range(2):
        boundaries = target[c].tolist()
        boundaries[0] = 0.0  # explicitly include the above-lid cap
        boundaries = [p / boundaries[-1] for p in boundaries]
        source_x = (source[c] / source[c, -1]).tolist()
        for k in range(3):
            for j in range(3):
                overlap = max(0., min(boundaries[k + 1], source_x[j + 1]) - max(boundaries[k], source_x[j]))
                expected[c, k] += overlap * source[c, -1] * heating[c, j] / target[c].diff()[k]
    actual = remap_heating(heating, source, target)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=1e-19)
    torch.testing.assert_close(
        (actual * target.diff()[..., None]).sum(1),
        (heating * source.diff()[..., None]).sum(1),
        rtol=2e-14, atol=1e-15,
    )


@pytest.mark.parametrize("defect", [0., 3.])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("levels", [12, 24, 40])
def test_fresh_cached_day_night_pressure_changes_and_native_defects(model, defect, renormalize, levels):
    cfg = RadiationDriverConfig(rad_time_step=3600, renormalize_sw_fluxes=renormalize)
    original = RadiationDriver(cfg, NativeGridRadiationPackage(defect))
    mapped = HostRadiationDriver(cfg, NativeGridRadiationPackage(defect))
    start = set_date(2001, 6, 21, 12)
    fresh = []
    raw_errors = []
    for step in range(5):
        inputs = radiation_inputs(model, step, levels)
        raw = original.step(start + set_time(step * 1200), 1200, **inputs)
        result = mapped.step(start + set_time(step * 1200), 1200, **inputs)
        fresh.append(result["do_rad"])
        # All optics, TOA/surface fluxes and native caches remain upstream values.
        for key in ("dfsw", "ufsw", "flxnet", "flux_sw_surf", "flux_lw_surf"):
            torch.testing.assert_close(result[key], raw[key], rtol=0, atol=0)
        for key, value in original.saved.items():
            same_tree(value, mapped.saved[key])
        heat = torch.stack((result["tdtsw"], result["tdtlw"]), dim=-1)
        integral = (inputs["phalf"].diff()[..., None] * CP_AIR / GRAV * heat).sum(1)
        expected = flux_convergence(result)
        expected[:, 1] += defect
        torch.testing.assert_close(integral, expected, rtol=3e-14, atol=2e-12)
        torch.testing.assert_close(result["tdt_rad"], result["tdtsw"] + result["tdtlw"], rtol=0, atol=0)
        native = torch.stack((result["native_tdtsw"], result["native_tdtlw"]), dim=-1)
        native_integral = (result["radiation_pflux"].diff()[..., None] * CP_AIR / GRAV * native).sum(1)
        torch.testing.assert_close(integral, native_integral, rtol=3e-14, atol=2e-12)
        raw_integral = (inputs["phalf"].diff() * CP_AIR / GRAV * raw["tdt_rad"]).sum(1)
        raw_errors.append(float((raw_integral - expected.sum(-1)).abs().max()))
    assert fresh == [True, False, False, True, False]
    assert max(raw_errors) > 1.0  # demonstrates the old W/m² defect
    assert (result["dfsw"][:, 0] == 0).any()
    assert (result["dfsw"][:, 0] > 0).any()


def test_mapped_physics_restart_and_transactional_failure(model):
    driver = HostRadiationDriver(
        RadiationDriverConfig(rad_time_step=180, renormalize_sw_fluxes=True),
        NativeGridRadiationPackage(),
    )
    adapter, state = initial(model, radiation=driver)
    state, report = adapter.apply(state, 60)
    assert "vfs_radiation_pflux" in state.radiation["saved"]
    assert driver.saved is None
    resumed = adapter.unpack(adapter.pack(state))
    for _ in range(3):
        before = state
        state, report = adapter.apply(state, 60)
        resumed, _ = adapter.apply(resumed, 60)
        same_tree(adapter.pack(state), adapter.pack(resumed))
        energy_close(adapter, before, state, report)
        torch.testing.assert_close(
            report.energy_changes["radiation_flux_residual"] / 60,
            torch.zeros(model.mesh.nCells), rtol=0, atol=2e-10,
        )
    packed = adapter.pack(state)
    with patch.object(adapter, "apply_result", side_effect=ValueError("deliberate")):
        with pytest.raises(ValueError, match="deliberate"):
            adapter.apply(state, 60)
    same_tree(packed, adapter.pack(state))
    assert driver.saved is None and driver.lw_alarm == 1


@pytest.mark.parametrize("bad", ["nonfinite", "reversed", "nonzero_top"])
def test_remap_rejects_invalid_native_pressure(bad):
    source = torch.tensor([[0., 1e4, 1e5]], dtype=torch.float64)
    target = torch.tensor([[1e3, 1e4, 1e5]], dtype=torch.float64)
    if bad == "nonfinite":
        source[:, 1] = float("nan")
    elif bad == "reversed":
        source[:, 1] = 2e5
    else:
        source[:, 0] = 1e3
    with pytest.raises(ValueError, match="pressures"):
        remap_heating(torch.ones(1, 2, 2), source, target)
