"""Independent combined budgets, impulse adjoints, tiling and data provenance."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch
import pytest
import torch

pytest.importorskip("pyam21")
from test_pyam21_workflow import model, initial, PrescribedRadiationPackage
from workflows.exchange import Exchange
from workflows.pyam21.surface import prepare_surface
from workflows.pyam21.radiation import (
    build_radiation,
    reference_pressures,
    climatology_boxes,
)
from workflows.coupled.ocean_ice import OceanIceReceiver
from workflows.coupled.driver import exchange_step
from ocean_ice import OceanIce
from ridging import CategoryIce, RidgingState
from snow import SnowOceanIce
from roquet import RoquetOcean
from vfs.ocean import OceanColumn
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig


def surface_case(model, cover=True):
    mesh = model.mesh
    ocean = RoquetOcean(mesh, OceanColumn(3, total_depth=200.0, gravity=9.8))
    h = torch.tensor([20.0, 60.0, 120.0]).expand(mesh.nCells, -1).clone()
    s = ocean.state(
        u_e=torch.full((mesh.nEdges, 3), 0.1),
        h=h,
        conservative_temperature=torch.full_like(h, 3.0),
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4,
        eddy_energy=0.01,
    )
    area = torch.tensor([0.4, 0.1]).expand(mesh.nCells, -1).clone() * (
        1 if cover else 0
    )
    ice = CategoryIce(
        area,
        area * torch.tensor([0.3, 1.0]),
        torch.full((mesh.nEdges, 1), -0.05) if cover else torch.zeros(mesh.nEdges, 1),
    )
    snow = SnowOceanIce(OceanIce(ocean))
    state = snow.bare(RidgingState(s, ice))
    return OceanIceReceiver(snow), state


def total_water(adapter, air, receiver, water):
    A = adapter.model.mesh.cell_area
    atmospheric = adapter.model.diagnostics(air.atmosphere)["species"].sum()
    ocean = water.coupled.ocean.ocean
    return (
        atmospheric
        + (
            A
            * (
                ocean.column.rho0 * ocean.h.sum(-1)
                + receiver.H.mass(water.coupled)[:, 0]
            )
        ).sum()
    )


@pytest.mark.parametrize("cover", [False, True])
@pytest.mark.parametrize("radiation", [False, True])
def test_combined_first_law_and_reciprocal_impulse(model, cover, radiation):
    driver = (
        RadiationDriver(
            RadiationDriverConfig(rad_time_step=120), PrescribedRadiationPackage()
        )
        if radiation
        else None
    )
    adapter, air = initial(model, radiation=driver)
    receiver, water = surface_case(model, cover)
    A = model.mesh.cell_area
    E0 = model.diagnostics(air.atmosphere)["energy"] + receiver.H.total_energy(
        water.coupled
    )
    W0 = total_water(adapter, air, receiver, water)
    salt0 = water.coupled.ocean.ocean.salt_h.clone()
    out, sea, x, ar, sr = exchange_step(
        adapter,
        air,
        receiver,
        water,
        60,
        ice_temperature=torch.full_like(water.snow_mass, 270.0),
    )
    E1 = model.diagnostics(out.atmosphere)["energy"] + receiver.H.total_energy(
        sea.coupled
    )
    upstream = ar.energy_changes["am2_energy_residual"]
    assert abs(float((E1 - E0 - (A * (x.toa + upstream)).sum()) / A.sum())) < 2e-6
    assert abs(float((total_water(adapter, out, receiver, sea) - W0) / A.sum())) < 1e-9
    torch.testing.assert_close(sea.coupled.ocean.ocean.salt_h, salt0, rtol=0, atol=0)
    assert sr.energy_residual.abs().max() < 2e-6
    assert sr.water_residual.abs().max() < 1e-10
    opposite = adapter.mapping.transpose(
        (A[:, None] * (x.water_impulse + x.stress.sum(1)))[:, None]
    )
    scale = (sr.ocean_impulse.abs() + sr.ice_impulse.abs() + opposite.abs()).clamp_min(
        1
    )
    assert ((sr.ocean_impulse + sr.ice_impulse + opposite).abs() / scale).max() < 5e-13
    for k, v in ar.column_result["tile_fluxes"].items():
        torch.testing.assert_close(v.sum(-1), ar.fluxes[k], rtol=1e-12, atol=1e-13)
    assert not torch.equal(
        sea.coupled.ocean.ocean.temperature(), water.coupled.ocean.ocean.temperature()
    )
    assert not torch.equal(sea.coupled.ocean.ocean.u_e, water.coupled.ocean.ocean.u_e)
    assert air.time_seconds == 0 and out.time_seconds == 60


def test_cached_longwave_uses_same_bottom_interface(model):
    driver = RadiationDriver(
        RadiationDriverConfig(rad_time_step=180), PrescribedRadiationPackage()
    )
    adapter, air = initial(model, radiation=driver)
    air, _ = adapter.apply(air, 60)
    air = replace(air, surface={**air.surface, "t_surf": air.surface["t_surf"] + 5})
    _, r = adapter.apply(air, 60)
    torch.testing.assert_close(
        r.fluxes["lw_net_surface"],
        -r.column_result["radiation"]["flxnet"][:, -1],
        rtol=0,
        atol=0,
    )


def test_exchange_rejects_wrong_mesh_and_atomic_failure(model):
    adapter, air = initial(model)
    receiver, water = surface_case(model)
    before = water.coupled.ocean.ocean.temp_h.clone()
    with patch.object(receiver, "finish", side_effect=ValueError("deliberate")):
        with pytest.raises(ValueError, match="deliberate"):
            exchange_step(
                adapter,
                air,
                receiver,
                water,
                60,
                ice_temperature=torch.full_like(water.snow_mass, 270.0),
            )
    assert air.time_seconds == 0
    torch.testing.assert_close(water.coupled.ocean.ocean.temp_h, before, rtol=0, atol=0)
    tiles = receiver.tiles(water, torch.full_like(water.snow_mass, 270.0))
    with pytest.raises(ValueError, match="fractions"):
        replace(tiles, fraction=tiles.fraction * 0.9).validate(model.mesh)


def test_snow_deposition_sublimation_and_exhaustion(model):
    adapter, air = initial(model)
    receiver, water = surface_case(model)
    tiles = receiver.tiles(water, torch.full_like(water.snow_mass, 270.0))
    air = prepare_surface(air, tiles)
    c, g = adapter.mapping.columns(air.atmosphere, air.cloud_fraction)
    out, *_ = adapter.column_call(air, 60, c, g, surface_tiles=tiles)
    x = Exchange.from_am2(adapter, air, 60, c, out, tiles)
    # Deliberate exact prescribed streams exercise receiver independently of AM2.
    z = torch.zeros_like(x.rain)
    vapor = torch.stack((-z - 0.2, z + 0.1, z - 0.3), -1)
    x = replace(
        x, vapor=vapor, rain=z + 0.2, snow=z + 0.4, stress=torch.zeros_like(x.stress)
    )
    candidate, x, imp = receiver.prepare(water, x, tiles)
    end, r = receiver.finish(water, candidate, x, imp)
    assert r.water_residual.abs().max() < 1e-10
    assert (end.snow_mass[:, 1] > water.snow_mass[:, 1]).all()
    with pytest.raises(ValueError, match="exhausts frozen"):
        receiver.prepare(water, replace(x, vapor=torch.full_like(vapor, 1e6)), tiles)


def test_radiation_builder_missing_data_and_reference_grid(model, tmp_path):
    with pytest.raises(FileNotFoundError, match="no fallback"):
        build_radiation({}, model, tmp_path)
    pref = reference_pressures(model.column)
    assert pref.shape == (model.column.n_levels + 1, 2)
    assert (pref[1:] > pref[:-1]).all()
    assert pref[-1].tolist() == [101325.0, 81060.0]
    boxes = climatology_boxes(model.mesh, 1.0)
    assert len(boxes) == model.mesh.nCells
    for lo, la in boxes:
        assert lo[1] > lo[0] and la[1] > la[0]
    with pytest.raises(ValueError):
        climatology_boxes(model.mesh, 0)


def test_radiation_manifest_binds_actual_files_and_geometry(model, tmp_path):
    for f in ("stdlvls", "h2ocoeff_ckd_speccombwidebds_hi00", "h2ockd2.1_data"):
        (tmp_path / f).write_text("builder orchestration fixture, NOT radiation data")
    with patch(
        "workflows.pyam21.radiation.SeaEsfRad",
        return_value=PrescribedRadiationPackage(),
    ) as constructor:
        a = build_radiation({}, model, tmp_path)
        b = build_radiation({}, model, tmp_path, width_degrees=2.0)
        (tmp_path / "stdlvls").write_text("changed")
        c = build_radiation({}, model, tmp_path)
    assert a.identity != b.identity and a.identity != c.identity
    assert len(constructor.call_args.args[2]) == model.mesh.nCells
    assert a.manifest["package"].endswith("SeaEsfRad")
    from workflows.pyam21.radiation import HostRadiationDriver

    assert isinstance(a.driver, HostRadiationDriver)
    assert a.manifest["schema"] == 2
    assert "above-lid" in a.manifest["radiation_lid_policy"]


@pytest.mark.skipif(
    not os.environ.get("PYAM21_RADIATION_DATA"),
    reason="real GFDL SeaEsfRad data not supplied",
)
def test_real_seaesf_opt_in():
    from workflows.pyam21.setup import example

    source = os.environ["PYAM21_SOURCE"]
    adapter, air = example(
        source,
        cells=16,
        levels=24,
        radiation_data=os.environ["PYAM21_RADIATION_DATA"],
        radiation_step=120,
    )
    first, r = adapter.apply(air, 60)
    resumed = adapter.unpack(adapter.pack(first))
    end, s = adapter.apply(first, 60)
    replay, t = adapter.apply(resumed, 60)
    assert r.column_result["radiation"]["do_rad"]
    assert not s.column_result["radiation"]["do_rad"]
    assert torch.isfinite(s.energy_changes["radiation_flux_residual"]).all()
    torch.testing.assert_close(
        end.atmosphere.entropy_content,
        replay.atmosphere.entropy_content,
        rtol=0,
        atol=0,
    )
    third, u = adapter.apply(end, 60)
    assert u.column_result["radiation"]["do_rad"]
    for report in (r, s, t, u):
        torch.testing.assert_close(
            report.energy_changes["radiation_flux_residual"] / 60,
            torch.zeros_like(report.energy_changes["radiation_flux_residual"]),
            rtol=0, atol=1e-8,
        )


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({}, FileNotFoundError, "aerosol.optical.dat"),
        ({"do_swaerosol": False}, NotImplementedError, "both the SW and the LW"),
        ({"aerosol_data_set": "   "}, NotImplementedError, "shettle_fenn"),
        (
            {"optical_filename": "custom.optical.dat"},
            FileNotFoundError,
            "custom.optical.dat",
        ),
    ],
)
def test_radiation_setup_selects_supported_aerosols_before_loading_tables(
    tmp_path, overrides, error, match
):
    from workflows.pyam21.setup import example
    from pyam21.radiation.aerosol import AerosolOptics, SULFATE_INDICES, _TARGET
    from pyam21.radiation.sea_esf import _check_config

    # Parse unchanged upstream namelist sections. The actual aerosol constructor
    # must reach its missing optical file, beyond the package's shallower check.
    fixtures = Path(__file__).parent / "fixtures/pyam21"
    for case, name in (
        ("gcss_bomex", "bomex-radiation.nml"),
        ("gcss_astex", "astex-aerosol.nml"),
    ):
        target = tmp_path / "reference/input" / case / "input.nml"
        target.parent.mkdir(parents=True)
        shutil.copyfile(fixtures / name, target)

    def inspect(nml, *args, **kwargs):
        _check_config(nml)
        assert nml["physics_driver_nml"]["do_radiation"]
        species = nml["aerosol_nml"]["data_names"]
        # No fabricated optical data: real file I/O intentionally stops here.
        try:
            AerosolOptics(nml, str(tmp_path), None, species)
        except FileNotFoundError:
            names = nml["aerosolrad_package_nml"]["aerosol_optical_names"]
            required = {f"sulfate_{v}%" for v in SULFATE_INDICES}
            required.update(
                _TARGET.get(n, n) for n in species if not n.startswith("so4")
            )
            assert required <= set(names)
            raise

    with (
        patch("workflows.pyam21.radiation.build_radiation", side_effect=inspect),
        pytest.raises(error, match=match),
    ):
        example(
            tmp_path,
            cells=16,
            levels=12,
            radiation_data=tmp_path,
            namelist_overrides={"aerosolrad_package_nml": overrides},
        )


def test_shared_exchange_retains_deliberate_upstream_heat_defect(model):
    adapter, air = initial(model)
    receiver, water = surface_case(model)
    A = model.mesh.cell_area
    original = adapter.column_call

    def defective(*args, **kwargs):
        out, surf, mem, rad = original(*args, **kwargs)
        out = {
            **out,
            "tdt": out["tdt"] + 0.01,
            "endpoint_temperature": out["endpoint_temperature"] + 0.6,
        }
        return out, surf, mem, rad

    E0 = model.diagnostics(air.atmosphere)["energy"] + receiver.H.total_energy(
        water.coupled
    )
    with patch.object(adapter, "column_call", side_effect=defective):
        end, sea, x, ar, sr = exchange_step(
            adapter,
            air,
            receiver,
            water,
            60,
            ice_temperature=torch.full_like(water.snow_mass, 270.0),
        )
    defect = ar.energy_changes["am2_energy_residual"]
    assert (defect > 1000).all()
    E1 = model.diagnostics(end.atmosphere)["energy"] + receiver.H.total_energy(
        sea.coupled
    )
    assert abs(float((E1 - E0 - (A * (x.toa + defect)).sum()) / A.sum())) < 2e-6


@pytest.mark.parametrize("dt", [0, -1, 1.5, float("nan"), True])
def test_invalid_exchange_interval_rejected_before_physics(model, dt):
    adapter, air = initial(model)
    receiver, water = surface_case(model)
    with patch.object(
        adapter, "column_call", side_effect=AssertionError("must not run")
    ):
        with pytest.raises(ValueError, match="interval"):
            exchange_step(
                adapter,
                air,
                receiver,
                water,
                dt,
                ice_temperature=torch.full_like(water.snow_mass, 270.0),
            )


def test_mismatched_shared_water_record_rejected(model):
    adapter, air = initial(model)
    receiver, water = surface_case(model)
    tiles = receiver.tiles(water, torch.full_like(water.snow_mass, 270.0))
    air = prepare_surface(air, tiles)
    c, g = adapter.mapping.columns(air.atmosphere, air.cloud_fraction)
    out, surf, mem, rad = adapter.column_call(air, 60, c, g, surface_tiles=tiles)
    x = Exchange.from_am2(adapter, air, 60, c, out, tiles)
    out["exchange"] = replace(x, rain=x.rain + 0.1)
    with pytest.raises(ValueError, match="disagrees"):
        adapter.apply_result(air, 60, c, g, out, surf, mem, rad)


def test_shared_integrated_flux_accepts_upstream_tile_roundoff():
    """A valid rate comparison must survive integration and tile summation."""
    from workflows.pyam21.energy import _check_shared_flux

    dtype = torch.float64
    eps = torch.finfo(dtype).eps
    dt = 1200
    # Opposite-sign tile rates almost cancel. The aggregate rate can differ
    # from the tile sum within the 256-eps envelope allowed by column_call.
    rate = torch.tensor([[2.0, -2.0, 0.0]], dtype=dtype)
    recorded = (dt * rate).sum(-1)
    aggregate = rate.sum(-1) + 64 * eps
    actual = dt * aggregate
    _check_shared_flux(actual, recorded, field="vapor", dt=dt, tile_rates=rate)

    # The same checker must still reject a real disagreement in the flux.
    with pytest.raises(ValueError, match=r"field=vapor cell=0.*absolute_error"):
        _check_shared_flux(
            actual + 1.0e-5, recorded, field="vapor", dt=dt, tile_rates=rate
        )


def test_shared_precipitation_record_keeps_strict_tolerance():
    from workflows.pyam21.energy import _check_shared_flux

    rain = torch.tensor([0.2, 0.0], dtype=torch.float64)
    _check_shared_flux(rain, rain, field="rain", dt=1200)
    with pytest.raises(ValueError, match=r"field=rain cell=0"):
        _check_shared_flux(
            rain + torch.tensor([1e-7, 0.0]), rain, field="rain", dt=1200
        )
