"""Optional SSPRK(4,3) dynamics: tableau, positivity, and real ocean/ice tests."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / p) for p in (
    "sea_ice", "temperature_coordinate", "ocean_column"
)]

from explicit_rk import ssprk43_dynamics
from snow import SnowOceanIce
from test_ocean_ice_coupling import setup, ridge_mesh


class LinearTestDynamics:
    """Scalar exponential test with stage-dependent boundary wind work."""

    def __init__(self, rate):
        self.rate = rate
        self.calls = []

    def tendency(self, value, dt, **options):
        self.calls.append(dt)
        return self.rate * value, SimpleNamespace(wind_power=value)

    def add(self, state, rhs, dt):
        return state + dt * rhs

    def blend(self, first, second, weight):
        return (1 - weight) * first + weight * second

    def check(self, state):
        assert torch.isfinite(state).all()


def test_ssprk43_four_half_euler_stages_stability_polynomial_and_wind_weights():
    model = LinearTestDynamics(0.7)
    y0 = torch.tensor(1.3, dtype=torch.float64)
    dt = 0.4
    actual, work = ssprk43_dynamics(model, y0, dt)
    z = 0.7 * dt
    stability = 1 + z + z*z/2 + z**3/6 + z**4/48
    torch.testing.assert_close(actual, y0 * stability, rtol=2e-15, atol=0)
    assert model.calls == [dt/2]*4

    # Reconstruct stage states independently and check work quadrature.
    h = dt/2
    y1 = y0*(1 + 0.7*h)
    y2 = y1*(1 + 0.7*h)
    y3 = (2/3)*y0 + (1/3)*y2*(1 + 0.7*h)
    expected = dt*(y0/6 + y1/6 + y2/6 + y3/2)
    torch.testing.assert_close(work, expected, rtol=2e-15, atol=0)


def test_ssprk43_imaginary_axis_interval_exceeds_ssprk3():
    # SSPRK(4,3): |R(i*x)|^2-1 =
    # x^4 (x^4+16*x^2-96)/2304.
    x = (-8 + 4 * 10**0.5)**0.5
    assert 2.15 < x < 2.16
    # SSPRK(3,3) interval is sqrt(3).
    assert x / (3**0.5) > 1.24


@pytest.mark.parametrize("snow", [False, True])
def test_ocean_ice_ssprk43_real_state_preserves_inventory_and_admissibility(
    ridge_mesh, snow
):
    coupled, state = setup(ridge_mesh)
    if snow:
        model = SnowOceanIce(coupled)
        state = model.bare(state)
    else:
        model = coupled
    model.check(state)
    before = state.coupled if snow else state
    old = before.ocean.ocean
    area = ridge_mesh.cell_area[:, None]
    old_water = (area * old.h).sum()
    old_salt = (area * old.salt_h).sum()
    old_ice = (area * before.ice.volume).sum()
    old_snow = (area * state.snow_mass).sum() if snow else None
    end, work = model.dynamics_step(state, 0.1, integrator="ssprk43")
    model.check(end)
    result = end.coupled if snow else end
    new = result.ocean.ocean
    torch.testing.assert_close((area * new.h).sum(), old_water, rtol=2e-12, atol=1)
    torch.testing.assert_close((area * new.salt_h).sum(), old_salt, rtol=2e-12, atol=1)
    torch.testing.assert_close((area * result.ice.volume).sum(), old_ice, rtol=2e-12, atol=1)
    if snow:
        torch.testing.assert_close(
            (area * end.snow_mass).sum(), old_snow, rtol=2e-12, atol=1
        )
    assert (new.h > 0).all()
    assert (new.salt_h >= 0).all()
    assert (result.ice.volume >= 0).all()
    assert torch.isfinite(work).all()


@pytest.mark.parametrize("snow", [False, True])
def test_default_ssprk3_is_identical_to_explicit_selector(ridge_mesh, snow):
    coupled, original = setup(ridge_mesh)
    if snow:
        model = SnowOceanIce(coupled)
        state = model.bare(original)
    else:
        model, state = coupled, original
    a, wa = model.dynamics_step(state, 0.1)
    b, wb = model.dynamics_step(state, 0.1, integrator="ssprk3")
    assert torch.equal(wa, wb)
    if snow:
        assert torch.equal(a.snow_mass, b.snow_mass)
        a, b = a.coupled, b.coupled
    for name in ("u_e", "h", "temp_h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(a.ocean.ocean, name), getattr(b.ocean.ocean, name))
    for name in ("area", "volume", "velocity"):
        assert torch.equal(getattr(a.ice, name), getattr(b.ice, name))


def test_rejects_unknown_ocean_ice_tableau(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    with pytest.raises(ValueError, match="integrator"):
        coupled.dynamics_step(state, 0.1, integrator="bad")
