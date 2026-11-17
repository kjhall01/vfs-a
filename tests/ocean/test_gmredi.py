"""Conservation, geometry and independent work checks for research eddy transport."""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate"),
    str(Path(__file__).resolve().parents[2] / "validation/ocean_column"),
]
from roquet import RoquetOcean
from eddy import GMRedi
from eddy_energy import EddyState
from vfs.ocean.stratified import TKEState
from test_jmd95_hamiltonian import mesh, case


def setup(mesh, coast=False):
    old, s = case(mesh, coast)
    model = RoquetOcean(mesh, old.column, domain=old.domain)
    k = s.h * 1e-4
    if coast:
        k = old.domain.apply_cell(k)
    return model, model.with_eddy_reservoir(TKEState(s, k), 0.1)


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize(
    "coefficients", [(1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)]
)
def test_conservation_closed_circulation_and_variance(mesh, coast, coefficients):
    m, initial = setup(mesh, coast)
    eddy = GMRedi(*coefficients)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward autograd")):
        end, r = eddy.apply(m, initial, 3600.0)
    s, t = initial.ocean, end.ocean
    eps = torch.finfo(s.h.dtype).eps
    assert torch.equal(s.h, t.h) and torch.equal(s.u_e, t.u_e)
    assert torch.count_nonzero(r.streamfunction[:, [0, -1]]) == 0
    assert torch.count_nonzero(r.vertical_transport[:, [0, -1]]) == 0
    transport_scale = (
        r.horizontal_transport.abs().max() + r.vertical_transport.abs().max()
    )
    assert r.volume_residual.abs().max() <= 32 * eps * transport_scale
    for a, b in [
        (s.temp_h, t.temp_h),
        (s.salt_h, t.salt_h),
        (s.tracer_h[..., 0], t.tracer_h[..., 0]),
        (initial.tke_h, end.tke_h),
    ]:
        volume = m.energy.mass_area(s)
        assert (volume * (b - a)).sum().abs() <= 64 * eps * (volume * a.abs()).sum()
        if coast:
            assert torch.equal(a[~m.ocean.domain.wet_cell], b[~m.ocean.domain.wet_cell])
    torch.testing.assert_close(t.tracer_h, s.tracer_h, rtol=0, atol=2e-12)
    concentrations = torch.stack(
        (
            s.temperature(),
            s.salinity(),
            s.tracer_h[..., 0] / s.h,
            initial.tke_h / s.h,
            initial.eddy_h / s.h,
        ),
        -1,
    )
    norm = (
        mesh.cell_area[:, None, None] * s.h[..., None] * concentrations.square()
    ).sum((0, 1))
    assert (r.redi_variance_change <= 64 * eps * norm).all()
    # Independent direct Hamiltonian subtraction, tolerance based on its large datum.
    expected = m.total_energy(end) - m.total_energy(initial)
    assert abs(r.energy_change - expected) <= 32 * eps * m.total_energy(initial).abs()
    resolved_work = m.hamiltonian.finite_change(initial, end, include_eddy=False)
    eddy_work = m.energy.mass_area(s) * (end.eddy_h - initial.eddy_h)
    scale = resolved_work.abs().sum() + eddy_work.abs().sum()
    assert (
        abs(
            r.energy_change
            - r.gm_energy_change
            - r.redi_energy_change
            - r.reservoir_change
        )
        <= 128 * eps * scale
    )
    assert abs(r.energy_change) <= 128 * eps * (
        scale + (m.energy.mass_area(s) * initial.eddy_h).sum()
    )


def test_vertical_derivative_on_uneven_layers():
    z = torch.tensor([[-1.0, -4.0, -12.0, -30.0]], dtype=torch.float64)
    ids, weights = GMRedi._vertical_rows(z)
    torch.testing.assert_close((weights * z.flatten()[ids]).sum(-1), torch.ones_like(z))
    torch.testing.assert_close(weights.sum(-1), torch.zeros_like(z), rtol=0, atol=1e-16)


def test_density_tangent_nullspace_and_tilted_grid_metric(mesh):
    m, state = setup(mesh)
    s = state.ocean
    z = m.ocean.energy.base.height(s)
    # Constant CT and SA linear in physical height: horizontally flat density,
    # despite the unequal and horizontally varying layer-center heights.
    ct = torch.full_like(z, 10.0)
    sa = 35.0 - 0.001 * z
    s = s.with_tensors(s.u_e, s.h, s.h * ct, s.h * sa, s.tracer_h)
    state = EddyState(s, state.tke_h, state.eddy_h)
    eddy = GMRedi(1000.0, 1000.0)
    slope, _, _, _, _, _, _, operator, _ = eddy._geometry(m, state)
    assert slope.abs().max() < 1e-16
    torch.testing.assert_close(
        operator.apply(sa[..., None])[..., 0],
        torch.zeros_like(slope),
        rtol=0,
        atol=1e-19,
    )
    # With a lateral salinity gradient, the unclipped slope still annihilates
    # the reference buoyancy itself (discrete along-density derivative).
    sa = sa + 0.1 * mesh.cell_xyz[:, :1]
    state = EddyState(
        s.with_tensors(s.u_e, s.h, s.h * ct, s.h * sa, s.tracer_h),
        state.tke_h,
        state.eddy_h,
    )
    geom = eddy._geometry(m, state)
    operator = geom[-2]
    b = (
        -s.column.gravity
        / s.column.rho0
        * m.config.anomaly(ct, sa, torch.zeros_like(z))
    )
    assert geom[0].abs().max() > 1e-8
    assert operator.apply(b[..., None]).abs().max() < 1e-22


def test_disabled_identity_and_failure_guards(mesh):
    m, state = setup(mesh)
    end, r = GMRedi().apply(m, state, 1.0)
    for name in ("u_e", "h", "temp_h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(state.ocean, name), getattr(end.ocean, name))
    assert r.energy_change == 0
    for values in [(-1.0, 0.0), (float("nan"), 0.0)]:
        with pytest.raises(ValueError):
            GMRedi(*values)
    with pytest.raises(ValueError):
        GMRedi().apply(m, state, 0.0)
    with pytest.raises(FloatingPointError, match="GM outflow"):
        GMRedi(1e20, 0.0).apply(m, state, 1.0)
    with pytest.raises(FloatingPointError, match="Redi explicit"):
        GMRedi(0.0, 1e20).apply(m, state, 1.0)


def test_weak_stratification_no_transport(mesh):
    m, state = setup(mesh)
    s = state.ocean
    # Vertically uniform buoyancy, with lateral gradients: no finite slope.
    ct = (10.0 + mesh.cell_xyz[:, :1]).expand_as(s.h)
    sa = torch.full_like(s.h, 35.0)
    state = EddyState(
        s.with_tensors(s.u_e, s.h, s.h * ct, s.h * sa, s.tracer_h),
        state.tke_h,
        state.eddy_h,
    )
    end, r = GMRedi(1000.0, 1000.0).apply(m, state, 60.0)
    assert torch.count_nonzero(r.slope) == 0
    assert torch.equal(end.ocean.temp_h, state.ocean.temp_h)


def test_combines_with_resolved_step(mesh):
    m, state = setup(mesh)
    eddy = GMRedi(1000.0, 1000.0)
    for _ in range(3):
        state, _ = m.step_with_report(state, 60.0)
        state, report = eddy.apply(m, state, 60.0)
    assert torch.isfinite(report.energy_change)


def test_rejects_redi_negative_tracer_without_mutating_input():
    from eddy_campaign import make

    torch.set_default_dtype(torch.float64)
    m, state = make()
    s = state.ocean
    tracer = torch.zeros_like(s.tracer_h)
    tracer[0, 3, 0] = s.h[0, 3]
    state = EddyState(
        s.with_tensors(s.u_e, s.h, s.temp_h, s.salt_h, tracer),
        state.tke_h,
        state.eddy_h,
    )
    before = tracer.clone()
    with pytest.raises(FloatingPointError, match="not positivity preserving"):
        GMRedi(0.0, 1000.0).apply(m, state, 1.0)
    assert torch.equal(state.ocean.tracer_h, before)


def test_positive_redi_transports_sharp_stores_without_clipping():
    from eddy_campaign import make

    torch.set_default_dtype(torch.float64)
    m, state = make()
    s = state.ocean
    tracer = torch.zeros_like(s.tracer_h)
    tracer[0, 3, 0] = s.h[0, 3]
    k = torch.full_like(state.tke_h, 1e-20)
    k[0, 3] = s.h[0, 3] * 1e-4
    state = EddyState(
        s.with_tensors(s.u_e, s.h, s.temp_h, s.salt_h, tracer), k, state.eddy_h
    )
    before = tracer.clone()
    with pytest.raises(FloatingPointError, match="not positivity preserving"):
        GMRedi(0.0, 1000.0).apply(m, state, 1.0)
    end, r = GMRedi(0.0, 1000.0, positive_redi=True).apply(m, state, 1.0)
    assert r.redi_limited_fluxes > 0
    assert (end.tke_h >= 0).all() and (end.ocean.tracer_h >= 0).all()
    assert (end.eddy_h >= 0).all() and torch.equal(tracer, before)
    # Independent global inventories and full-H subtraction.
    mass = m.energy.mass_area(s)
    for left, right in (
        (k, end.tke_h),
        (tracer[..., 0], end.ocean.tracer_h[..., 0]),
        (s.salt_h, end.ocean.salt_h),
    ):
        torch.testing.assert_close(
            (mass * left).sum(), (mass * right).sum(), rtol=2e-14, atol=0
        )
    torch.testing.assert_close(
        m.total_energy(state), m.total_energy(end), rtol=2e-14, atol=0
    )
    q = GMRedi._contents(state) / s.h[..., None]
    norm = (s.mesh.cell_area[:, None, None] * s.h[..., None] * q.square()).sum((0, 1))
    assert (r.redi_variance_change <= 64 * torch.finfo(k.dtype).eps * norm).all()
    scale = m.hamiltonian.finite_change(state, end, include_eddy=False).abs().sum()
    scale += (mass * state.eddy_h).sum()
    assert (
        r.redi_local_energy_residual.abs().sum()
        <= 256 * torch.finfo(k.dtype).eps * scale
    )


@pytest.mark.parametrize("coast", [False, True])
def test_positive_row_fluxes_have_local_funding_and_conserve_each_field(mesh, coast):
    m, state = setup(mesh, coast)
    op = GMRedi(0.0, 1000.0)
    *_, operator, measure = op._geometry(m, state)
    shape = (*state.ocean.h.shape, 3)
    generator = torch.Generator().manual_seed(87)
    amounts = torch.rand(shape, generator=generator, dtype=state.ocean.h.dtype)
    amounts[::3, 1, 0] = 0
    transfer = 1e6 * torch.randn(
        (*measure.shape, 3), generator=generator, dtype=amounts.dtype
    )
    alpha = operator.positive_flux_scale(transfer, amounts)
    assert ((alpha >= 0) & (alpha <= 1)).all() and (alpha < 1).any()
    change = operator.transpose(alpha * transfer)
    assert (amounts + change >= 0).all()
    torch.testing.assert_close(
        change.sum((0, 1)), torch.zeros(3, dtype=amounts.dtype), rtol=0, atol=1e-12
    )
    # A smooth positive field with a tiny flux is exactly unchanged by the limiter.
    ones = operator.positive_flux_scale(transfer * 1e-30, torch.ones_like(amounts))
    assert torch.equal(ones, torch.ones_like(ones))


def test_positive_redi_is_exactly_original_when_unlimited(mesh):
    m, state = setup(mesh)
    original, old = GMRedi(100.0, 100.0).apply(m, state, 1.0)
    enabled, new = GMRedi(100.0, 100.0, positive_redi=True).apply(m, state, 1.0)
    assert new.redi_limited_fluxes == 0
    for name in ("temp_h", "salt_h", "tracer_h", "h", "u_e"):
        assert torch.equal(getattr(original.ocean, name), getattr(enabled.ocean, name))
    assert torch.equal(original.tke_h, enabled.tke_h)
    assert torch.equal(original.eddy_h, enabled.eddy_h)


def test_finite_energy_work_against_independent_midpoint_gradient(mesh):
    m, state = setup(mesh)
    end, report = GMRedi(1000.0, 1000.0).apply(m, state, 3600.0)
    s, t = state.ocean, end.ocean
    # H is quadratic in CT, linear in SA and TKE at fixed h/u. Its midpoint
    # derivative integrates the finite step exactly, including cross-stage work.
    th = ((s.temp_h + t.temp_h) / 2).detach().requires_grad_()
    sh = ((s.salt_h + t.salt_h) / 2).detach().requires_grad_()
    kh = ((state.tke_h + end.tke_h) / 2).detach().requires_grad_()
    eh = ((state.eddy_h + end.eddy_h) / 2).detach().requires_grad_()
    mid = EddyState(s.with_tensors(s.u_e, s.h, th, sh, s.tracer_h), kh, eh)
    gradients = torch.autograd.grad(m.total_energy(mid), (th, sh, kh, eh))
    work = [
        g * d
        for g, d in zip(
            gradients,
            (
                t.temp_h - s.temp_h,
                t.salt_h - s.salt_h,
                end.tke_h - state.tke_h,
                end.eddy_h - state.eddy_h,
            ),
        )
    ]
    expected = sum(w.sum() for w in work)
    scale = sum(w.abs().sum() for w in work)
    assert abs(report.energy_change - expected) < 64 * torch.finfo(th.dtype).eps * scale


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("fields", [1, 5])
def test_factored_redi_matches_explicit_stencil_and_adjoint(mesh, coast, fields):
    """Compare against the original eight-entry operator, including its bound."""
    m, state = setup(mesh, coast)
    eddy = GMRedi(0.0, 1000.0)
    metrics = eddy._metrics(m, state)
    z, vertical_ids, vertical_weights, i, j, dc, _, ids, zn = metrics
    slope, *_, operator, measure = eddy._geometry(m, state, metrics=metrics)
    relative = slope - zn
    rows = torch.cat(
        (ids[i][..., None], ids[j][..., None], vertical_ids[i], vertical_ids[j]), -1
    ).reshape(-1, 8)
    weights = torch.cat(
        (
            -torch.ones_like(z[i])[..., None] / dc[..., None],
            torch.ones_like(z[j])[..., None] / dc[..., None],
            0.5 * relative[..., None] * vertical_weights[i],
            0.5 * relative[..., None] * vertical_weights[j],
        ),
        -1,
    ).reshape(-1, 8)
    generator = torch.Generator().manual_seed(902)
    values = torch.randn((*z.shape, fields), generator=generator, dtype=z.dtype)
    flux = torch.randn((*slope.shape, fields), generator=generator, dtype=z.dtype)
    explicit = (weights[..., None] * values.reshape(-1, fields)[rows]).sum(1)
    torch.testing.assert_close(
        operator.apply(values).reshape_as(explicit), explicit, rtol=5e-13, atol=1e-18
    )
    transpose = (
        torch.zeros_like(values)
        .reshape(-1, fields)
        .index_add(
            0,
            rows.flatten(),
            (weights[..., None] * flux.reshape(-1, fields)[:, None]).reshape(
                -1, fields
            ),
        )
    )
    torch.testing.assert_close(
        operator.transpose(flux).reshape_as(transpose),
        transpose,
        rtol=5e-13,
        atol=1e-18,
    )
    bound = (
        torch.zeros_like(z)
        .flatten()
        .index_add(
            0,
            rows.flatten(),
            (
                weights.abs()
                * measure.flatten()[:, None]
                * weights.abs().sum(1)[:, None]
            ).flatten(),
        )
    )
    torch.testing.assert_close(
        operator.step_bound(measure).flatten(), bound, rtol=5e-13, atol=1e-18
    )
    torch.testing.assert_close(
        (operator.apply(values) * flux).sum(),
        (values * operator.transpose(flux)).sum(),
        rtol=5e-13,
        atol=1e-18,
    )
    # Closed-boundary constant nullspace and non-positive variance production.
    torch.testing.assert_close(
        operator.transpose(flux).sum((0, 1)),
        torch.zeros(fields, dtype=z.dtype),
        rtol=0,
        atol=1e-17,
    )
    gradient = operator.apply(values)
    tendency = operator.transpose(-measure[..., None] * gradient)
    torch.testing.assert_close(
        (values * tendency).sum(),
        -(measure[..., None] * gradient.square()).sum(),
        rtol=5e-13,
        atol=1e-17,
    )
