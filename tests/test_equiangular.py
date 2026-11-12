import math
import pytest
import torch
from vfs.geometry import EquiangularGrid
from vfs.geometry.remap import LatLonGrid


def test_geometry_and_legacy_compatibility():
    g = EquiangularGrid(18, 36, radius=2.)
    assert isinstance(g, LatLonGrid)
    torch.testing.assert_close(g.cell_area.sum(), torch.tensor(16*math.pi, dtype=torch.float64))
    assert torch.count_nonzero(g.north_face_length[[0, -1]]) == 0
    assert g.lat_deg.abs().max() < 90
    torch.testing.assert_close(g.mean(torch.ones(2, 18, 36, dtype=torch.float64)), torch.ones(2, dtype=torch.float64))


def test_flux_conservation_adjoint_and_dissipation():
    torch.manual_seed(13)
    g = EquiangularGrid(12, 24, radius=1.)
    q = torch.randn(3, 12, 24, dtype=torch.float64)
    e = torch.randn_like(q)
    n = torch.randn(3, 13, 24, dtype=torch.float64)
    torch.testing.assert_close(g.integral(g.div(e, n)), torch.zeros(3, dtype=q.dtype), atol=1e-13, rtol=0)
    ge, gn = g.grad(q)
    torch.testing.assert_close(g.integral(q*g.div(e,n)), -g.face_inner(ge,gn,e,n), atol=1e-12, rtol=1e-13)
    torch.testing.assert_close(g.integral(q*g.laplacian(q)), -g.face_inner(ge,gn,ge,gn))
    assert (g.integral(q*g.laplacian(q)) < 0).all()
    assert torch.count_nonzero(g.laplacian(torch.ones_like(q))) == 0


@pytest.mark.parametrize('mode', ['z', 'x', 'xy'])
def test_spherical_harmonic_refinement(mode):
    errors = []
    for size in (16, 32, 64):
        g = EquiangularGrid(size, 2*size, radius=1.)
        lat, lon = map(torch.deg2rad, g.meshgrid())
        q, eigenvalue = {'z': (lat.sin(), 2), 'x': (lat.cos()*lon.cos(), 2),
                        'xy': (lat.cos().square()*(2*lon).sin(), 6)}[mode]
        error = g.laplacian(q)+eigenvalue*q
        errors.append(g.mean(error.square()).sqrt().item())
        assert torch.isfinite(error).all()  # Includes polar rows.
    assert errors[2] < errors[1] < errors[0], errors


def test_autograd_metric_energy():
    g = EquiangularGrid(4, 8, radius=1.)
    q = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    e, n = g.grad(q)
    derivative, = torch.autograd.grad(.5*g.face_inner(e,n,e,n), q)
    torch.testing.assert_close(derivative, -g.cell_area*g.laplacian(q))
    assert torch.autograd.gradcheck(lambda x: g.laplacian(x), (q,))


def test_wind_adapter_and_validation():
    g = EquiangularGrid.regular(4, 8)
    u = torch.ones(4, 8, dtype=torch.float64)
    v = torch.zeros_like(u)
    assert torch.count_nonzero(g.div(*g.wind_to_faces(u,v))) == 0
    with pytest.raises(ValueError): g.grad(u.float())
    with pytest.raises(ValueError): g.grad(u.T)
    with pytest.raises(ValueError): g.div(u, torch.zeros(2,5,8,dtype=u.dtype))
    with pytest.raises(ValueError): EquiangularGrid(4, 8, radius=-1)
    with pytest.raises(ValueError): EquiangularGrid(4, 2)
    moved = g.to(dtype=torch.float32)
    assert moved.laplacian(u.float()).dtype == torch.float32
