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


def test_curl_gradient_stokes_and_conservative_averaging():
    torch.manual_seed(17)
    g = EquiangularGrid(16, 32, radius=1.)
    q = torch.randn(2,16,32,dtype=torch.float64)
    # Test circulation residual, rather than magnifying roundoff by tiny areas.
    cg = g.curl(*g.grad(q))
    torch.testing.assert_close(cg*g.corner_area, torch.zeros_like(cg), atol=3e-15, rtol=0)
    east = torch.randn_like(q)
    north = torch.randn(2,17,32,dtype=q.dtype)
    c = g.curl(east,north)
    torch.testing.assert_close(g.corner_integral(c), torch.zeros(2,dtype=q.dtype), atol=1e-13, rtol=0)
    torch.testing.assert_close(g.corner_area.sum(), g.cell_area.sum())
    torch.testing.assert_close(c[...,0,:], c[...,0,:1].expand_as(c[...,0,:]))
    # Discrete Stokes on a zonally complete interior band.
    band = (c[...,3:9,:]*g.corner_area[3:9]).sum((-2,-1))
    boundary = (east[...,2,:]*g.east_distance[2]-east[...,8,:]*g.east_distance[8]).sum(-1)
    torch.testing.assert_close(band,boundary,atol=1e-13,rtol=1e-13)
    z = torch.randn_like(c)
    torch.testing.assert_close(g.integral(g.corners_to_cells(z)),g.corner_integral(z),atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(g.corners_to_cells(torch.ones_like(c)),torch.ones_like(q))


def test_curl_solid_body_polar_caps_and_collocated_refinement():
    errors=[]
    for size in (16,32,64):
        g=EquiangularGrid(size,2*size,radius=1.)
        lat,lon=map(torch.deg2rad,g.meshgrid())
        u=lat.cos()  # Unit angular velocity around the polar axis.
        c=g.curl(u,torch.zeros(size+1,2*size,dtype=u.dtype))
        sinlat=g.lat_deg.deg2rad().sin()
        bounds=torch.cat((sinlat.new_tensor([-1.]),sinlat,sinlat.new_tensor([1.])))
        exact=(bounds[1:]+bounds[:-1])[:,None].expand_as(c)
        # Exact dual-area average of 2 sin(latitude), including polar caps.
        torch.testing.assert_close(c,exact,atol=2e-12,rtol=2e-12)
        centered=g.curl_from_collocated(u,torch.zeros_like(u))
        errors.append(g.mean((centered-2*lat.sin()).square()).sqrt().item())
    assert errors[2]<errors[1]<errors[0], errors


def test_curl_autograd_and_validation():
    g=EquiangularGrid(3,6,radius=1.)
    e=torch.randn(3,6,dtype=torch.float64,requires_grad=True)
    n=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(g.curl,(e,n))
    with pytest.raises(ValueError): g.curl(e,n[None])
    with pytest.raises(ValueError): g.curl_from_collocated(e,e,location='bad')
    torch.testing.assert_close(g.curl_from_collocated(e,e,location='corner'),g.curl(*g.wind_to_faces(e,e)))


def test_tilted_rotation_curl_refinement():
    errors=[]
    for size in (16,32,64):
        g=EquiangularGrid(size,2*size,radius=1.)
        lat,lon=map(torch.deg2rad,g.meshgrid())
        # Solid-body rotation about x, with analytic radial curl 2 cos(lat) cos(lon).
        curl=g.curl_from_collocated(-lat.sin()*lon.cos(),lon.sin())
        errors.append(g.mean((curl-2*lat.cos()*lon.cos()).square()).sqrt().item())
        assert torch.isfinite(curl).all()
    assert errors[2]<errors[1]<errors[0], errors
