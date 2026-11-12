"""Dataset layouts, polar topology and manufactured collocated diagnostics."""
import math
import pytest
import torch
from vfs.geometry import EquiangularGrid


def coordinates(n=16, poles=False, descending=False, offset=-180.):
    lat = torch.linspace(-90.,90.,n+1,dtype=torch.float64) if poles else torch.linspace(-90.+90/n,90.-90/n,n,dtype=torch.float64)
    if descending:
        lat = lat.flip(0)
    lon = (torch.arange(2*n,dtype=torch.float64)*180/n+offset+180)%360-180
    return lat,lon


@pytest.mark.parametrize('poles',[False,True])
@pytest.mark.parametrize('descending',[False,True])
def test_coordinate_geometry_identities(poles,descending):
    lat,lon=coordinates(poles=poles,descending=descending,offset=117.)
    g=EquiangularGrid.from_coordinates(lat,lon,radius=1.)
    torch.testing.assert_close(g.lat_deg,lat)
    torch.testing.assert_close(g.lon_deg,lon)
    assert g.pole_centered==poles
    torch.testing.assert_close(g.cell_area.sum(),torch.tensor(4*math.pi,dtype=lat.dtype))
    torch.testing.assert_close(g.corner_area.sum(),g.cell_area.sum())
    rng=torch.Generator().manual_seed(918)
    q=torch.randn(2,*g.shape,dtype=lat.dtype,generator=rng)
    e=torch.randn(q.shape,dtype=lat.dtype,generator=rng)
    n=torch.randn(2,g.nlat+1,g.nlon,dtype=lat.dtype,generator=rng)
    ge,gn=g.grad(q)
    torch.testing.assert_close(g.integral(q*g.div(e,n)),-g.face_inner(ge,gn,e,n),atol=2e-12,rtol=2e-14)
    torch.testing.assert_close(g.integral(g.div(e,n)),torch.zeros(2,dtype=lat.dtype),atol=2e-13,rtol=0)
    torch.testing.assert_close(g.curl(ge,gn)*g.corner_area,torch.zeros_like(n),atol=4e-15,rtol=0)
    c=g.curl(e,n)
    torch.testing.assert_close(g.corner_integral(c),torch.zeros(2,dtype=lat.dtype),atol=1e-13,rtol=0)
    torch.testing.assert_close(g.integral(g.corners_to_cells(n)),g.corner_integral(n),atol=1e-13,rtol=0)
    if poles:
        assert torch.count_nonzero(ge[..., [0,-1],:])==0
        assert torch.count_nonzero(g.corner_area[[0,-1]])==0
        div=g.div(e,n)
        torch.testing.assert_close(div[...,0,:],div[...,0,:1].expand_as(div[...,0,:]))
        # Arbitrary duplicate pole samples represent their mean, not nlon DOFs.
        torch.testing.assert_close(g.laplacian(q),g.laplacian(g._polar_mean(q)),rtol=0,atol=0)


@pytest.mark.parametrize('poles',[False,True])
def test_axis_reversal_and_longitude_roll(poles):
    lat,lon=coordinates(poles=poles,offset=15.)
    g=EquiangularGrid.from_coordinates(lat,lon,radius=2.)
    rev=EquiangularGrid.from_coordinates(lat.flip(0),lon.roll(7),radius=2.)
    q=torch.randn(*g.shape,dtype=lat.dtype)
    u,v=torch.randn(2,*g.shape,dtype=lat.dtype)
    reorder=lambda x:x.flip(-2).roll(7,-1)
    for fn,args in [('laplacian',(q,)),('div_from_collocated',(u,v)),('curl_from_collocated',(u,v))]:
        torch.testing.assert_close(getattr(rev,fn)(*(reorder(x) for x in args)),reorder(getattr(g,fn)(*args)),atol=1e-12,rtol=1e-13)
    for a,b in zip(rev.grad(reorder(q)),g.grad(q)):
        torch.testing.assert_close(a,reorder(b),atol=1e-13,rtol=1e-13)


def test_bounds_and_invalid_coordinates():
    lat,lon=coordinates(descending=True,poles=True)
    g=EquiangularGrid.from_coordinates(lat,lon)
    pairs=lambda b:torch.stack((b[:-1],b[1:]),-1)
    other=EquiangularGrid.from_coordinates(lat,lon,lat_bounds=pairs(g.lat_bounds_deg),lon_bounds=pairs(g.lon_bounds_deg)%360)
    torch.testing.assert_close(other.cell_area,g.cell_area,rtol=0,atol=0)
    for a,b in [(lat[1:],lon),(lat,torch.cat((lon,lon[:1]))),(lat,lon.flip(0)),(lat,lon+torch.arange(len(lon))*.01),(lat*1.01,lon),(lat,lon*float('nan'))]:
        with pytest.raises(ValueError):EquiangularGrid.from_coordinates(a,b)
    with pytest.raises(ValueError):EquiangularGrid.from_coordinates(lat,lon,lat_bounds=g.lat_bounds_deg+.1)
    with pytest.raises(ValueError):EquiangularGrid.from_coordinates(lat,lon,lon_bounds=g.lon_bounds_deg+.1)
    with pytest.raises(ValueError):EquiangularGrid.from_coordinates(lat,lon,radius=0.)


def test_era5_shape_polar_vector_and_autograd():
    # Actual 0.25-degree coordinate layout, without downloading a dataset.
    lat=torch.linspace(90.,-90.,721,dtype=torch.float64)
    lon=torch.arange(1440,dtype=lat.dtype)*.25
    g=EquiangularGrid.from_coordinates(lat,lon)
    assert g.shape==(721,1440)
    constant=torch.ones(g.shape,dtype=lat.dtype)
    torch.testing.assert_close(g.mean(constant),torch.tensor(1.,dtype=lat.dtype))
    assert torch.count_nonzero(g.laplacian(constant))==0
    small=EquiangularGrid.from_coordinates(*coordinates(4,True,True),radius=1.)
    p,l=map(torch.deg2rad,small.meshgrid())
    u,v=-p.sin()*l.cos(),l.sin().expand_as(p)
    e,n=small.wind_to_faces(u,v)
    # A smooth Cartesian wind has longitude-dependent polar components.
    torch.testing.assert_close(n[1],v[0])
    assert n[1].std()>0
    u=u.requires_grad_();v=v.clone().requires_grad_()
    assert torch.autograd.gradcheck(small.curl_from_collocated,(u,v))
    assert torch.autograd.gradcheck(small.div_from_collocated,(u,v))
    q=torch.randn_like(u,requires_grad=True)
    e,n=small.grad(q)
    derivative,=torch.autograd.grad(.5*small.face_inner(e,n,e,n),q)
    torch.testing.assert_close(derivative,-small.cell_area*small.laplacian(q),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('poles',[False,True])
@pytest.mark.parametrize('mode',['axial','tilted','degree2'])
def test_manufactured_collocated_refinement(poles,mode):
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('equigrid_qualification',Path(__file__).parents[1]/'tools/qualify_equiangular.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[module.measure(n,poles,mode) for n in (16,32,64)]
    for quantity in ['div','curl','stress_curl']:
        errors=[r[quantity]['rms'] for r in rows]
        assert errors[2]<errors[1]<errors[0],(mode,poles,quantity,errors)


@pytest.mark.parametrize('descending',[False,True])
def test_native_pole_layout_solid_rotation(descending):
    g=EquiangularGrid.from_coordinates(*coordinates(32,True,descending),radius=1.)
    lat,lon=map(torch.deg2rad,g.meshgrid())
    c=g.curl(*g.wind_to_faces(lat.cos(),torch.zeros_like(lat)))
    expected=torch.zeros_like(c)
    expected[1:-1]=(lat.sin()[1:]+lat.sin()[:-1])
    torch.testing.assert_close(c,expected,atol=2e-13,rtol=2e-13)


def test_radius_scaling_and_xarray_boundary():
    from pathlib import Path
    import importlib.util
    import xarray as xr
    lat,lon=coordinates(8,True,True)
    one=EquiangularGrid.from_coordinates(lat,lon,radius=1.)
    two=EquiangularGrid.from_coordinates(lat,lon,radius=2.)
    p,l=map(torch.deg2rad,one.meshgrid())
    u,v=-p.sin()*l.cos(),l.sin()
    torch.testing.assert_close(two.curl_from_collocated(u,v),one.curl_from_collocated(u,v)/2)
    torch.testing.assert_close(two.div_from_collocated(u,v),one.div_from_collocated(u,v)/2)
    torch.testing.assert_close(two.laplacian(u),one.laplacian(u)/4)
    torch.testing.assert_close(two.cell_area,4*one.cell_area)
    spec=importlib.util.spec_from_file_location('dataset_example',Path(__file__).parents[1]/'examples/equiangular_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=xr.DataArray(u.numpy()[None],dims=('time','latitude','longitude'),coords={'time':[0],'latitude':lat.numpy(),'longitude':lon.numpy()})
    b=xr.DataArray(v.numpy()[None],dims=a.dims,coords=a.coords)
    result,_=module.diagnose(a,b,radius=1.)
    assert result['curl'].dims==a.dims
    torch.testing.assert_close(torch.as_tensor(result['curl'].values)[0],one.curl_from_collocated(u,v))
    assert result['latitude'].equals(a['latitude'])
    with pytest.raises(ValueError):module.diagnose(a,b.assign_coords(longitude=b.longitude+1))


def test_coordinate_storage_is_owned():
    import numpy as np
    lat=np.linspace(-90.,90.,17)[::-1]  # Negative-stride NumPy input.
    lon=np.arange(32,dtype=np.float64)*360/32
    lon.setflags(write=False)
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('error')
        g=EquiangularGrid.from_coordinates(lat,lon)
    lat[0]=89.
    assert g.lat_deg[0]==90.
    original=torch.linspace(-90.,90.,17,dtype=torch.float64,requires_grad=True)
    other=EquiangularGrid.from_coordinates(original,lon)
    assert not other.lat_deg.requires_grad
    assert other.lat_deg.data_ptr()!=original.data_ptr()
