"""Offline tests of the optional campaign; no network or cloud imports."""
from pathlib import Path
import importlib.util
import argparse
import json
import numpy as np
import pytest
import torch
import xarray as xr
from vfs.geometry import EquiangularGrid

spec=importlib.util.spec_from_file_location('wb2_campaign',Path(__file__).parents[1]/'validation/equiangular_wb2/run.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def sample(poles=True,descending=True,n=16):
    lat=np.linspace(-90,90,n+1) if poles else np.linspace(-90+90/n,90-90/n,n)
    if descending:lat=lat[::-1].copy()
    lon=np.arange(2*n)*180/n
    g=EquiangularGrid.from_coordinates(lat,lon,radius=campaign.WB_RADIUS)
    p,l=map(np.deg2rad,map(np.asarray,g.meshgrid()))
    u=-np.sin(p)*np.cos(l)+np.cos(p);v=np.sin(l)+np.cos(p)
    return g,u,v,50000+1000*np.cos(p)*np.cos(l)


@pytest.mark.parametrize('poles',[False,True])
@pytest.mark.parametrize('descending',[False,True])
def test_independent_budgets_and_coordinates(poles,descending):
    g,u,v,phi=sample(poles,descending)
    result=campaign.budgets(g,u,v,phi)
    for quantity in ('divergence','curl'):
        assert all(r['relative']<2e-12 for r in result[quantity].values())
    campaign.invariance(g,u,v)
    with pytest.raises(AssertionError):campaign.closure(1.,-1.,2.)


def test_spherical_reference_distinguishes_curvature():
    errors=[]
    for size in (16,32,64):
        g,_,_,_=sample(n=size)
        lat,lon=map(np.deg2rad,map(np.asarray,g.meshgrid()))
        u=np.cos(lat);v=np.zeros_like(u)
        refs=campaign.references(g,u,v)
        expected=2*np.sin(lat)/g.radius
        errors.append(np.nanmax(np.abs(refs['spherical_fd'][1]-expected)))
        # WB2's flat-component formula gives approximately half the axial curl.
        interior=(np.abs(lat)>20*np.pi/180)&(np.abs(lat)<70*np.pi/180)
        np.testing.assert_allclose(refs['wb2_formula_replay'][1][interior],expected[interior]/2,rtol=.01)
    assert errors[2]<errors[1]<errors[0]


def test_area_overlap_and_polar_vector_basis():
    source,u,v,_=sample(n=64)
    target=EquiangularGrid.from_coordinates(np.linspace(-90,90,17),np.arange(32)*360/32,radius=source.radius)
    weights=campaign.overlap_weights(source,target)
    values=np.random.default_rng(1).normal(size=source.shape)
    coarse=campaign.average(weights,values)
    torch.testing.assert_close(target.integral(coarse),source.integral(campaign.tensor(values)),rtol=1e-12,atol=.1)
    torch.testing.assert_close(campaign.average(weights,np.ones(source.shape)),torch.ones(target.shape,dtype=torch.float64))
    # A constant Cartesian x direction projected on each source tangent plane
    # has one Cartesian cap mean, but longitude-dependent target components.
    p,l=map(np.deg2rad,map(np.asarray,source.meshgrid()))
    cu,cv=campaign.vector_average(source,target,weights,-np.sin(l),-np.sin(p)*np.cos(l))
    assert cu[0].std()>.5 and cv[0].std()>.5
    x=values.copy();x[0]=np.nan
    result=campaign.average(weights,x)
    assert torch.isnan(result[-1]).all()  # Source latitude descends, target ascends.
    assert torch.isfinite(result[1:-1]).all()


def test_local_snapshot_pipeline_and_missing_data(tmp_path):
    g,u,v,phi=sample()
    ds=xr.Dataset({name:(('latitude','longitude'),x) for name,x in [('u',u),('v',v),('phi',phi)]},coords={'latitude':g.lat_deg.numpy(),'longitude':g.lon_deg.numpy()})
    path=tmp_path/'local.nc';ds.to_netcdf(path)
    args=argparse.Namespace(input=str(path),weatherbench2=False,lat='latitude',lon='longitude',radius=g.radius,times=None,levels=[250,500],u='u',v='v',geopotential='phi',divergence='divergence',vorticity='vorticity',no_coarse=False,no_plots=True)
    with xr.open_dataset(path) as data:result=campaign.evaluate(data,args,tmp_path)
    assert len(result['cases'])==1
    row=json.loads((tmp_path/'case-01.json').read_text())
    assert row['comparisons']['spherical_fd']['curl']['below_85deg']['correlation']>.9
    assert 'archive' not in row['comparisons']
    ds.u[1,1]=np.nan
    with pytest.raises(ValueError,match='finite'):campaign.evaluate(ds,args,tmp_path)
