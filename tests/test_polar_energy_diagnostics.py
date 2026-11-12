"""Offline analytic and bookkeeping checks for optional diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from vfs.geometry import EquiangularGrid


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'validation/polar_energy'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


polar=load('polar');mapping=load('mapping')


@pytest.mark.parametrize('mode',['x','z','xz'])
def test_closed_form_means_against_independent_quadrature(mode):
    edges=np.deg2rad(np.array([-90.,-89.,-43.,2.,58.,89.,90.]))
    lon=np.array([-.3,.2,1.1,2.5])
    truth=polar.q_average(mode,edges,lon)
    nodes,weights=np.polynomial.legendre.leggauss(24)
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        p=(a+b)/2+(b-a)/2*nodes
        for j,(c,d) in enumerate(zip(lon[:-1],lon[1:])):
            l=(c+d)/2+(d-c)/2*nodes
            w=weights[:,None]*weights[None,:]*np.cos(p[:,None])
            reference=(w*polar.q_point(mode,p[:,None],l[None,:])).sum()/w.sum()
            assert truth[i,j]==pytest.approx(reference,abs=3e-14)


@pytest.mark.parametrize('poles',[False,True])
@pytest.mark.parametrize('mode',['x','z','xz'])
@pytest.mark.parametrize('rotation',[False,True])
def test_exact_face_integrals_recover_control_volume_means(poles,mode,rotation):
    result=polar.measure(32,poles,mode,rotation)
    for key in ('exact_flux_div_vs_cell_mean','exact_circulation_curl_vs_dual_mean'):
        assert result[key]['global']['max']<2e-12


def test_last_ring_converges_but_is_not_second_order():
    values=[polar.measure(n,True,'x',True)['vfs_cell_curl_vs_cell_mean']['last_nonpolar_ring']['rms'] for n in (32,64,128)]
    assert all(1.8<a/b<2.2 for a,b in zip(values[:-1],values[1:]))


@pytest.mark.parametrize('poles',[False,True])
def test_analogous_kinetic_energy_velocity_derivative(poles):
    g=EquiangularGrid.from_coordinates(np.linspace(90,-90,9),np.arange(16)*22.5,radius=1.) if poles else EquiangularGrid(8,16,radius=1.)
    h=torch.full(g.shape,2.,dtype=torch.float64)
    u=torch.full(g.shape,.3,dtype=torch.float64,requires_grad=True)
    v=torch.full((g.nlat+1,g.nlon),-.2,dtype=torch.float64,requires_grad=True)
    k=mapping.face_kinetic(g,h,u,v,rho0=1.)
    du,dv=torch.autograd.grad(k,(u,v))
    # Quadratic homogeneity: velocity work is exactly twice kinetic energy.
    assert float(((u*du).sum()+(v*dv).sum()).detach())==pytest.approx(2*float(k.detach()),rel=2e-15)
    assert k>0
    assert torch.autograd.gradcheck(lambda a,b:mapping.face_kinetic(g,h,a,b,rho0=1.),(u,v))


def test_mapping_inventory_and_energy_bookkeeping():
    r=mapping.measure(32,8,True)
    assert r['constant_tracer_max_error']<3e-16
    assert r['tracer_range'][0]>=.2-3e-16 and r['tracer_range'][1]<=1.+3e-16
    assert r['energy_density_transfer_defect_J']+r['energy_reconstruction_defect_J']==pytest.approx(r['total_energy_change_J'],rel=2e-14)
