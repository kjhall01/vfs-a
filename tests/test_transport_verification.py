"""C5 infrastructure checks; numerical allowances are not scientific thresholds."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

spec = importlib.util.spec_from_file_location('c5', Path(__file__).resolve().parents[1]/'tools/run_transport_verification.py')
c5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c5)
r = c5.reference


def setup_module():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)


@pytest.mark.parametrize('flow', r.FLOWS)
def test_analytic_tangency_and_full_characteristic_return(flow):
    points = np.array([[1.,0.,0.],[.3,.4,np.sqrt(.75)],[-.8,.6,0.]])
    def rhs(t, flat):
        return r.velocity(flat.reshape(-1,3),t,flow,1.,1.).ravel()
    solution = solve_ivp(rhs,(0.,1.),points.ravel(),method='DOP853',rtol=1e-12,atol=1e-14,t_eval=[1.])
    assert solution.success
    np.testing.assert_allclose(solution.y[:,-1].reshape(-1,3),points,rtol=0,atol=5e-11)
    for time in (0.,.3,.5,1.):
        np.testing.assert_allclose((r.velocity(points,time,flow,1.,1.)*points).sum(-1),0,atol=2e-15)


def test_case4_background_and_time_reversal_are_not_negative_timesteps():
    point = np.array([[np.sqrt(.5),np.sqrt(.5),0.]])
    initial = r.velocity(point,0.,'NL2010_case4',1.,1.)
    final = r.velocity(point,1.,'NL2010_case4',1.,1.)
    # Deformation reverses, but the positive zonal background does not.
    assert initial[0,2] == pytest.approx(10.)
    assert final[0,2] == pytest.approx(-10.)
    np.testing.assert_allclose(initial[0,:2],final[0,:2],atol=2e-15)
    middle = r.velocity(point,.5,'NL2010_case4',1.,1.)
    assert np.linalg.norm(middle) == pytest.approx(2*np.pi)


def test_reference_quadrature_against_independent_gaussian_integral():
    model, _, _ = c5.make_case(16,'rotation_tilted','smooth',12)
    high, _ = r.cell_average(model.mesh,'rotation_tilted',0.,'smooth',12)
    low, _ = r.cell_average(model.mesh,'rotation_tilted',0.,'smooth',8)
    area = model.mesh.cell_area.numpy()
    exact_mean = .001*(1-np.exp(-20))/10
    assert abs(np.average(high[:,1],weights=area)-exact_mean) < abs(np.average(low[:,1],weights=area)-exact_mean)
    assert abs(np.average(high[:,1],weights=area)-exact_mean) < 2e-12
    points = np.array([[1.,0.,0.]])
    assert r.profile(points,'empty_condensate')[0,1] == 0
    assert r.profile(points,'empty_condensate')[0,0] > 0


@pytest.mark.parametrize('flow',r.FLOWS)
def test_production_constant_composition_and_prescribed_wind(flow):
    model,state,_ = c5.make_case(16,flow,'constant',12)
    time,dt = 123.,30.
    updated, rows, raw = c5.advance(model,state,time,dt,flow,raw=True)
    assert [row['time_s'] for row in rows] == [time,time+dt,time+dt/2]
    assert all(row['dt_s'] == dt for row in rows)
    assert all(row['constant_composition_error'] < 1e-13 for row in rows)
    assert all(row['exact_water_ledger'] for row in rows)
    assert all(row['max_discrete_velocity_divergence_per_s'] > 0 for row in rows)
    assert torch.equal(updated.u_e,c5.normal_velocity(model.mesh,time+dt,flow))
    assert not torch.equal(updated.ps,state.ps)  # No carrier reset.
    for row,stage in zip(rows,raw):
        assert torch.equal(stage['initial']['u_e'],c5.normal_velocity(model.mesh,row['time_s'],flow))
    with pytest.raises(ValueError,match='Positive'):
        c5.advance(model,state,time,-dt,flow)


def test_histogram_incidence_matches_small_oracle_and_detects_corruption():
    generator = torch.Generator().manual_seed(123)
    i = torch.randint(0,12,(37,),generator=generator)
    j = torch.randint(0,12,(37,),generator=generator)
    transfers = torch.randn(37,4,generator=generator)
    actual = c5.face_sum(i,j,transfers,12)
    expected = c5.energy.incident_sum(i,j,transfers,12)
    torch.testing.assert_close(actual,expected,rtol=0,atol=8e-15)
    broken = transfers.clone()
    broken[0,2] += 1.
    assert not torch.allclose(c5.face_sum(i,j,broken,12),expected)


def test_stage_audit_sensitivity_to_ledger_flux_and_limiter_corruption():
    model,state,_ = c5.make_case(16,'rotation_equatorial','smooth',12)
    updated,info = model.transport.euler_with_info(state,30.)
    with pytest.raises(ValueError,match='authoritative ledger'):
        c5.audit_stage(model,state,replace(updated,water_content=updated.water_content+1.),info,30.)
    with pytest.raises(ValueError,match='limiter'):
        c5.audit_stage(model,state,updated,{**info,'limited_faces':-1},30.)
    base = c5.audit_stage(model,state,updated,info,30.,oracle=True)
    flux = info['integrated_flux'].clone()
    flux[0,1] += 1e10
    bad = c5.audit_stage(model,state,updated,{**info,'integrated_flux':flux},30.)
    assert bad['species_incidence_max_error_kg'][1] > base['species_incidence_max_error_kg'][1] + 1e8
    assert bad['species_flux_sum_minus_carrier_max_kg'] > base['species_flux_sum_minus_carrier_max_kg'] + 1e8


def test_report_includes_bottom_interface_errors():
    model,state,_ = c5.make_case(16,'rotation_tilted','constant',12)
    exact,_ = r.cell_average(model.mesh,'rotation_tilted',0.,'constant',12)
    before = c5.snapshot(model,state,exact)
    water = state.water_content.clone()
    water[:,1,1] += 1.
    after = c5.snapshot(model,replace(state,water_content=water),exact)
    assert after['area_error_kg_per_kg']['L2'][1] > before['area_error_kg_per_kg']['L2'][1] + 1e-6
    assert after['per_interface_area_error'][0] == before['per_interface_area_error'][0]


def test_constant_composition_face_is_preserved_after_paired_fix():
    from vfs.atmosphere.gibbs_means import gibbs_face
    from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics
    thermo = MixtureThermodynamics()
    q = torch.tensor([[.9889,.01,.001,.0001]])
    temperature = torch.tensor([280.])
    _,face,_ = gibbs_face(thermo,temperature,torch.tensor([1e5]),q,
                        temperature,torch.tensor([9e4]),q)
    # Deliberate numerical change: #17's historical counterexample remains in
    # its committed report. Current production must preserve the endpoint q.
    torch.testing.assert_close(face,q,rtol=2e-14,atol=0)
