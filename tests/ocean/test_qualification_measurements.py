"""Check campaign references and instrumentation, not empirical accuracy gates."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def analysis():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path=Path(__file__).parents[2]/'tools/analyze_ocean_qualification.py'
    spec=importlib.util.spec_from_file_location('qualification_analysis_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def test_phase_instrument_recovers_resolved_harmonic(analysis):
    time=np.linspace(0,30,301);period=10.;ratio=.97
    phase=np.angle(np.exp(1j*2*np.pi*ratio*time/period))
    result=analysis.phase_metrics(time,phase,np.ones_like(time),period)
    assert result['frequency_over_reference']==pytest.approx(ratio,abs=1e-14)
    assert result['final_phase_error_rad']==pytest.approx(2*np.pi*(ratio-1)*3,abs=1e-14)
    assert result['envelope_min']==result['envelope_max']==1


def test_wave_initial_projection_and_linear_reference(analysis):
    driver=analysis.load_driver()
    model,state,energy,h0,mode,vector,omega,_=driver.wave_setup(32)
    area=model.mesh.cell_area[:,None]
    projection=(area*mode*(state.h-h0)*vector).sum()/((area*mode.square()).sum()*vector.square().sum())
    assert float(projection)==pytest.approx(.1,abs=1e-12)
    result=analysis.geometry_reference(driver,32,np.linspace(0,6*np.pi/omega,145))
    assert result['projected_amplitude_m'][0]==pytest.approx(.1,abs=1e-14)
    assert result['weighted_laplacian_symmetry_error']<1e-22
    assert result['minimum_eigenvalue']>-1e-22


@pytest.mark.parametrize('profile',['smooth','sharp'])
def test_prescribed_carrier_preserves_integrated_contents(analysis,profile):
    driver=analysis.load_driver();report={}
    args=SimpleNamespace(cells=16,profile=profile,circuits=.1,cfl=.2)
    with torch.no_grad(): driver.transport(args,report,lambda:None)
    for field in report['fields'].values():
        assert abs(field['inventory_change'])<1e-11
    assert report['fields']['passive']['minimum']>=0


def test_initial_balance_reference_reduction_and_rest(analysis):
    path=Path(__file__).parents[2]/'tools/probe_ocean_initial_balance.py'
    spec=importlib.util.spec_from_file_location('qualification_initial_balance_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    with torch.no_grad(): result=module.measure(32,2)
    # At reference composition the new active candidate must reduce to the
    # independent constant-density operator, even when that operator is not
    # in exact discrete balance. Do not assert an empirical jet accuracy gate.
    assert result['candidate_minus_constant_density_momentum_rms_m_s2']<1e-13
    assert result['candidate_minus_constant_density_h_rate_rms_m_s']<1e-13
    for rest in result['rest'].values():
        assert rest['h_rate_rms_m_s']==0
        assert rest['momentum_rms_m_s2']<1e-13
