"""Small offline candidate tests, not activation or scientific qualification."""
import importlib.util
from pathlib import Path

import pytest
import torch
from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics

spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'tools/investigate_condensate_means.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_face_and_cp_ledger_probes():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rows=probe.measure()
    eps=torch.finfo(torch.float64).eps
    for row in rows:
        result=row['candidate']
        # Numerical roundoff regression allowance, not a scientific threshold.
        assert abs(result['residual_J_per_kg']) <= 256*eps*max(1,result['work_scale_J_per_kg'])
        assert max(result['symmetry_max']) == 0
        assert result['minimum_fraction'] >= 0
        assert abs(result['fraction_sum_error']) <= 4*eps
        assert result['cp_ledger']['minimum_amount'] >= 0
        if result['constant_trace_error'] is not None:
            assert result['constant_trace_error'] <= 8*eps
            assert result['cp_ledger']['constant_fraction_error'] <= 8*eps
    controls=[r['unpaired_species_only']['residual_J_per_kg'] for r in rows]
    assert max(map(abs,controls))>1  # Detect a species-only change.
    equal=rows[0]['candidate']
    thermo=MixtureThermodynamics()
    q=torch.tensor([.9889,.01,.001,.0001]);T=torch.tensor(280.);p=torch.tensor(1e5)
    torch.testing.assert_close(torch.tensor(equal['entropy']),thermo.entropy(T,p,q),rtol=2e-14,atol=1e-12)
    torch.testing.assert_close(torch.tensor(equal['specific_volume']),thermo.specific_volume(T,p,q),rtol=2e-14,atol=0)


@pytest.mark.parametrize('water_left,water_right',[
    ([.01,0.,0.],[.01,0.,0.]),
    ([.01,0.,0.],[.02,.002,.001]),
    ([.01,.001,0.],[.02,0.,.002]),
    ([.01,.001,.0001],[.02,.002,.001]),
])
def test_boundary_derivatives_are_finite_and_match_one_sided_differences(water_left,water_right):
    torch.set_default_dtype(torch.float64)
    thermo=MixtureThermodynamics()
    water=torch.tensor([water_left,water_right],requires_grad=True)
    def value(w):
        q=torch.cat((1-w.sum(-1,keepdim=True),w),-1)
        s,f,a=probe.candidate(thermo,torch.tensor(240.),torch.tensor(3e4),q[0],
                             torch.tensor(310.),torch.tensor(1e5),q[1])
        return s/1000+(f*f.new_tensor([1.,2.,3.,4.])).sum()+a
    objective=value(water)
    gradient,=torch.autograd.grad(objective,water)
    assert torch.isfinite(gradient).all()
    # Positive directions remain in the admissible composition domain at zero;
    # compare successive one-sided differences, not a negative-cloud probe.
    direction=torch.full_like(water,.001)
    exact=(gradient*direction).sum()
    differences=[(value(water+h*direction)-objective)/h for h in (1e-3,5e-4,2.5e-4)]
    error=(differences[-1]-exact).abs()
    truncation=(differences[-1]-differences[-2]).abs()
    assert error <= 2*truncation+128*torch.finfo(water.dtype).eps*abs(objective)/2.5e-4
