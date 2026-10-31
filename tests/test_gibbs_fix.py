"""Current numerical behavior: paired faces and actual spherical bounded steps."""
import importlib.util
from pathlib import Path

import torch

spec=importlib.util.spec_from_file_location('fix',Path(__file__).resolve().parents[1]/'tools/check_gibbs_fix.py')
fix=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


def test_actual_gibbs_pairing_and_negative_control():
    torch.set_default_dtype(torch.float64)
    rows=fix.face_checks()
    eps=torch.finfo(torch.float64).eps
    for row in rows:
        assert abs(row['residual_J_per_kg'])<=256*eps*max(1.,row['work_scale_J_per_kg'])
        if row['constant_fraction_error'] is not None:
            # New face check accounts for subtraction in the unchanged exact
            # gas-logmean branch. Historical fixture tolerances are untouched.
            assert row['constant_fraction_error']<=32*eps*row['logmean_roundoff_condition']
    assert max(abs(x['corrupted_entropy_residual']['residual_J_per_kg']) for x in rows)>1.


def test_actual_euler_and_ssprk3_composition_ledgers_and_bounds():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rows,_=fix.model_checks()
    eps=torch.finfo(torch.float64).eps
    assert len(rows)==12
    active=False
    for row in rows:
        if row['final_constant_fraction_error'] is not None:
            assert row['final_constant_fraction_error']<=16*eps
        assert min(row['final_minimum_fraction'])>=0
        for stage in row['stages']:
            assert stage['exact_water_ledger']
            assert stage['minimum_carrier_kg']>0
            assert min(stage['minimum_fraction'][:2])>0
            assert min(stage['minimum_fraction'][2:])>=0
            active |= stage['limited_faces']>0
            if stage['constant_fraction_error'] is not None:
                assert stage['constant_fraction_error']<=16*eps
                assert stage['high_flux_constant_error_Pa_m2']<=32*eps*stage['high_flux_scale_Pa_m2']
    assert active  # One-sided condensate support exercises the production limiter.


def test_zero_condensate_one_sided_complete_step_derivative():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    result=fix.zero_derivative()
    assert result['gradient_finite']
    rows=result['one_sided']
    assert all(row['same_limiter_masks'] for row in rows)
    rounding=256*torch.finfo(torch.float64).eps*abs(result['objective'])/rows[-1]['h']
    truncation=2*abs(rows[-1]['fd']-rows[-2]['fd'])
    assert rows[-1]['absolute_error']<=rounding+truncation
