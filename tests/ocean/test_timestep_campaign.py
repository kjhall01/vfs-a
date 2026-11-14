"""Timestep qualification must not confuse completion with accuracy."""
import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[2]/'validation/timestep/run.py'
spec=importlib.util.spec_from_file_location('timestep_campaign',path)
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def test_subcycles_bound_physical_mixing_interval():
    assert campaign.substeps(240,60)==4
    assert campaign.substeps(601,60)==11


def test_completion_alone_never_qualifies():
    row=dict(status='completed',max_energy_error_J_kg=0.,max_mass_error_relative=0.,
             max_salt_error_relative=0.,max_tracer_error_relative=0.)
    limits=dict(u=.001,energy=.001,inventory=1e-10)
    assert not campaign.qualifies(row,None,limits)
    assert campaign.qualifies(row,dict(u=.0005),limits)
    assert not campaign.qualifies(row,dict(u=.002),limits)
    row['max_salt_error_relative']=1e-8
    assert not campaign.qualifies(row,dict(u=0.),limits)


def test_endpoint_error_uses_mass_weights():
    a={k:torch.tensor([[0.,2.]]) for k in ('u','CT','SA','h','k')}
    b={k:torch.zeros(1,2) for k in a}
    b.update(weight=torch.tensor([[3.,1.]]),edge_weight=torch.tensor([[1.,3.]]))
    errors=campaign.compare(a,b)
    assert errors['CT']==pytest.approx(1.)
    assert errors['u']==pytest.approx(3**.5)


def test_atomic_report_preserves_previous_after_nan(tmp_path):
    path=tmp_path/'result.json'
    campaign.write(path,dict(status='running'))
    old=path.read_text()
    with pytest.raises(ValueError):campaign.write(path,dict(error=float('nan')))
    assert path.read_text()==old
