"""Retained Held–Suarez source equations; no legacy model/framework required."""
import pytest
import torch
from dataclasses import replace
from vfs import State, DryAtmosphere
from vfs.atmosphere.held_suarez import HeldSuarezForcing, HeldSuarezConstants
from vfs.core.state import interface_mass
from vfs.core.integrators import ssprk3
from test_conservative_imex import setup


@pytest.mark.parametrize('overrides', [False, True])
@pytest.mark.parametrize('levels', [1, 3, 12])
def test_source_equations_and_input_preservation(levels, overrides):
    _, state, _ = setup(levels)
    saved = [x.clone() for x in (state.u_e, state.mtheta_i, state.ps)]
    c = HeldSuarezConstants()
    f = HeldSuarezForcing(Teq=torch.tensor(270.), kT=torch.tensor(2e-6), kv=torch.full((state.mesh.nCells, levels), 3e-6)) if overrides else HeldSuarezForcing()
    pi = state.interface_pressure()
    lat = state.mesh.cell_xyz[:, 2:3].clamp(-1., 1.).asin()
    sigma = (pi / c.p0).clamp_min(1e-12)
    teq = torch.maximum((c.T0-c.delta_T_y*lat.sin().square()-c.delta_theta_z*sigma.log()*lat.cos().square()) * sigma.pow(state.column.kappa), torch.tensor(c.T_min))
    sigma_i = (pi / state.ps.clamp_min(1.)).clamp_min(1e-12)
    kt = c.k_a + (c.k_s-c.k_a)*((sigma_i-c.sigma_b)/(1-c.sigma_b)).clamp(0.,1.)*lat.cos().pow(4)
    sf = (state.full_pressure()/state.ps.clamp_min(1.)).clamp_min(1e-12)
    kv = c.k_f*((sf-c.sigma_b)/(1-c.sigma_b)).clamp(0.,1.)
    if overrides:
        teq, kt, kv = f.Teq, f.kT, f.kv
    expected_heat = interface_mass(state.layer_dp())*kt*(teq/(state.exner_interface()/state.column.c_p).clamp_min(1e-12)-state.theta())
    expected_drag = -state.cell_to_edge(kv)*state.u_e
    actual = f(state)
    torch.testing.assert_close(actual.mtheta_i, expected_heat, rtol=0, atol=0)
    torch.testing.assert_close(actual.u_e, expected_drag, rtol=0, atol=0)
    assert torch.count_nonzero(actual.ps) == 0 and actual.tracer_m is None
    for x, before in zip((state.u_e, state.mtheta_i, state.ps), saved):
        torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.mesh is state.mesh and actual.column is state.column


def test_budgets_stage_composition_and_gradients():
    d, s, _ = setup()
    s = State(*(x.clone().requires_grad_() for x in (s.u_e, s.mtheta_i, s.ps)), mesh=s.mesh, column=s.column)
    model = DryAtmosphere(s.mesh, s.column, dynamics=d)
    f = HeldSuarezForcing()
    terms = f.tendency_terms(s)
    summed = terms['thermal_relaxation'] + terms['rayleigh_drag']
    for name in ('u_e','mtheta_i','ps'):
        torch.testing.assert_close(getattr(summed,name),getattr(f(s),name),rtol=0,atol=0)
    assert d.boundary_energy_rate_closed(s,terms['rayleigh_drag']) < 0
    for term in terms.values():
        assert torch.count_nonzero(term.ps) == 0
    power = sum(d.boundary_energy_rate_closed(s,t) for t in terms.values())
    torch.testing.assert_close(power,d.boundary_energy_rate_closed(s,f(s)),rtol=2e-15,atol=0)
    gradients = torch.autograd.grad(power,(s.u_e,s.mtheta_i,s.ps))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients)
    stages=[]
    def rhs(stage):
        stages.append(stage)
        return model.tendency(stage)+f(stage)
    out = ssprk3(rhs,s,1.)
    assert len(stages)==3 and stages[1] is not s and stages[2] is not s
    expected = ssprk3(lambda stage:d.tendency(stage)+f(stage),s,1.)
    for name in ('u_e','mtheta_i','ps'):
        torch.testing.assert_close(getattr(out,name),getattr(expected,name),rtol=0,atol=0)


def test_disabled_sources_and_prescribed_buffer_gradients():
    _,s,_=setup()
    c=replace(HeldSuarezConstants(),k_a=0.,k_s=0.,k_f=0.)
    rhs=HeldSuarezForcing(c)(s)
    assert not rhs.u_e.any() and not rhs.mtheta_i.any()
    teq=torch.tensor(270.,requires_grad=True)
    f=HeldSuarezForcing(Teq=teq)
    assert f.Teq is teq and 'Teq' in f.state_dict()
    gradient,=torch.autograd.grad(f(s).mtheta_i.sum(),(teq,))
    assert gradient>0
