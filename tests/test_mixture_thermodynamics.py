"""Thermodynamic identities independent of any proposed moist discretization."""
import pytest
import torch
from vfs.atmosphere.mixture_thermodynamics import (
    MixtureThermodynamics, water_mixing_ratios_to_mass_fractions,
    mass_fractions_to_water_mixing_ratios,
)

@pytest.mark.parametrize('water', [[0.,0.,0.],[.02,0.,0.],[.01,.002,.001]])
def test_first_law_and_inversions(water):
    m=MixtureThermodynamics()
    q=water_mixing_ratios_to_mass_fractions(torch.tensor([water],dtype=torch.float64))
    T=torch.tensor([280.],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([80000.],dtype=torch.float64,requires_grad=True)
    m.validate(T,p,q)
    e=m.internal_energy(T,q); s=m.entropy(T,p,q); alpha=m.specific_volume(T,p,q)
    torch.testing.assert_close(m.temperature_from_internal_energy(e,q),T)
    torch.testing.assert_close(m.temperature_from_entropy(s,p,q),T)
    torch.testing.assert_close(m.enthalpy(T,q),e+p*alpha)
    dsT,dsp=torch.autograd.grad(s.sum(),(T,p),retain_graph=True)
    daT,dap=torch.autograd.grad(alpha.sum(),(T,p),retain_graph=True)
    deT,=torch.autograd.grad(e.sum(),(T,),retain_graph=True)
    # de = T ds - p d(alpha) at fixed composition, tested in T and p directions.
    torch.testing.assert_close(deT,T*dsT-p*daT)
    torch.testing.assert_close(T*dsp-p*dap,torch.zeros_like(T),atol=1e-12,rtol=0)
    torch.testing.assert_close(mass_fractions_to_water_mixing_ratios(q),torch.tensor([water],dtype=torch.float64))


def test_reference_latent_heat_and_condensate_loading():
    m=MixtureThermodynamics();T=torch.tensor([250.,300.],dtype=torch.float64)
    qv=torch.tensor([[0.,1.,0.,0.]],dtype=torch.float64).expand(2,-1)
    ql=torch.tensor([[0.,0.,1.,0.]],dtype=torch.float64).expand(2,-1)
    torch.testing.assert_close(m.enthalpy(T,qv)-m.enthalpy(T,ql),m.latent_heat_vaporization(T))
    dry=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
    cloudy=torch.tensor([[.98,0.,.02,0.]],dtype=torch.float64)
    assert m.gas_constant(cloudy)<m.gas_constant(dry)
    assert m.heat_capacity_p(cloudy)>m.heat_capacity_p(dry)


def test_dry_limit_and_invalid_state():
    m=MixtureThermodynamics();q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
    T=torch.tensor([290.],dtype=torch.float64);p=torch.tensor([1e5],dtype=torch.float64)
    torch.testing.assert_close(m.internal_energy(T,q),(m.cp_d-m.R_d)*T-m.cp_d*m.T_ref)
    torch.testing.assert_close(m.entropy(T,p,q),m.cp_d*torch.log(T/m.T_ref))
    with pytest.raises(ValueError,match='sum to one'):
        m.validate(T,p,q*.9)
    with pytest.raises(ValueError,match='positive'):
        m.validate(-T,p,q)
