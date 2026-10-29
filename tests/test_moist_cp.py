import torch
from vfs.atmosphere.moist_cp import interface_pressure_mass, reconstruct_moist_cp
from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics


def setup():
    p=torch.tensor([[5000.,45000.,100000.],[5000.,47000.,102000.]],dtype=torch.float64)
    m=interface_pressure_mass(p.diff())
    s=torch.tensor([[500.,300.,100.],[510.,290.,110.]],dtype=torch.float64)
    q=torch.tensor([[[.01,.001,.0],[.015,.002,.001],[.02,.003,.0]]],dtype=torch.float64).expand(2,-1,-1)
    return p,m,s,q


def test_species_integral_and_entropy_integral():
    p,m,s,q=setup();r=reconstruct_moist_cp(p,m*s,m[...,None]*q)
    torch.testing.assert_close((r['dp'][...,None]*r['fractions'][...,1:].mean(-2)).sum(-2),(m[...,None]*q).sum(-2))
    torch.testing.assert_close((r['dp']*r['entropy'].mean(-1)).sum(-1),(m*s).sum(-1))
    assert (r['fractions']>=0).all()


def test_entropy_reference_covariance():
    p,m,s,q=setup();r=reconstruct_moist_cp(p,m*s,m[...,None]*q)
    offsets=p.new_tensor([20.,80.,-30.,10.])
    full=torch.cat([1-q.sum(-1,keepdim=True),q],-1)
    shifted=reconstruct_moist_cp(p,m*(s+(full*offsets).sum(-1)),m[...,None]*q)
    # Reconstructed entropy shifts by exactly the local species reference shift.
    torch.testing.assert_close(shifted['entropy']-r['entropy'],(r['fractions']*offsets).sum(-1))
    # Using that same shifted EOS gives identical physical temperature.
    t=MixtureThermodynamics().temperature_from_entropy(shifted['entropy']-(r['fractions']*offsets).sum(-1),r['pressure'],r['fractions'])
    torch.testing.assert_close(t,r['temperature'])


def test_isentropic_dry_temperature():
    p,m,s,q=setup();s=torch.ones_like(s)*300;q=torch.zeros_like(q)
    r=reconstruct_moist_cp(p,m*s,m[...,None]*q);t=MixtureThermodynamics()
    expected=t.T_ref*torch.exp(s[...,:-1,None]/t.cp_d)*(r['pressure']/t.p_d_ref)**(t.R_d/t.cp_d)
    torch.testing.assert_close(r['temperature'],expected)
