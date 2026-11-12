"""Check the offline two-component column metric and independent transport reference."""
import importlib.util
from pathlib import Path
import sys
import torch
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,OceanBoundaryStress

path=Path(__file__).resolve().parents[2]/'validation/ocean_column/run.py'
spec=importlib.util.spec_from_file_location('column_campaign_test',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def test_column_native_metric_and_rotation():
    mesh=module.ColumnMetric();o=StratifiedOcean(mesh,OceanColumn(4,total_depth=100.))
    h=torch.full((1,4),25.,dtype=torch.float64)
    u=h.new_tensor([[.2,.1,-.3,.4],[.1,-.2,.2,.3]])
    s=o.state(u_e=u,h=h,temperature=h*0+12,salinity=h*0+35)
    m=TKEOcean(o);state=m.state(s,tke=h*0)
    expected=.5*1025*(h*u.square()).sum()
    torch.testing.assert_close(m.budgets(state)['kinetic'],expected,rtol=2e-15,atol=0)
    rotated=module.rotate(state,.7)
    torch.testing.assert_close(m.budgets(rotated)['kinetic'],expected,rtol=2e-15,atol=0)
    energy=o.energy.base.mechanical
    torch.testing.assert_close((h*energy.kinetic_thickness_gradient(o.check_state(s))).sum(),expected,rtol=2e-15,atol=0)

def test_column_wind_impulse_two_components():
    mesh=module.ColumnMetric();o=StratifiedOcean(mesh,OceanColumn(4,total_depth=100.))
    h=torch.full((1,4),25.,dtype=torch.float64)
    m=TKEOcean(o,boundary=OceanBoundaryStress())
    state=m.state(o.state(u_e=torch.zeros(2,4,dtype=h.dtype),h=h,temperature=h*0+12,salinity=h*0+35),tke=h*0)
    out,r=m.boundary_with_report(state,100.,surface_stress=h.new_tensor([.1,.2]))
    torch.testing.assert_close((h*out.ocean.u_e).sum(-1),100/1025*h.new_tensor([.1,.2]),rtol=2e-15,atol=0)
    assert abs(float(r.energy_residual))<1e-14

def test_short_control_transport_reference():
    r=module.run(4,30.,.05,'wind',mix_scalars=False)
    assert r['status']=='completed'
    actual=torch.tensor(r['transport']);reference=torch.tensor(r['analytic_wind_transport'])
    torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-10)
