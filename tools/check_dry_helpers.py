"""Additional same-environment regression fixture for dry helper extraction.

Capture with the pass-2 installed package before testing pass 3. Covers
the retained conservative diagonal model, including public diagnostics.
The unchanged fixture also archives 100 tensors from the removed canonical core.
"""
import argparse
import json
from pathlib import Path
import torch
from vfs import DryHydrostaticConservativeDynamics
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.core.state import State
from vfs.core.integrators import ssprk3


def clean(value):
    if hasattr(value,'location'):return value.values.detach().clone()
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:clean(x) for k,x in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    if isinstance(value,State):return {k:clean(getattr(value,k)) for k in ('u_e','ps','mtheta_i','tracer_m')}
    if value is None or isinstance(value,(float,int,str,bool)):return value
    raise TypeError(type(value))


def capture():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=0,verbose=False)
    A,B=make_hybrid_coefficients(4,p_top=5000.,sigma_power=1.2)
    col=CharneyPhillipsColumn(4,A,B,phi_s=200*mesh.cell_xyz[:,0],phi_full_mode='direct_u_gl2_adjoint')
    gen=torch.Generator().manual_seed(35)
    rand=lambda *shape:torch.randn(*shape,generator=gen)
    s=State.from_theta(8*rand(mesh.nEdges,4),300+3*rand(32,5),1e5+500*rand(32,1),mesh=mesh,column=col)
    result={}
    with torch.no_grad():
        for name,d in [('diagonal',DryHydrostaticConservativeDynamics())]:
            ctx=d._transport_primitives(s)
            outputs={'primitives':ctx,'rhs':d.tendency(s),'step':ssprk3(d,s,1.)}
            for method in ['tendency_terms','energy_terms','pressure_fields','thermo_fields','wind_fields',
                           'force_terms','force_balance_terms','rotation_work_terms','cfl_terms',
                           'divergence','vorticity','omega','scalar_energy_gradients',
                           'scalar_pressure_energy_gradient_terms','total_mass','total_energy']:
                outputs[method]=getattr(d,method)(s, dt=1.) if method=='cfl_terms' else getattr(d,method)(s)
            result[name]=clean(outputs)
    return result


def compare(a,b,path='dry'):
    if isinstance(b,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True,msg=lambda s:f'{path}: {s}')
        return 1
    if isinstance(b,dict):
        assert a.keys()==b.keys(),path
        return sum(compare(a[k],b[k],path+'/'+k) for k in b)
    if isinstance(b,list):
        assert len(a)==len(b)
        return sum(compare(x,y,path) for x,y in zip(a,b))
    assert a==b,(path,a,b)
    return 0


def check(actual, expected, mode="exact"):
    """Select a named policy without altering the historical exact comparator."""
    import importlib.util
    path = Path(__file__).with_name("fixture_regression.py")
    spec = importlib.util.spec_from_file_location("fixture_regression", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.compare_selected(actual, expected, "dry_helpers_pass2.pt", mode, compare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--mode', choices=('exact', 'portable'), default='exact')
    args = parser.parse_args()
    if args.capture and args.mode != 'exact':
        parser.error('capture cannot select portable comparison')
    fixture = Path(__file__).resolve().parents[1] / 'tests/fixtures/dry_helpers_pass2.pt'
    actual = capture()
    if args.capture:
        torch.save(actual, fixture)
        result = {'mode': 'capture_pass2'}
    else:
        result = {'mode': args.mode + '_comparison', 'status': 'pass'}
        try:
            result.update(check(actual, torch.load(fixture, weights_only=True), args.mode))
        except AssertionError as exc:
            result.update(status='fail', error=str(exc))
    import vfs
    result.update(source=vfs.__file__, torch=str(torch.__version__))
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    if result.get('status') == 'fail':
        raise SystemExit(1)


if __name__=='__main__':main()
