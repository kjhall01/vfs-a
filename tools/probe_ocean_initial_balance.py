"""Fine-mesh initial equilibrium measurements for frozen ocean candidate #36.

No trajectory or empirical accuracy gate. Compare independent continuum jet
terms and separately evaluate the full entropy candidate at the same state.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

spec=importlib.util.spec_from_file_location('initial_balance_qualification',Path(__file__).with_name('qualify_ocean_entropy.py'))
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)


def measure(cells, lloyd_iters):
    started=time.monotonic()
    mesh=q.VoronoiMesh(n_cells=cells,lloyd_iters=lloyd_iters,verbose=False)
    mesh_seconds=time.monotonic()-started
    w=q.p.load('fine_balance_williamson','verify_ocean_williamson.py')
    model=q.Ocean(mesh,q.OceanColumn(1,total_depth=w.H0,gravity=w.GRAVITY),dynamics=q.JMD95Dynamics(gravity=w.GRAVITY,omega=w.OMEGA))
    physical=w.initial_state(model)
    independent,arrays=w.initial_balance(w.make_model(mesh),physical)
    ref=q.p.previous.reference();energy=q.c.EntropyEnergy(model.dynamics.energy,ref)
    state=q.c.EntropyState.from_temperature(physical,ref)
    rhs,work=q.c.evaluate(model,state,1.,energy)
    edge=(mesh.dc_edge*mesh.dv_edge)[:,None];area=mesh.cell_area[:,None]
    rate,scale=q.c.contraction(energy.gradients(state),rhs)
    result=dict(cells=cells,edges=mesh.nEdges,lloyd_iterations_requested=lloyd_iters,
        mesh_build_seconds=mesh_seconds,mean_cell_length_km=float(mesh.cell_area.mean().sqrt()/1000),
        minimum_dc_km=float(mesh.dc_edge.min()/1000),maximum_dc_km=float(mesh.dc_edge.max()/1000),
        cell_area_cv=float(mesh.cell_area.std()/mesh.cell_area.mean()),
        lloyd_diagnostics=mesh.lloyd_diagnostics,constant_density_operator_diagnosis=independent,
        candidate_momentum_rms_m_s2=q.rms(rhs.u_e,edge),candidate_h_rate_rms_m_s=q.rms(rhs.h,area),
        candidate_minus_constant_density_momentum_rms_m_s2=q.rms(rhs.u_e-torch.from_numpy(arrays['rhs_u']),edge),
        candidate_minus_constant_density_h_rate_rms_m_s=q.rms(rhs.h-torch.from_numpy(arrays['rhs_h']),area),
        candidate_normalized_energy_work=float(rate/scale),rest={})
    for profile in ['flat','topography']:
        bottom=3000+200*mesh.cell_xyz[:,:1] if profile=='topography' else torch.full((cells,1),3000.)
        h=torch.cat([torch.full_like(bottom,500.),torch.full_like(bottom,1000.),bottom-1500.],-1)
        model=q.Ocean(mesh,q.OceanColumn(3,total_depth=bottom,rest_thickness=torch.tensor([500.,1000.,1500.])),dynamics=q.JMD95Dynamics())
        physical=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=torch.tensor([17.,10.,4.]).expand_as(h),salinity=torch.tensor([34.,35.,35.5]).expand_as(h))
        state=q.c.EntropyState.from_temperature(physical,ref);energy=q.c.EntropyEnergy(model.dynamics.energy,ref)
        rhs,work=q.c.evaluate(model,state,1.,energy)
        result['rest'][profile]=dict(momentum_rms_m_s2=q.rms(rhs.u_e,edge),momentum_max_m_s2=float(rhs.u_e.abs().max()),h_rate_rms_m_s=q.rms(rhs.h,area))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,required=True)
    parser.add_argument('--lloyd-iters',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(candidate_commit='705720c79e38c2f07de39d45a33c9ede50707b87',scope='initial tendencies only; no integration',status='running',
        dtype='float64',device='cpu',torch=torch.__version__,evaluation_dt_s=1.,
        source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,q.__file__,q.c.__file__,Path(__file__).with_name('verify_ocean_williamson.py')]})
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        with torch.no_grad(): report.update(measure(args.cells,args.lloyd_iters))
        report['status']='completed'
    except Exception as error:
        report.update(status='rejected',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-start
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
