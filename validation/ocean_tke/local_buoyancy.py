"""Locality and analytic pair-diffusion qualification; no external data."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import vfs
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean, TKEClosure


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    ocean=StratifiedOcean(mesh,OceanColumn(4,total_depth=100.))
    closure=TKEClosure(2.,c_epsilon=0.)
    h=torch.tensor([10.,30.,25.,35.],dtype=torch.float64).expand(mesh.nCells,4)
    state=ocean.state(u_e=torch.zeros(mesh.nEdges,4,dtype=h.dtype),h=h,
        temperature=h.new_tensor([15.,14.,10.,8.]).expand_as(h),
        salinity=h.new_tensor([34.9,35.,35.1,35.2]).expand_as(h))
    mass=ocean.energy.mass_area(state)*h
    reservoir=mass*.01
    g=torch.zeros(mesh.nCells,3,dtype=h.dtype);g[:,0]=.001
    end,r,w,lw,flux,residual=closure._mix_adjacent(ocean,state,reservoir,g,10.)
    physical=state.physical_state(ocean.reference)
    before_e=ocean.reference.evaluate(physical.salinity(),physical.temperature())['h0']
    new=end.physical_state(ocean.reference)
    after_e=ocean.reference.evaluate(new.salinity(),new.temperature())['h0']
    result={'dtype':'float64','device':'cpu','torch':torch.__version__,
        'max_layer_budget_residual_J_kg':float((residual.abs()/mass).max()),
        'interface_work_J_kg_column':float(w.sum()/mass.sum()),
        'upper_fraction':float((lw[:,0]/w[:,0]).mean()),
        'lower_fraction':float((lw[:,1]/w[:,0]).mean()),
        'remote_tke_unchanged':bool(torch.equal(r[:,2:],reservoir[:,2:])),
        'remote_entropy_unchanged':bool(torch.equal(end.entropy_h[:,2:],state.entropy_h[:,2:])),
        'potential_enthalpy_inventory_error_J_kg':float(((h*(after_e-before_e)).sum(-1)/h.sum(-1)).abs().max()),
        'salt_inventory_error':float((end.salt_h.sum(-1)-state.salt_h.sum(-1)).abs().max())}
    try:
        closure._mix_adjacent(ocean,state,reservoir*h.new_tensor([0.,0.,1e6,1e6]),g,10.)
        result['remote_funding_rejected']=False
    except ValueError as error:
        result['remote_funding_rejected']='adjacent layer' in str(error)
    # Independent exact solution of continuous two-box diffusion, same frozen G.
    duration=100.
    rate=g[:,0]*(1/h[:,0]+1/h[:,1])
    decay=torch.exp(-rate*duration)
    e0=before_e
    mean=(h[:,0]*e0[:,0]+h[:,1]*e0[:,1])/(h[:,0]+h[:,1])
    exact0=mean+(e0[:,0]-mean)*decay
    exact1=mean+(e0[:,1]-mean)*decay
    rows=[]
    for steps in [1,2,4,8,16]:
        current=state;budget=reservoir
        for _ in range(steps):
            current,budget,*_=closure._mix_adjacent(ocean,current,budget,g,duration/steps)
        p=current.physical_state(ocean.reference)
        e=ocean.reference.evaluate(p.salinity(),p.temperature())['h0']
        error=torch.maximum((e[:,0]-exact0).abs(),(e[:,1]-exact1).abs()).max()
        rows.append({'steps':steps,'max_enthalpy_error_J_kg':float(error)})
    result['pair_refinement']=rows
    result['source_hash']=hashlib.sha256((Path(vfs.__file__).parent/'ocean/stratified/tke.py').read_bytes()).hexdigest()
    result['driver_hash']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
