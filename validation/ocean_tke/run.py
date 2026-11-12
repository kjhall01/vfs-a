"""Offline qualification of optional TKE; writes compact scalar evidence only."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
import vfs
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean, TKEOcean, TKEClosure, AdaptiveThermalRemapper
from vfs.ocean.stratified.tke import diffuse


def run(layers, *, steps=20, dt=10.):
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    ocean=StratifiedOcean(mesh,OceanColumn(layers,total_depth=100.),remapper=AdaptiveThermalRemapper())
    model=TKEOcean(ocean,closure=TKEClosure(mixing_length=2.))
    h=torch.full((mesh.nCells,layers),100./layers,dtype=torch.float64)
    temperature=torch.linspace(15.,12.,layers,dtype=h.dtype).expand_as(h)
    salinity=torch.linspace(34.8,35.2,layers,dtype=h.dtype).expand_as(h)
    u=torch.linspace(.3,-.3,layers,dtype=h.dtype).expand(mesh.nEdges,layers)
    passive=torch.linspace(0.,1.,layers,dtype=h.dtype).expand_as(h)[...,None]
    s=ocean.state(u_e=u,h=h,temperature=temperature,salinity=salinity,tracer=passive)
    start=model.state(s,tke=torch.full_like(h,.01));state=start
    initial=model.budgets(start);mass=(ocean.energy.mass_area(s)*h).sum()
    logs=[];column_errors=[]
    with torch.no_grad():
        for _ in range(steps):
            state,r=model.mix_with_report(state,dt)
            column_errors.append(float((r.column_energy_residual.abs()/(ocean.energy.mass_area(s)*h).sum(-1)).max()))
            logs.append({k:float(getattr(r,k).sum()/mass) for k in r.__dict__})
    final=model.budgets(state)
    result={'layers':layers,'dt_s':dt,'steps':steps,'scope':'vertical mixing only; no dynamics',
        'energy_change_J_kg':float((final['total']-initial['total'])/mass),
        'kinetic_change_J_kg':float((final['kinetic']-initial['kinetic'])/mass),
        'tke_change_J_kg':float((final['tke']-initial['tke'])/mass),
        'min_specific_tke':float(state.specific_tke().min()),
        'salt_relative_change':float((final['salt']-initial['salt'])/initial['salt']),
        'passive_relative_change':float(((final['tracers']-initial['tracers'])/initial['tracers']).abs().max()),
        'entropy_change_J_kg_K':float((final['entropy']-initial['entropy'])/mass),
        'max_column_energy_residual_J_kg':max(column_errors),
        'integrated_transfers_J_kg':{k:sum(x[k] for x in logs) for k in logs[0]}}
    # One ordinary dynamics + ALE + TKE step; report its distinct ledgers.
    fractions=torch.arange(1,layers+1,dtype=h.dtype);fractions=fractions/fractions.sum()
    advanced,report=model.step_with_report(start,.1)
    try:
        mapped,remap=model.remap_with_report(advanced,advanced.ocean.h.sum(-1,keepdim=True)*fractions)
        result['combined_step']={'tke_transport_residual_J_kg':float(report.tke_transport_residual/mass),
            'remap_tke_change_J_kg':float(remap.tke_change/mass),
            'remap_total_energy_change_J_kg':float(remap.total_energy_change/mass),
            'min_remapped_tke':float(mapped.specific_tke().min())}
    except (ValueError,FloatingPointError) as e:
        result['combined_step']={'remap_rejected':str(e)}
    return result


def diffusion_convergence():
    mass=torch.tensor([[1.,2.]],dtype=torch.float64);g=mass.new_tensor([[.3]])
    initial=mass.new_tensor([[1.,0.]])
    # Independent continuous two-box diffusion solution at t=1.
    mean=1/3;difference=torch.exp(mass.new_tensor(-.45))
    exact=mass.new_tensor([[mean,mean]])+difference*mass.new_tensor([[2/3,-1/3]])
    rows=[]
    for count in [4,8,16,32,64]:
        x=initial
        for _ in range(count):x=diffuse(x,mass,g,1/count)
        rows.append({'steps':count,'max_error':float((x-exact).abs().max())})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    root=Path(vfs.__file__).parent
    paths=['ocean/stratified/tke.py','ocean/stratified/tke_model.py']
    report={'torch':torch.__version__,'dtype':'float64','device':'cpu',
        'installed_source_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        'driver_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'columns':[run(n) for n in (4,16,32)],'diffusion_convergence':diffusion_convergence()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
