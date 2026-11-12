"""Replay a pinned repository TKE source against the installed candidate wheel."""
import json,torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEClosure
import argparse, subprocess, sys, types
parser=argparse.ArgumentParser(description='Exact comparison with pinned pre-extraction TKE source')
parser.add_argument('--repository',required=True)
args=parser.parse_args()
parent='a2023f352b209e26454ed93894ef89aec3f3f726'
source=subprocess.check_output(['git','-C',args.repository,'show',parent+':vfs/ocean/stratified/tke.py'],text=True)
module=types.ModuleType('vfs.ocean.stratified.legacy_tke')
module.__package__='vfs.ocean.stratified'
sys.modules[module.__name__]=module
exec(compile(source,parent+':tke.py','exec'),module.__dict__)
mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
o=StratifiedOcean(mesh,OceanColumn(4,total_depth=100.))
h=torch.full((mesh.nCells,4),25.,dtype=torch.float64)
s=o.state(u_e=torch.linspace(.2,-.2,4,dtype=h.dtype).expand(mesh.nEdges,4),h=h,
    temperature=torch.linspace(15.,12.,4,dtype=h.dtype).expand_as(h),salinity=torch.full_like(h,35.))
count=0
for scalar in (0.,1e-4,1.,100.):
    for cls in (module.TKEClosure,TKEClosure):
        result=cls._heat(o,s,torch.full_like(h,scalar))
        if cls is module.TKEClosure:old=result
    assert torch.equal(old[0].entropy_h,result[0].entropy_h)
    assert torch.equal(old[1],result[1]);count+=2
old=module.TKEClosure(2.).apply(o,s,h*.01,10.)
new=TKEClosure(2.).apply(o,s,h*.01,10.)
for name in ('u_e','h','entropy_h','salt_h'):
    assert torch.equal(getattr(old[0],name),getattr(new[0],name));count+=1
assert torch.equal(old[1],new[1]);count+=1
for name in old[2].__dataclass_fields__:
    assert torch.equal(getattr(old[2],name),getattr(new[2],name));count+=1
print(json.dumps(dict(parent_commit='a2023f352b209e26454ed93894ef89aec3f3f726',exact_tensor_comparisons=count,status='pass'),indent=2))
