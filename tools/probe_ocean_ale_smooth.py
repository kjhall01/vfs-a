"""Single-remap smooth stratification resolution sweep; output path argument."""
import importlib.util,sys,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).parent
s=importlib.util.spec_from_file_location('probe',root/'probe_ocean_entropy_ale.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
rows=[]
with torch.no_grad():
 for n in (8,16,32,64,128):
  mesh=p.v.VoronoiMesh(n_cells=32,lloyd_iters=2,verbose=False)
  column=p.v.OceanColumn(n,total_depth=3000.)
  sigma=torch.linspace(0,1,n+1);depth=3000*(sigma[None,:]+.08*mesh.cell_xyz[:,:1]*torch.sin(torch.pi*sigma)[None,:]);h=depth.diff(dim=-1);center=.5*(depth[:,1:]+depth[:,:-1])
  model=p.v.Ocean(mesh,column,dynamics=p.v.JMD95Dynamics(omega=0.))
  physical=model.state(u_e=torch.zeros(mesh.nEdges,n),h=h,temperature=20-16*center/3000,salinity=34.5+center/3000)
  ref=p.p.q.p.previous.reference();state=p.p.front.EntropyState.from_temperature(physical,ref);energy=p.p.front.EntropyEnergy(model.dynamics.energy,ref)
  end,r=p.a.EntropyRemapper().remap_with_report(model,state,p.v.ReferenceGrid()(state),energy)
  targetcenter=end.h.cumsum(-1)-.5*end.h
  err=end.physical_state(ref).temperature()-(20-16*targetcenter/3000)
  rows.append(dict(layers=n,temperature_max_error_C=float(err.abs().max()),temperature_rms_error_C=float((end.h*err.square()).sum().div(end.h.sum()).sqrt()),energy_change_J={k:float(v) for k,v in r.energy_change.items()},energy_change_J_m2={k:float(v/mesh.cell_area.sum()) for k,v in r.energy_change.items()},actual_changed_columns=r.changed_columns))
Path(sys.argv[1]).write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
