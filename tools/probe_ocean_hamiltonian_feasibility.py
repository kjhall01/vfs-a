"""Weak stable/unstable stratification feasibility cases; output path argument."""
import importlib.util,sys,torch,json
from pathlib import Path
root=Path(__file__).parent
s=importlib.util.spec_from_file_location('testp',root/'probe_ocean_hamiltonian_ale.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
rows=[]
with torch.no_grad():
 for sign in [1.,-1.]:
  for amplitude in [1e-6,1e-4,.01,1.]:
   m,x,e=p.p.p.p.q.p.case('front');physical=x.physical_state(e.reference)
   physical=physical.with_tensors(torch.zeros_like(x.u_e),x.h,x.h*(10+sign*amplitude*torch.tensor([1.,0.,-1.])),x.h*35.,x.tracer_h)
   x=p.p.p.p.front.EntropyState.from_temperature(physical,e.reference)
   row=dict(sign=sign,amplitude_C=amplitude)
   try:
    y,r=p.h.HamiltonianRemapper(method='pcm').remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
    row.update(status='completed',min_entropy_change=float(r.column_entropy_change.min()))
   except Exception as exc:row.update(status='rejected',reason=str(exc))
   rows.append(row)
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
