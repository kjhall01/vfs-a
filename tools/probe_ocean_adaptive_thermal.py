"""Run inherited Hamiltonian audits/waves with adaptive layerwise correction."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
s=importlib.util.spec_from_file_location('adaptive_parent',Path(__file__).with_name('probe_ocean_hamiltonian_ale.py'))
p=importlib.util.module_from_spec(s);sys.modules[s.name]=p;s.loader.exec_module(p)
a=p.p.p.load('adaptive_thermal_candidate','ocean_adaptive_thermal_ale.py')

def feasibility():
    rows=[]
    for sign in [1.,-1.]:
        for amplitude in [1e-6,1e-4,.01,1.]:
            m,x,e=p.p.p.p.q.p.case('front');physical=x.physical_state(e.reference)
            physical=physical.with_tensors(torch.zeros_like(x.u_e),x.h,x.h*(10+sign*amplitude*torch.tensor([1.,0.,-1.])),x.h*35.,x.tracer_h)
            x=p.p.p.p.front.EntropyState.from_temperature(physical,e.reference)
            row=dict(sign=sign,amplitude_C=amplitude)
            try:
                y,r=a.AdaptiveThermalRemapper(method='pcm').remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
                scale=sum(v.abs() for v in p.h.column_terms(e,x.physical_state(e.reference))[0].values()).clamp_min(1)
                row.update(status='completed',max_relative_energy_residual=float((r.column_energy_residual.abs()/scale).max()),
                    min_entropy_change=float(r.column_entropy_change.min()),min_relative_entropy_change=float((r.column_entropy_change/(e.mass_area(x)*x.entropy_h.abs()).sum(-1).clamp_min(1)).min()),max_abs_temperature_increment=float(r.temperature_increment.abs().max()),
                    entropy_active_columns=int(r.entropy_constraint_columns.sum()),iterations=r.iterations)
            except (ValueError,FloatingPointError) as exc:row.update(status='rejected',reason=str(exc))
            rows.append(row)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layers',type=int,default=8);parser.add_argument('--periods',type=float,default=.05);parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(status='running',device='cpu',dtype='float64',source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,a.__file__,p.__file__]})
    original=p.h.HamiltonianRemapper;p.h.HamiltonianRemapper=a.AdaptiveThermalRemapper
    try:
        with torch.no_grad():
            if args.audit:report.update(audits=p.audits(),feasibility=feasibility())
            else:report['wave']=p.wave(args.layers,args.periods);report['wave']['scheme']='adaptive-thermal'
        report['status']='completed'
    except Exception as exc:report.update(status='rejected',reason=str(exc));raise
    finally:
        p.h.HamiltonianRemapper=original
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
