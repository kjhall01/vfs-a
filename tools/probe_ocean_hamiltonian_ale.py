"""Local Hamiltonian ALE measurements; report rejected cases explicitly."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
spec=importlib.util.spec_from_file_location('hamiltonian_ale_parent',Path(__file__).with_name('probe_ocean_enthalpy_ale.py'))
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
h=p.p.load('hamiltonian_ale_candidate','ocean_hamiltonian_ale.py')


def audits():
    rows=[]
    for kind in ['front','tilted','near_fresh','upper_salt']:
        for coast in [False,True]:
            for method in ['pcm','plm']:
                model,state,energy=p.p.p.q.p.case(kind,coast)
                row=dict(kind=kind,coast=coast,method=method)
                try:
                    target=state.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
                    end,r=h.HamiltonianRemapper(method=method).remap_with_report(model,state,target,energy)
                    scale=sum(v.abs() for v in h.column_terms(energy,state.physical_state(energy.reference))[0].values()).clamp_min(1)
                    row.update(status='completed',energy_J_m2={k:float(v/model.mesh.cell_area.sum()) for k,v in r.energy_change.items()},
                        provisional_energy_J_m2={k:float(v/model.mesh.cell_area.sum()) for k,v in r.provisional_energy_change.items()},
                        max_column_relative_energy_residual=float((r.column_energy_residual.abs()/scale).max()),
                        temperature_increment_range=[float(r.temperature_increment.min()),float(r.temperature_increment.max())],
                        min_column_entropy_change_J_K=float(r.column_entropy_change.min()),entropy_change_J_K=float(r.entropy_change))
                except (ValueError,FloatingPointError) as error:row.update(status='rejected',reason=str(error))
                rows.append(row)
    return rows


def wave(n,periods):
    metrics=dict(max_column_relative_energy_residual=0.,max_abs_temperature_increment_C=0.,sum_column_residual_J=0.)
    original=p.e.EnthalpyRemapper
    class ObservedRemapper(h.HamiltonianRemapper):
        def remap_with_report(self,model,state,target,energy):
            end,r=super().remap_with_report(model,state,target,energy)
            scale=sum(v.abs() for v in h.column_terms(energy,state.physical_state(energy.reference))[0].values()).clamp_min(1)
            metrics['max_column_relative_energy_residual']=max(metrics['max_column_relative_energy_residual'],float((r.column_energy_residual.abs()/scale).max()))
            metrics['max_abs_temperature_increment_C']=max(metrics['max_abs_temperature_increment_C'],float(r.temperature_increment.abs().max()))
            metrics['sum_column_residual_J']+=float(r.column_energy_residual.sum())
            return end,r
    try:
        p.e.EnthalpyRemapper=ObservedRemapper
        result=p.wave(n,'enthalpy',periods);result['scheme']='hamiltonian';result.update(metrics);return result
    finally:p.e.EnthalpyRemapper=original


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layers',type=int,default=8);parser.add_argument('--periods',type=float,default=.05);parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(status='running',device='cpu',dtype='float64',torch=str(torch.__version__),source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,h.__file__,p.__file__,p.e.__file__]})
    try:
        with torch.no_grad():report['audits' if args.audit else 'wave']=audits() if args.audit else wave(args.layers,args.periods)
        report['status']='completed'
    except Exception as error:report.update(status='rejected',error=str(error));raise
    finally:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
