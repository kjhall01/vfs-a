"""Summarize EOS controls without treating failed prefixes as completed runs."""
import argparse,json
from pathlib import Path
from analyze_sustained import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--exact-input',type=Path);a=p.parse_args()
    data={}
    for path in a.input.glob('*.json'):
        r=json.loads(path.read_text())
        if 'eos' in r:data[path.stem]=r
    if len(data)!=15:raise ValueError(f'expected 15 cases, found {len(data)}')
    rows=[]
    for name,r in sorted(data.items()):
        row={k:r[k] for k in ('eos','solver','layers','dt_s','status','completed_hours','wall_s','failure','energy_error_J_kg','salt_relative_change','passive_relative_change')}
        row['file']=name+'.json'
        row['mean_iterations']=r['limiter'].get('coupled_iterations',0)/r['accepted_steps'] if r['accepted_steps'] else None
        rows.append(row)
    comparisons=[]
    for eos in ('jmd95','linear-gibbs','linear-cp'):
        for left,right,kind in [(f'{eos}-accelerated-16-300',f'{eos}-accelerated-16-75','timestep'),(f'{eos}-accelerated-16-300',f'{eos}-cautious-16-300','nonlinear_solver')]:
            x,y=data[left],data[right]
            if x['status']==y['status']=='completed' and x['completed_hours']==y['completed_hours']:
                comparisons.append(dict(kind=kind,left=left,right=right,metrics=compare(x,y)))
    result=dict(cases=rows,comparisons=comparisons,reference_note='same-EOS numerical comparisons only, no analytic trajectory or proof of convergence',
        max_abs_energy_residual_J_kg=max(abs(r['energy_error_J_kg']) for r in data.values()))
    if a.exact_input:
        exact={p.stem:json.loads(p.read_text()) for p in a.exact_input.glob('linear-cp-*.json')}
        result['exact_cases']=[{k:r.get(k) for k in ('layers','dt_s','status','completed_hours','wall_s','failure','energy_error_J_kg','limiter')} for _,r in sorted(exact.items())]
        result['exact_comparisons']=[]
        ref=exact['linear-cp-accelerated-16-18.75']
        for name,r in exact.items():
            if r['layers']==16 and r['status']==ref['status']=='completed' and r['completed_hours']==ref['completed_hours']:
                result['exact_comparisons'].append(dict(case=name,metrics=compare(r,ref)))
        result['same_eos_solver_comparison']=compare(exact['linear-cp-accelerated-16-300'],data['linear-cp-cautious-16-300'])
        result['long_attempt']=json.loads((a.exact_input/'48h.json').read_text())
        result['long_attempt'].pop('profiles')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
