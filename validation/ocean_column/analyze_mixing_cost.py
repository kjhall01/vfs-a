"""Compare the short mixing-cost experiments, retaining numerical-reference limits."""
import argparse
import json
from pathlib import Path
from analyze_sustained import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records={}
    names=['baseline','optimized']+[f'default-{dt}-0.95' for dt in (75,37.5,18.75)]
    names += [f'predictor-{dt}-{margin}' for margin in (.5,.95) for dt in (150,75,37.5,18.75)]
    names += ['predictor-150-0.8']+[f'{s}-{m}' for s in ('cooling','rain_heating') for m in ('default','predictor')]
    for name in names:
        r=json.loads((a.input/(name+'.json')).read_text())
        if r['status']!='completed' or r['completed_hours']!=2:raise ValueError(f'incomplete: {name}')
        records[name]=r
    pairs=[]
    def add(kind,left,right):pairs.append(dict(kind=kind,left=left,right=right,metrics=compare(records[left],records[right])))
    add('exact_cache_comparison','optimized','baseline')
    for label in ('default','predictor_half','predictor_95'):
        get=lambda dt: ('optimized' if dt==150 else f'default-{dt}-0.95') if label=='default' else f'predictor-{dt}-{.5 if label=="predictor_half" else .95}'
        for coarse,fine in ((150,75),(75,37.5),(37.5,18.75)):
            add('successive_dt_'+label,get(coarse),get(fine))
        for dt in (150,75,37.5):add('own_fine_reference_'+label,get(dt),get(18.75))
    for dt in (150,75,37.5,18.75):
        default='optimized' if dt==150 else f'default-{dt}-0.95'
        for margin in (.5,.95):add('same_dt_method_difference',f'predictor-{dt}-{margin}',default)
    add('margin_difference','predictor-150-0.8','predictor-150-0.5')
    for s in ('cooling','rain_heating'):add('forcing_smoke',s+'-predictor',s+'-default')
    out=dict(completed_cases=len(records),hours_per_case=2,
        max_abs_energy_residual_J_kg=max(abs(r['energy_error_J_kg']) for r in records.values()),
        max_abs_salt_relative_change=max(abs(r['salt_relative_change']) for r in records.values()),
        max_abs_passive_relative_change=max(abs(r['passive_relative_change']) for r in records.values()),
        exact_cached_profiles=records['baseline']['profiles']==records['optimized']['profiles'],
        exact_cached_limiter=records['baseline']['limiter']==records['optimized']['limiter'],
        reference='18.75 s same-method numerical solution; not analytic truth or a demonstrated converged solution',
        timings_note='Concurrent single-thread CPU campaign runs; use interleaved benchmark.json for cost comparisons',
        pairs=pairs)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='pairs'},indent=2))

if __name__=='__main__':main()
