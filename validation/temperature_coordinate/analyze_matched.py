"""Compact comparison metrics from the retained matched-EOS campaign JSON."""
import argparse,json
from pathlib import Path


def profile_mld(profile,depth=100.,threshold=.03):
    """First threshold crossing of mean anomaly profile, linearly interpolated.

    Not a potential-density MLD. Return depth if there is no crossing.
    """
    dz=depth/len(profile);initial=profile[0]
    for k in range(1,len(profile)):
        lower=profile[k]-initial;upper=profile[k-1]-initial
        if lower>threshold:
            return (k-.5)*dz+dz*(threshold-upper)/(lower-upper)
    return depth


def summarize(data):
    result=dict(kind=data['config']['kind'],cells=data['config']['cells'],layers=data['config']['layers'],initial_anomaly_mismatch_kg_m3=data['initial_anomaly_mismatch_kg_m3'])
    cases=data['cases']
    if set(cases)!= {'jmd95','linear'} or any(v['status']!='completed' for v in cases.values()):
        result['status']='incomplete';return result
    result.update(status='completed',final_difference=data['final_difference'])
    result['energy_errors_J_kg']={k:v['energy_minus_inputs_J_kg'] for k,v in cases.items()}
    if result['kind']=='wave':
        result['layers']=2
        result['frequency_over_jmd95_continuum']={k:v['frequency_over_reference'] for k,v in cases.items()}
        result['linear_vs_jmd95_frequency_percent']=100*(cases['linear']['frequency_over_reference']/cases['jmd95']['frequency_over_reference']-1)
    if result['kind']=='mixing':
        result['interpolated_mean_profile_mld_m']={k:profile_mld(v['final_profiles']['density_anomaly']) for k,v in cases.items()}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps([dict(file=f.name,**summarize(json.loads(f.read_text()))) for f in a.inputs],indent=2)+'\n')
