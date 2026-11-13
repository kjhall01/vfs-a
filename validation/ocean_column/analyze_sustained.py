"""Compact sustained-column evidence and conservative profile comparisons."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def thickness(profile):
    # Recover interfaces from layer-centre depths, including rain expansion.
    edges=[0.]
    for center in profile['depth']:edges.append(2*center-edges[-1])
    return np.diff(edges)


def compare(coarse,fine):
    a,b=coarse['profiles'][-1],fine['profiles'][-1]
    ha,hb=thickness(a),thickness(b)
    ea,eb=np.r_[0,np.cumsum(ha)],np.r_[0,np.cumsum(hb)]
    overlap=np.maximum(0,np.minimum(ea[1:,None],eb[None,1:])-np.maximum(ea[:-1,None],eb[None,:-1]))
    if not np.allclose(overlap.sum(1),ha,atol=1e-8,rtol=0):raise ValueError('different column extents')
    result={}
    for name in ('u','v','temperature','salinity','tke','diffusivity'):
        x=np.array(a[name]);y=overlap@np.array(b[name])/ha
        rms=float(np.sqrt(np.sum(ha*(x-y)**2)/ha.sum()))
        scale=float(np.sqrt(np.sum(ha*y*y)/ha.sum()))
        result[name]={'rms_difference':rms,'relative_l2':rms/scale if scale else None}
    x=np.array([a['u'],a['v']]);y=np.array([overlap@np.array(b[k])/ha for k in ('u','v')])
    result['velocity_relative_l2']=float(np.sqrt(np.sum(ha*(x-y)**2)/np.sum(ha*y*y)))
    result['temperature_mld_difference_m']=a['temperature_mld_m']-b['temperature_mld_m']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];raws={};omissions=[]
    for path in sorted(a.input.glob('*.json')):
        if path.name.endswith('summary.json'):continue
        r=json.loads(path.read_text());raws[path.stem]=r
        row={k:v for k,v in r.items() if k!='profiles'}
        d=r['limiter'];den=d.get('active_pairs',0)
        row['limited_pair_fraction']=d.get('limited_pairs',0)/den if den else 0
        row['accepted_fractions']={k:d['accepted_'+k]/d['proposed_'+k] if d.get('proposed_'+k,0) else None for k in ('exchange','salt','enthalpy')}
        row['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();row['file']=path.name
        row['initial_profile']=r['profiles'][0];row['final_profile']=r['profiles'][-1]
        row['hourly']=[{k:q[k] for k in ('time_s','energy_minus_input_J_kg','temperature_mld_m','limiter')} for q in r['profiles']]
        out=json.dumps(row,indent=2)+'\n'
        if len(out.encode())>131072:
            row.pop('hourly');out=json.dumps(row,indent=2)+'\n'
        if len(out.encode())>131072:raise ValueError('compact result exceeds cap')
        (a.output/path.name).write_text(out)
        if path.stat().st_size>131072:omissions.append(dict(file=path.name,bytes=path.stat().st_size,reason='raw profiles omitted; above 128 KiB'))
        rows.append({k:v for k,v in row.items() if k not in ('initial_profile','final_profile','hourly')})
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'omissions.json').write_text(json.dumps(omissions,indent=2)+'\n')
    def pick(n,dt,s,k=1e-4,safety=.5):
        return next((r for r in raws.values() if (r['layers'],r['dt_s'],r['scenario'],r['seed_tke'],r['backtrack_safety'])==(n,dt,s,k,safety) and r['status']=='completed'),None)
    comparisons=[]
    def add(kind,left,right,**meta):
        if left and right:comparisons.append(dict(kind=kind,**meta,metrics=compare(left,right)))
    for s in ('wind','cooling','rain_heating'):
        for n in (16,32,64):add('timestep',pick(n,300,s),pick(n,150,s),scenario=s,layers=n)
        for n in (16,32):add('vertical',pick(n,150,s),pick(2*n,150,s),scenario=s,coarse_layers=n)
    add('finer_timestep',pick(32,150,'wind'),pick(32,75,'wind'),layers=32)
    for k in (1e-6,1e-3):add('seed',pick(32,150,'wind',k),pick(32,150,'wind'),seed=k)
    for safety in (.25,.75):add('limiter_safety',pick(32,150,'wind',safety=safety),pick(32,150,'wind'),safety=safety)
    (a.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')

if __name__=='__main__':main()
