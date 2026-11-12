"""Compact evidence and scientific figures from raw campaign output.

Never copies an oversized raw profile file; lists omitted files explicitly.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];omissions=[]
    for path in sorted(a.input.glob('*.json')):
        if path.name=='summary.json':continue
        raw=json.loads(path.read_text())
        row={k:v for k,v in raw.items() if k!='profiles'}
        row.update(raw_file=path.name,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row['initial_profile']=raw['profiles'][0];row['final_profile']=raw['profiles'][-1]
        row['hourly_energy_error_J_kg']=[[q['time_s'],q['energy_minus_input_J_kg']] for q in raw['profiles']]
        if raw['analytic_wind_transport'] is not None and raw['completed_hours']>0:
            expected=np.array(raw['analytic_wind_transport'])
            row['transport_relative_error']=float(np.linalg.norm(np.array(raw['transport'])-expected)/np.linalg.norm(expected))
        rows.append(row)
        if path.stat().st_size>131072:omissions.append(dict(file=path.name,bytes=path.stat().st_size,reason='raw hourly profiles exceed 128 KiB; omitted'))
    # Each case is a compact independent result, not chunks of oversized files.
    for row in rows:
        target=a.output/row['raw_file']
        text=json.dumps(row,indent=2)+'\n'
        if len(text.encode())>131072:raise ValueError('compact case unexpectedly exceeds cap')
        target.write_text(text)
    summary=[{k:v for k,v in r.items() if k not in ('initial_profile','final_profile','hourly_energy_error_J_kg')} for r in rows]
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'omissions.json').write_text(json.dumps(omissions,indent=2)+'\n')
    controls={(r['layers'],r['dt_s']):r['final_profile'] for r in rows if not r['mix_scalars']}
    def velocity(n,dt):
        q=controls[n,dt]
        return np.array([q['u'],q['v']])
    comparisons=[]
    for n in (16,32,64):
        coarse,fine=velocity(n,300),velocity(n,150)
        comparisons.append(dict(kind='timestep',layers=n,relative_velocity_difference=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))))
    for n in (16,32):
        coarse=velocity(n,150);fine=velocity(2*n,150).reshape(2,n,2).mean(-1)
        comparisons.append(dict(kind='vertical',coarse_layers=n,fine_layers=2*n,relative_velocity_difference=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))))
    (a.output/'profile-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    colors={16:'#4477aa',32:'#ee7733',64:'#228833'}
    for n in (16,32,64):
        active=sorted([r for r in rows if r['layers']==n and r['scenario']=='wind' and r['mix_scalars'] and r['seed_tke']==1e-4],key=lambda r:r['dt_s'])
        axes[0,0].plot([r['dt_s'] for r in active],[r['completed_hours']*3600 for r in active],'o-',color=colors[n],label=f'{n} layers')
        control=sorted([r for r in rows if r['layers']==n and not r['mix_scalars']],key=lambda r:r['dt_s'])
        axes[0,1].plot([r['dt_s'] for r in control],[r['transport_relative_error'] for r in control],'o-',color=colors[n],label=f'{n} layers')
        fine=next((r for r in control if r['dt_s']==150),None)
        if fine:
            q=fine['final_profile'];depth=q['depth']
            axes[1,0].plot(q['u'],depth,color=colors[n],label=f'{n} layers')
            axes[1,1].semilogx(q['tke'],depth,color=colors[n],label=f'{n} layers')
    axes[0,0].set(xscale='log',xlabel='Timestep (s)',ylabel='Accepted duration (s)',title='Active scalar mixing: rejection time')
    axes[0,1].set(xscale='log',yscale='log',xlabel='Timestep (s)',ylabel='Relative transport error',title='48 h controls: analytic integral reference')
    axes[1,0].set(xlabel='Eastward current (m/s)',ylabel='Depth (m)',title='Controls only: 48 h current, dt=150 s',ylim=(50,0))
    axes[1,1].set(xlabel='TKE (m²/s²)',ylabel='Depth (m)',title='Controls only: 48 h TKE, dt=150 s',ylim=(50,0))
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Prescribed-length ocean TKE: column campaign\nScalar-mixing cases fail; controls do not qualify entrainment',fontsize=12)
    fig.savefig(a.output/'campaign.png',dpi=110)
    plt.close(fig)
    figure=a.output/'campaign.png'
    if figure.stat().st_size>131072:
        omissions.append(dict(file=figure.name,bytes=figure.stat().st_size,reason='generated figure exceeds 128 KiB; omitted from publication'))
        (a.output/'omissions.json').write_text(json.dumps(omissions,indent=2)+'\n')

if __name__=='__main__':main()
