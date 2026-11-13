"""Optional scientific profile figures from compact sustained-campaign records."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in a.input.glob('*.json'):
        r=json.loads(path.read_text())
        if isinstance(r,dict) and 'final_profile' in r and r.get('status')=='completed':rows.append(r)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fields=[('u','Eastward current (m/s)'),('temperature','Potential temperature (°C)'),('tke','TKE (m²/s²)')]
    fig,axes=plt.subplots(3,3,figsize=(10,8),constrained_layout=True,sharey=True)
    for i,scenario in enumerate(('wind','cooling','rain_heating')):
        for j,(field,label) in enumerate(fields):
            ax=axes[i,j]
            for n in (16,32,64):
                r=next((r for r in rows if (r['scenario'],r['layers'],r['dt_s'],r['seed_tke'],r['backtrack_safety'])==(scenario,n,150,1e-4,.5)),None)
                if r is None:continue
                q=r['final_profile'];ax.plot(q[field],q['depth'],label=f'{n} layers')
            ax.set(xlabel=label,ylim=(100,0),title=scenario.replace('_',' '))
            if field=='tke':ax.set_xscale('log')
            if j==0:ax.set_ylabel('Depth (m)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Active scalar mixing: 48-hour profiles, dt=150 s\nResolution comparison; no observational reference')
    fig.savefig(a.output/'profiles.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,4),constrained_layout=True,sharey=True)
    for ax,(field,label) in zip(axes,fields):
        for safety in (.25,.5,.75):
            r=next((r for r in rows if (r['scenario'],r['layers'],r['dt_s'],r['seed_tke'],r['backtrack_safety'])==('wind',32,150,1e-4,safety)),None)
            if r is None:continue
            q=r['final_profile'];ax.plot(q[field],q['depth'],label=f'factor={safety}')
        ax.set(xlabel=label,ylim=(100,0));ax.grid(alpha=.2);ax.legend(fontsize=8)
        if field=='tke':ax.set_xscale('log')
    axes[0].set_ylabel('Depth (m)');fig.suptitle('Trial-reduction factor sensitivity: 32 layers, dt=150 s, wind, 48 h')
    fig.savefig(a.output/'limiter-sensitivity.svg');plt.close(fig)
    for path in a.output.glob('*.svg'):
        print(path.name,path.stat().st_size,'bytes; omit from upload if above 131072')

if __name__=='__main__':main()
