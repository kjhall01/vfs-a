"""Render compact SVGs from completed scalar campaign reports (matplotlib)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reports=[json.loads(p.read_text()) for p in sorted(args.directory.glob('*.json'))]
    initial=[r for r in reports if r.get('scope')=='initial tendencies only; no integration']
    reports=[r for r in reports if 'kind' in r]
    if any(r['status']!='completed' for r in reports):
        raise ValueError('Only completed campaigns can be plotted')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'vfs-ocean-qualification','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for r in sorted([r for r in reports if r['kind']=='wave'],key=lambda r:(r['cells'],-r['external_cfl'])):
        samples=r['samples'];t=np.array([s['time_s'] for s in samples])/r['period_s']
        phase=np.unwrap([s['phase_rad'] for s in samples])-2*np.pi*t
        label=f"{r['cells']} cells, CFL {r['external_cfl']}"
        style='--' if r['external_cfl']==.1 else '-'
        axes[0,0].plot(t,phase,label=label,linestyle=style)
        axes[0,1].plot(t,[s['envelope_over_initial'] for s in samples],label=label,linestyle=style)
        axes[1,0].plot(t,[s['off_mode_h_rms_m'] for s in samples],label=label,linestyle=style)
        axes[1,1].plot(t,[s['energy_change_J_m2'] for s in samples],label=label,linestyle=style)
    titles=['Phase error relative to continuum','Projected modal envelope (not pure damping)','Thickness outside projected mode','Total energy change (includes roundoff)']
    units=['radians','initial amplitude = 1','m RMS','J/m²']
    for ax,title,unit in zip(axes.flat,titles,units):
        ax.set(xlabel='Continuum periods',ylabel=unit,title=title);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(args.output/'waves.svg',metadata={'Date':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for profile in ['smooth','sharp']:
        rows=sorted([r for r in reports if r['kind']=='transport' and r['profile']==profile],key=lambda r:r['cells'])
        n=[r['cells'] for r in rows]
        axes[0].loglog(n,[r['fields']['salinity']['rms'] for r in rows],'o-',label=profile)
        axes[1].plot(n,[r['fields']['salinity']['overshoot'] for r in rows],'o-',label=profile)
        axes[2].plot(n,[r['fields']['passive']['overshoot'] for r in rows],'o-',label=profile)
    for ax,title,y in zip(axes,['Salinity error after one circuit','Salinity above initial maximum','Passive above initial maximum'],['RMS practical salinity','practical salinity','tracer units']):
        ax.set(xlabel='Periodic cells',ylabel=y,title=title,xticks=[32,64,128]);ax.grid(alpha=.2)
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend()
    fig.savefig(args.output/'transport.svg',metadata={'Date':None});plt.close(fig)
    rows=[dict(cells=r['cells'],diagnosis=r['constant_density_operator_diagnosis']) for r in reports if r['kind']=='balance' and r['profile']=='jet' and r['dt_s']==600]
    rows += [dict(cells=r['cells'],diagnosis=r['constant_density_operator_diagnosis']) for r in initial if r['lloyd_iterations_requested']==20]
    rows.sort(key=lambda r:r['cells'])
    if rows:
        fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
        for term in ['pressure','kinetic','rotation']:
            ax.loglog([r['cells'] for r in rows],[r['diagnosis']['term_errors_m_s2'][term]['l2'] for r in rows],'o-',label=term)
        ax.loglog([r['cells'] for r in rows],[r['diagnosis']['momentum_residual_m_s2']['l2'] for r in rows],'s--',label='total momentum imbalance')
        ax.set(xlabel='Spherical cells',ylabel='RMS acceleration error (m/s²)',title='Initial jet balance: 20 Lloyd iterations')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.savefig(args.output/'initial-balance.svg',metadata={'Date':None});plt.close(fig)


    for path in args.output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

if __name__=='__main__':main()
