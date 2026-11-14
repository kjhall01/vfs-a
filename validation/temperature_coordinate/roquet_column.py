"""Matched-buoyancy warm-column JMD95/Roquet comparison, isolated physics.

Uses explicit GSW initialization, same wind/heat forcing and mixing substeps.
No horizontal dynamics/ALE; no physical-accuracy acceptance threshold.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from matched_eos import build,anomaly,rms
from roquet_campaign import model
from roquet import convert_jmd95_initialization
from analyze_matched import profile_mld


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--layers',type=int,default=32);p.add_argument('--steps',type=int,default=144);p.add_argument('--dt',type=float,default=600.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    j,_,sj,_,h0,meta,f=build('mixing',32,a.layers,True);s=sj.ocean;r=model(s.mesh,s.column,8)
    z=r.ocean.energy.base.height(s);xyz=s.mesh.cell_xyz
    fields=convert_jmd95_initialization(potential_temperature_ipts68=s.temperature(),practical_salinity=s.salinity(),pressure_dbar=-s.column.rho0*s.column.gravity*z/1e4,longitude=torch.atan2(xyz[:,1:2],xyz[:,0:1])*180/math.pi,latitude=torch.asin(xyz[:,2:3])*180/math.pi)
    sa=fields['absolute_salinity'];c=r.config;target=anomaly(j,s)
    d0=c.t+2*c.tt*c.CT0-c.tz*z;rhs=target-c.s*(sa-c.SA0);disc=d0.square()+4*c.tt*rhs
    if (disc<=0).any():raise ValueError('no matched CT root')
    ct=c.CT0+2*rhs/(d0-torch.sqrt(disc))
    sr=r.state(u_e=s.u_e,h=s.h,conservative_temperature=ct,absolute_salinity=sa,tracer=s.tracer_h/s.h[...,None],tke=sj.tke_h/s.h)
    report=dict(layers=a.layers,dt=a.dt,requested_steps=a.steps,initial_anomaly_mismatch=float((c.anomaly(ct,sa,z)-target).abs().max()),cases={})
    final={}
    with torch.no_grad():
        for name,m,start in [('jmd95',j,sj),('roquet',r,sr)]:
            state=start;initial=m.total_energy(start);mass=(m.energy.mass_area(s)*s.h).sum();external=0.;count=0;error=None
            try:
                for n in range(a.steps):
                    st,k,br=m.boundary.apply(m.facade,state.ocean,state.tke_h,a.dt,surface_stress=f['surface_stress'],_physical=state.ocean)
                    st,hr=m.heating.apply(m.facade,st,a.dt,shortwave=f['shortwave'],surface_heat=f['surface_heat'],_physical=st)
                    st,k,mr=m.closure.apply(m.ocean,st,k,a.dt);state=type(state)(st,k);external+=br.surface_work+hr.external_heat;count=n+1
            except (ValueError,FloatingPointError) as exc:error=str(exc)
            st=state.ocean;density=anomaly(m,st) if name=='jmd95' else c.anomaly(st.temperature(),st.salinity(),r.ocean.energy.base.height(st))
            profile=density.mean(0).tolist()
            report['cases'][name]=dict(status='rejected' if error else 'completed',error=error,completed_steps=count,energy_minus_inputs_J_kg=float((m.total_energy(state)-initial-external)/mass),min_tke=float((state.tke_h/st.h).min()),density_profile=profile,interpolated_mean_profile_mld_m=profile_mld(profile))
            final[name]=(st,density);a.output.write_text(json.dumps(report,indent=2)+'\n');print(name,report['cases'][name]['status'],flush=True)
        if all(v['status']=='completed' for v in report['cases'].values()):
            x,y=final['jmd95'],final['roquet'];area=s.mesh.cell_area[:,None];edge=(s.mesh.dc_edge*s.mesh.dv_edge)[:,None]
            report['difference']=dict(density_rms_kg_m3=rms(x[1]-y[1],area),velocity_rms_m_s=rms(x[0].u_e-y[0].u_e,edge))
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    return int(any(v['status']!='completed' for v in report['cases'].values()))
if __name__=='__main__':raise SystemExit(main())
