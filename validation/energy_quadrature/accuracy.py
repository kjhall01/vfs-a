"""Compare discrete JMD95 energies/forces to higher quadrature, not analytical truth."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean,OceanColumn,OceanState
from vfs.ocean.jmd95 import JMD95Dynamics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=0,rotation_scheme='compatible')
    rows=[]
    with torch.no_grad():
        for depth in [100.,4000.,6000.]:
            for layers in [8,32,64]:
                column=OceanColumn(layers,total_depth=depth)
                r=torch.linspace(0,1,layers+1)[None,:]
                interfaces=depth*(r+.15*mesh.cell_xyz[:,:1]*torch.sin(2*torch.pi*r)/(2*torch.pi))
                h=interfaces[:,1:]-interfaces[:,:-1];z=(interfaces[:,1:]+interfaces[:,:-1])/2
                for profile in ['smooth','front','reference']:
                    t=2+20*torch.exp(-z/(depth*.15))+.5*mesh.cell_xyz[:,1:2]
                    s=35+.6*torch.tanh((z/depth-.25)*8)+.1*mesh.cell_xyz[:,2:3]
                    if profile=='front':
                        t=t+3*torch.tanh(20*mesh.cell_xyz[:,:1]);s=s+.5*torch.tanh(20*mesh.cell_xyz[:,1:2])
                    if profile=='reference':t=torch.full_like(h,10.);s=torch.full_like(h,35.)
                    state=OceanState.from_temperature(torch.zeros(mesh.nEdges,layers),h,t,s,mesh=mesh,column=column)
                    outputs={}
                    for q in [2,3,4,8,12]:
                        dyn=JMD95Dynamics(quadrature_order=q);energy=dyn.energy
                        total,g=energy.stratification(state)
                        force=dyn.tendency_terms(state)['u_e']['hamiltonian_pressure']
                        mass=(1025*mesh.cell_area[:,None]*h).sum()
                        outputs[q]=(total/mass,g,force)
                    e_ref,g_ref,f_ref=outputs[12]
                    for q,(e,g,f) in outputs.items():
                        errors={k:float((v-g_ref[k]).norm()/g_ref[k].norm().clamp_min(1e-300)) for k,v in g.items()}
                        rows.append(dict(depth_m=depth,layers=layers,profile=profile,order=q,points=q*q,
                            anomaly_energy_error_J_kg=float(e-e_ref),gradient_relative_l2=errors,
                            pressure_error_rms_m_s2=float((f-f_ref).square().mean().sqrt()),
                            pressure_reference_rms_m_s2=float(f_ref.square().mean().sqrt()),
                            pressure_relative_l2=float((f-f_ref).norm()/f_ref.norm().clamp_min(1e-300))))
    result=dict(reference='same piecewise-uniform states with order 12; order 8 convergence control included',torch=torch.__version__,rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for q in [2,3,4,8]:
        selected=[x for x in rows if x['order']==q and x['profile']!='reference']
        print(q,'max pressure rms',max(x['pressure_error_rms_m_s2'] for x in selected),'max relative',max(x['pressure_relative_l2'] for x in selected),'max energy J/kg',max(abs(x['anomaly_energy_error_J_kg']) for x in selected),flush=True)


if __name__=='__main__':main()
