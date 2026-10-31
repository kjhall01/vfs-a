"""Proposed Gibbs numerical revision: actual faces, bounded stages and probes.

Records intended numerical changes without generating replacement fixtures.
The numerical revision must be selected explicitly; historical baseline stays.
"""
import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil

import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.atmosphere.gibbs_means import gibbs_face


def sibling(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


energy=sibling('run_energy_verification')
probe=sibling('investigate_condensate_means')
common=energy.common


def face_checks():
    thermo=probe.MixtureThermodynamics()
    rows=[]
    cases=probe.cases()
    q=[.9889,.01,.001,.0001]
    for epsilon in (.001*(1-1e-6),.001*(1+1e-6)):
        cases.append((f'near_series_boundary_{epsilon}',280.,1e5,q,280.*(1+epsilon),1e5*(1-epsilon),q))
    for name,Ti,pi,qi,Tj,pj,qj in cases:
        Ti,pi,qi,Tj,pj,qj=[torch.as_tensor(x,dtype=torch.float64) for x in (Ti,pi,qi,Tj,pj,qj)]
        rhoi=pi/(thermo.gas_constant(qi)*Ti)
        rhoj=pj/(thermo.gas_constant(qj)*Tj)
        density_i=torch.cat((rhoi.reshape(1),rhoi*qi[:2]))
        density_j=torch.cat((rhoj.reshape(1),rhoj*qj[:2]))
        x=(density_j-density_i)/(density_j+density_i)
        # Outside the cancellation-safe series, the existing log difference
        # has subtraction conditioning. Report it; do not alter gas logmeans.
        condition=torch.ones_like(x)
        exact=x*x>=1e-6
        condition[exact]=(density_i[exact].log().abs()+density_j[exact].log().abs())/(density_j[exact].log()-density_i[exact].log()).abs()
        actual=gibbs_face(thermo,Ti,pi,qi,Tj,pj,qj)
        reference=probe.candidate(thermo,Ti,pi,qi,Tj,pj,qj)
        row=probe.decimal_identity(thermo,(Ti,pi,qi),(Tj,pj,qj),actual)
        row.update(case=name, logmean_roundoff_condition=float(condition.max().clamp_min(1)), candidate_difference=[float((x-y).abs().max()) for x,y in zip(actual,reference)],
                   constant_fraction_error=float((actual[1]-qi).abs().max()) if torch.equal(qi,qj) else None)
        # Deliberately retain the WRONG entropy trace to ensure the independent
        # identity diagnostic responds; no production tensor is modified.
        row['corrupted_entropy_residual']=probe.decimal_identity(thermo,(Ti,pi,qi),(Tj,pj,qj),
                                                               (actual[0]+1,actual[1],actual[2]))
        rows.append(row)
    return rows


def make_case(kind,reverse=False,near=False):
    mesh=VoronoiMesh(n_cells=16,lloyd_iters=0,verbose=False)
    a,b=make_hybrid_coefficients(3,p_top=5000.)
    model=MoistAtmosphere(mesh,CharneyPhillipsColumn(3,a,b,phi_full_mode='direct_u_gl2_adjoint'))
    x=mesh.cell_xyz[:,:1]
    scale=1e-10 if near else 1.
    pressure=100000.+scale*8000*x
    temperature=280.+scale*(15*x+torch.arange(4)[None,:])
    liquid,ice=(0.,0.) if kind=='zero' else (.001,.0001)
    if kind=='one_sided':
        liquid=.001*(x>0)
        ice=.0001*(x<0)
    state=model.state_from_fields(temperature=temperature,surface_pressure=pressure,
        normal_wind=model.normal_wind(eastward=-10. if reverse else 10.,northward=2.),
        vapor=.01,liquid=liquid,ice=ice)
    return model,state


def fractions(model,state):
    mass,species=energy.independent_amounts(state,model.dynamics.kernel.gravity)
    return species/mass[:,None]


def model_checks():
    rows=[];raw={}
    for kind in ('constant','zero','one_sided'):
        for reverse in (False,True):
            for near in (False,True):
                name=f'{kind}-reverse{reverse}-near{near}'
                model,initial=make_case(kind,reverse,near)
                result,budgets,stages=energy.record_stages(model,initial,10.)
                public=model(initial,10.)
                if not all(torch.equal(getattr(result,k),getattr(public,k)) for k in energy.MOIST):
                    raise AssertionError('Observed SSPRK3 differs from public step')
                q0=fractions(model,initial)
                details=[]
                for stage,budget in zip(stages,budgets):
                    output=replace(initial,**stage['euler_output'])
                    info=stage['info']
                    entry=dict(budget)
                    entry['constant_fraction_error']=(float((fractions(model,output)-q0).abs().max())
                                                      if kind!='one_sided' else None)
                    entry['high_flux_constant_error_Pa_m2']=(float((info['high_integrated_flux']-
                        10.*info['carrier'][:,None]*q0[0]).abs().max()) if kind!='one_sided' else None)
                    entry['high_flux_scale_Pa_m2']=float((10.*info['carrier']).abs().max())
                    details.append(entry)
                rows.append(dict(case=name,stages=details,
                    final_constant_fraction_error=float((fractions(model,result)-q0).abs().max()) if kind!='one_sided' else None,
                    final_minimum_fraction=fractions(model,result).amin(0).tolist()))
                raw[name]=dict(initial=energy.payload(initial),stages=stages,final=energy.payload(result),
                               mesh=model.mesh.state_dict(),column=model.column.state_dict())
    return rows,raw


def zero_derivative():
    """One-sided full-step composition probe at zero cloud; report switches."""
    model,state=make_case('zero')
    leaf=state.water_content.clone().requires_grad_()
    weights=torch.sin(torch.arange(leaf.numel()).reshape(leaf.shape)*.37+.2)
    def value(w):
        output,info=model.transport.step_with_info(replace(state,water_content=w),1.)
        objective=(output.water_content*weights/100.).mean()+output.entropy_content.mean()/1e7
        return objective,info
    base,base_info=value(leaf)
    grad,=torch.autograd.grad(base,leaf)
    direction=torch.zeros_like(leaf)
    direction[...,1]=1.
    ad=(grad*direction).sum()
    rows=[]
    for h in (1e-2,1e-3,1e-4,1e-5):
        obj,info=value(leaf+h*direction)
        fd=(obj-base)/h
        rows.append(dict(h=h,fd=float(fd),absolute_error=float(abs(fd-ad)),
                         same_limiter_masks=all(torch.equal(x['beta']<1,y['beta']<1) for x,y in zip(info,base_info)),
                         substeps=[x['substeps'] for x in info],limited_faces=[x['limited_faces'] for x in info]))
    return dict(objective=float(base),ad=float(ad),gradient_finite=bool(torch.isfinite(grad).all()),
                base_substeps=[x['substeps'] for x in base_info],
                base_limited_faces=[x['limited_faces'] for x in base_info],one_sided=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--numerical-revision',choices=[common.NUMERICAL_COMMIT],required=True)
    parser.add_argument("--structural-revision", choices=[common.STRUCTURAL_COMMIT])
    args=parser.parse_args()
    config=dict(dtype='float64',device='cpu',cells=16,levels=3,dt_s=10.,
                scope='actual face and production Euler/SSPRK3; no baseline adoption')
    with common.recorded_run(args.repo,args.output,'GIBBS-FIX-REVIEW',config,__file__,args.numerical_revision, structural_revision=args.structural_revision):
        for name in ('investigate_condensate_means.py','run_energy_verification.py'):
            shutil.copyfile(Path(__file__).with_name(name),args.output/name)
        common.write_json(args.output/'faces.json',face_checks())
        metrics,raw=model_checks()
        common.write_json(args.output/'stages.json',metrics)
        torch.save(energy.archive(raw),args.output/'states-and-ledgers.pt')
        common.write_json(args.output/'zero-derivative.json',zero_derivative())


if __name__=='__main__':
    main()
