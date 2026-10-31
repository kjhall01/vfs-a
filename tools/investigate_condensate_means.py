"""Offline face/CP-ledger investigation; never replaces a production operator."""
import argparse
from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path

import torch
from vfs.atmosphere.gibbs_means import gibbs_face, logmean, partial_chemical
from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics
from vfs.atmosphere.bounded_transport import limited_traces

spec = importlib.util.spec_from_file_location('manifest', Path(__file__).with_name('run_verification.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def candidate(t, Ti, pi, qi, Tj, pj, qj):
    """Experimental density means, kg/m³; full matched (s,q,alpha) tuple.

    This explicitly duplicates the short face formula for investigation only.
    The sole proposed algebraic change is condensate d*=L(rho_i,rho_j)*avg(q).
    Entropy and alpha MUST then be recomputed from the changed normalized q
    and total face density. No constant-composition detector or zero branch.
    """
    rhoi = pi/(t.gas_constant(qi)*Ti)
    rhoj = pj/(t.gas_constant(qj)*Tj)
    di, dj = rhoi[...,None]*qi, rhoj[...,None]*qj
    condensate = logmean(rhoi,rhoj)[...,None] * (0.5*(qi[...,2:]+qj[...,2:]))
    density_species = torch.cat((logmean(di[...,:2],dj[...,:2]),condensate),-1)
    density = density_species.sum(-1)
    q = density_species/density[...,None]
    eta = .5*(partial_chemical(t,Ti,pi,qi)/Ti[...,None]
              + partial_chemical(t,Tj,pj,qj)/Tj[...,None])
    cp, R = t.heat_capacity_p(q), t.gas_constant(q)
    k = t.enthalpy_reference(q)-cp*t.T_ref
    Tbar = .5*(Ti+Tj)
    s = (Tbar*(cp-R)/logmean(Ti,Tj) + Tbar*k/(Ti*Tj) - (q*eta).sum(-1)
         + .5*(rhoi*t.gas_constant(qi)+rhoj*t.gas_constant(qj))/density)
    return s,q,1/density


def decimal_identity(t, left, right, traces):
    """Independent 60-digit endpoint potentials, not production chemical helpers.

    Measure sum_a q*_a Delta mu_a + s* Delta T - alpha* Delta p, J/kg.
    Convert candidate float traces exactly to Decimal: this is a residual check
    of computed traces against independent thermodynamics, not an exact proof.
    """
    D = lambda x: Decimal.from_float(float(x))
    with localcontext() as ctx:
        ctx.prec = 60
        cp = list(map(D,[t.cp_d,t.cp_v,t.cp_l,t.cp_i]))
        gas = list(map(D,[t.R_d,t.R_v,0.,0.]))
        href = list(map(D,[0.,t.L_v_ref,0.,-t.L_f_ref]))
        pref = list(map(D,[t.p_d_ref,t.p_v_ref,1.,1.]))
        tref = D(t.T_ref)
        def mu(endpoint):
            temperature,pressure,q = endpoint
            temperature,pressure = D(temperature),D(pressure)
            q = list(map(D,q))
            R = sum(q[a]*gas[a] for a in range(4))
            result=[]
            for a in range(4):
                entropy=cp[a]*(temperature/tref).ln()+href[a]/tref
                if gas[a] != 0:
                    entropy-=gas[a]*(pressure*q[a]*gas[a]/R/pref[a]).ln()
                enthalpy=cp[a]*(temperature-tref)+href[a]
                result.append(enthalpy-temperature*entropy)
            return result
        mui,muj=mu(left),mu(right)
        s,q,alpha=traces
        terms=[D(q[a])*(muj[a]-mui[a]) for a in range(4)]
        terms += [D(s)*(D(right[0])-D(left[0])), -D(alpha)*(D(right[1])-D(left[1]))]
        residual=sum(terms)
        scale=sum(abs(x) for x in terms)
        return dict(residual_J_per_kg=float(residual), work_scale_J_per_kg=float(scale),
                    relative_to_work=float(abs(residual)/scale) if scale else None)


def cp_transfer(qi,qj,face):
    """Two cells × three CP interfaces; production limiter/ledger, toy fluxes.

    Layer amounts 100 Pa m² give CP carrier [50,100,50] in each cell.
    Layer-edge fluxes [1,2] give horizontal CP flux [.5,1.5,1] Pa m²/s.
    Four explicit vertical links complete this small graph. It is an incidence
    test with prescribed fluxes, not a spherical dycore or coupled energy test.
    """
    mass=qi.new_tensor([50.,100.,50.]*2)
    i=torch.tensor([0,1,2,0,1,3,4])
    j=torch.tensor([3,4,5,1,2,4,5])
    flux=qi.new_tensor([.5,1.5,1.,.2,-.1,-.2,.1])
    q=torch.cat((qi.expand(3,4),qj.expand(3,4)))
    high=torch.cat((face.expand(3,4),qi.expand(2,4),qj.expand(2,4)))
    target=mass.clone()
    for a,b,f in zip(i,j,flux):
        target[a]-=f
        target[b]+=f
    _,_,info=limited_traces(mass,q,torch.zeros_like(flux),high,torch.zeros(7,3),
                           i,j,flux,1.,target=target,floors=qi.new_tensor([1e-14,1e-14,0.,0.]))
    ledger=info['amounts']
    # Independent direct face sums, not production divergence/index_add.
    expected=mass[:,None]*q
    for a,b,transfer in zip(i,j,info['integrated_flux']):
        expected[a]-=transfer
        expected[b]+=transfer
    return dict(maximum_ledger_incidence_error=float((ledger-expected).abs().max()),
                species_inventory_change=(ledger.sum(0)-(mass[:,None]*q).sum(0)).tolist(),
                minimum_amount=float(ledger.min()),limited_faces=info['limited_faces'],
                constant_fraction_error=float((ledger/target[:,None]-q).abs().max())
                    if torch.equal(qi,qj) else None,
                constant_flux_error=float((info['integrated_flux']-flux[:,None]*qi).abs().max())
                    if torch.equal(qi,qj) else None)


def cases():
    q=[.9889,.01,.001,.0001]
    rows=[('equal',280.,1e5,q,280.,1e5,q),
          ('constant_pressure_jump',280.,1e5,q,280.,9e4,q),
          ('constant_temperature_jump',240.,1e5,q,310.,1e5,q),
          ('constant_both_jumps',240.,3e4,q,310.,1e5,q),
          ('constant_equal_density',240.,6e4,q,300.,7.5e4,q),
          ('anticorrelated_cloud_density',280.,1e3,[.59,.01,.4,0.],280.,1e5,[.99,.01,0.,0.]),
          ('varying_composition',240.,3e4,q,310.,1e5,[.977,.02,.002,.001]),
          ('zero_condensates',240.,3e4,[.99,.01,0.,0.],310.,1e5,[.99,.01,0.,0.]),
          ('one_sided_cloud',240.,3e4,[.99,.01,0.,0.],310.,1e5,q),
          ('opposite_cloud_species',240.,3e4,[.989,.01,.001,0.],310.,1e5,[.989,.01,0.,.001])]
    for eps in (1e-3,1e-6,1e-9,1e-12):
        rows.append((f'near_equal_{eps}',280.,1e5,q,280.*(1+eps),1e5*(1-eps),q))
    generator=torch.Generator().manual_seed(31)
    for k in range(100):
        water=.01*torch.rand(2,3,generator=generator)
        fractions=torch.cat((1-water.sum(-1,keepdim=True),water),-1)
        temp=180+140*torch.rand(2,generator=generator)
        pressure=1000+99000*torch.rand(2,generator=generator)
        rows.append((f'varying_{k}',temp[0],pressure[0],fractions[0],temp[1],pressure[1],fractions[1]))
    return rows


def measure():
    thermo=MixtureThermodynamics()
    rows=[]
    for name,Ti,pi,qi,Tj,pj,qj in cases():
        Ti,pi,qi,Tj,pj,qj=[torch.as_tensor(x,dtype=torch.float64) for x in (Ti,pi,qi,Tj,pj,qj)]
        outputs={label:fn(thermo,Ti,pi,qi,Tj,pj,qj) for label,fn in [('production',gibbs_face),('candidate',candidate)]}
        row=dict(case=name,inputs=[float(Ti),float(pi),qi.tolist(),float(Tj),float(pj),qj.tolist()])
        for label,fn in [('production',gibbs_face),('candidate',candidate)]:
            s,q,alpha=outputs[label]
            reverse=fn(thermo,Tj,pj,qj,Ti,pi,qi)
            residual=decimal_identity(thermo,(Ti,pi,qi),(Tj,pj,qj),(s,q,alpha))
            rhoi=pi/(thermo.gas_constant(qi)*Ti)
            rhoj=pj/(thermo.gas_constant(qj)*Tj)
            di,dj=rhoi*qi[2:],rhoj*qj[2:]
            dc=q[2:]/alpha
            residual['condensate_density_interval_excess_kg_m3']=(torch.maximum(dc-torch.maximum(di,dj),torch.minimum(di,dj)-dc)).tolist()
            residual.update(entropy=float(s),fractions=q.tolist(),specific_volume=float(alpha),
                            symmetry_max=[float((a-b).abs().max()) for a,b in zip((s,q,alpha),reverse)],
                            fraction_sum_error=float(q.sum()-1),minimum_fraction=float(q.min()),
                            constant_trace_error=float((q-qi).abs().max()) if torch.equal(qi,qj) else None,
                            cp_ledger=cp_transfer(qi,qj,q))
            row[label]=residual
        # Deliberate unpaired species-only substitution is a negative control.
        row['unpaired_species_only']=decimal_identity(thermo,(Ti,pi,qi),(Tj,pj,qj),
                                                      (outputs['production'][0],outputs['candidate'][1],outputs['production'][2]))
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-revision", choices=[common.STRUCTURAL_COMMIT])
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    configuration=dict(scope='offline face means and toy CP ledger only',dtype='float64',device='cpu',seed=31,
                       candidate='L(total_density_i,total_density_j)*arithmetic_condensate_fraction',
                       independent_reference='60-digit Decimal endpoint chemical potentials')
    with common.recorded_run(args.repo,args.output,'C5-CONSTANT-COMPOSITION-INVESTIGATION',configuration,__file__,structural_revision=args.structural_revision):
        common.write_json(args.output/'faces.json',measure())


if __name__=='__main__':
    main()
