"""Layerwise thermal projection with local H and entropy constraints.

Sequential weighted minimum-change linearized solves; not a globally optimal
nonlinear optimizer. Weights are layer mass fractions. Full nonlinear checks
remain authoritative; an unsuccessful or rank-deficient solve rejects explicitly.
"""
from dataclasses import dataclass, replace
import torch
from ocean_entropy_candidate import validate
from enthalpy_ale_candidate import EnthalpyRemapper
from hamiltonian_ale_candidate import column_terms, HamiltonianRemapReport


@dataclass(frozen=True)
class AdaptiveThermalReport(HamiltonianRemapReport):
    entropy_constraint_columns: torch.Tensor
    iterations: int


def minimum_change_step(displacement, energy_gradient, entropy_gradient,
                        energy_residual, entropy_change, weights, entropy_tolerance):
    """Solve the mass-weighted two-constraint linearized projection per column.

    Orthogonalize the entropy gradient against the energy gradient instead of
    subtracting two nearly equal entries of a 2x2 normal-equation determinant.
    Returns a step and the active entropy constraint; no forward autograd.
    """
    g,s,w=energy_gradient,entropy_gradient,weights
    gg=(g*g/w).sum(-1)
    if (gg<=0).any():raise ValueError('degenerate thermal energy gradient')
    step=-displacement+(g/w)*( ((g*displacement).sum(-1)-energy_residual)/gg )[:,None]
    predicted=entropy_change+(s*step).sum(-1)
    active=predicted < -entropy_tolerance
    orthogonal=s-g*((s*g/w).sum(-1)/gg)[:,None]
    norm=(orthogonal.square()/w).sum(-1)
    scale=(s.square()/w).sum(-1)
    deficient=norm<=64*torch.finfo(w.dtype).eps*scale
    if (active&deficient).any():
        raise ValueError('energy and entropy constraints lack independent thermal directions')
    denominator=torch.where(active,norm,torch.ones_like(norm))
    step=step+(orthogonal/w)*torch.where(active,-predicted/denominator,torch.zeros_like(norm))[:,None]
    return step,active


class AdaptiveThermalRemapper:
    def __init__(self, *, method='plm'):
        self.remapper=EnthalpyRemapper(method=method)

    def remap_with_report(self,model,state,target_h,energy):
        provisional,report=self.remapper.remap_with_report(model,state,target_h,energy)
        before=validate(model,state,energy);after=provisional.physical_state(energy.reference)
        old,_=column_terms(energy,before);mass=energy.mass_area(state);wet=mass[:,0]>0
        base=energy.base;reference=energy.reference;theta0=after.temperature();eps=torch.finfo(theta0.dtype).eps
        depth=after.h.sum(-1);weights=after.h/depth[:,None]
        column_mass=mass[:,0]*depth;safe_mass=torch.where(wet,column_mass,torch.ones_like(column_mass))
        energy_scale=sum(v.abs() for v in old.values()).clamp_min(1)
        entropy_scale=(mass*state.entropy_h.abs()).sum(-1).clamp_min(1)
        energy_tolerance=256*eps*energy_scale
        entropy_tolerance=512*eps*entropy_scale
        if provisional is state:
            zero=state.h.sum(-1)*0
            return state,AdaptiveThermalReport(**report.__dict__,column_energy_residual=zero,
                column_entropy_change=zero,temperature_increment=torch.zeros_like(state.h),
                provisional_energy_change=report.energy_change,entropy_constraint_columns=torch.zeros_like(wet),iterations=0)
        node,qweight=base._quadrature(after.h)
        z=base.height(after)[...,None]+after.h[...,None]*(node-.5)
        theta=theta0;ever_active=torch.zeros_like(wet)
        for iteration in range(24):
            physical=after.with_tensors(after.u_e,after.h,after.h*theta,after.salt_h,after.tracer_h)
            terms,_=column_terms(energy,physical)
            ref=reference.evaluate(physical.salinity(),theta)
            residual=sum(terms[k]-old[k] for k in old)
            entropy_change=(mass*(after.h*ref['entropy']-state.entropy_h)).sum(-1)
            if iteration>=2 and (residual.abs()<=energy_tolerance).all() and (entropy_change>=-entropy_tolerance).all():break
            _,pt,_,_=base.parcel_potential(theta[...,None],physical.salinity()[...,None],z)
            g=weights*(ref['h_theta']+(pt*qweight).sum(-1))
            s=weights*ref['entropy_theta']
            step,active=minimum_change_step(theta-theta0,g,s,residual/safe_mass,
                entropy_change/safe_mass,weights,entropy_tolerance/safe_mass)
            ever_active=ever_active|(active&wet)
            step=torch.where(wet[:,None],step,torch.zeros_like(step))
            # Fraction-to-boundary damping changes the attempted step, never
            # clips a state or substitutes a floor for invalid thermodynamics.
            low=-2.5/reference.to_90+1024*eps;high=40./reference.to_90-1024*eps
            room=torch.where(step>0,high-theta,theta-low)
            ratio=torch.where(step!=0,room/torch.where(step!=0,step.abs(),torch.ones_like(step)),torch.ones_like(step))
            fraction=torch.minimum(torch.ones_like(depth),.9*ratio.amin(-1))
            if (fraction<=0).any():raise ValueError('thermal projection has no interior-domain step')
            theta=theta+fraction[:,None]*step
        eta=reference.evaluate(after.salinity(),theta,entropy_derivatives=True)['entropy']
        end=replace(provisional,entropy_h=torch.where(wet[:,None],after.h*eta,state.entropy_h))
        final=validate(model,end,energy);final_terms,_=column_terms(energy,final)
        residual=sum(final_terms[k]-old[k] for k in old)
        entropy_change=(mass*(end.entropy_h-state.entropy_h)).sum(-1)
        if (residual.abs()>energy_tolerance).any():raise FloatingPointError('adaptive thermal projection failed column energy residual check')
        if (entropy_change < -entropy_tolerance).any():raise ValueError('adaptive thermal projection decreases column entropy')
        old_h0=reference.evaluate(before.salinity(),before.temperature())['h0']
        new_h0=reference.evaluate(final.salinity(),final.temperature())['h0']
        residuals=dict(report.content_residuals);residuals['entropy_h']=(end.entropy_h-state.entropy_h).sum(-1)
        terms0,terms1=energy.terms(state),energy.terms(end)
        result=AdaptiveThermalReport(residuals,report.momentum_residual,
            {k:terms1[k]-terms0[k] for k in terms0},(final.temp_h-before.temp_h).sum(-1),
            report.changed_columns,entropy_change.sum(),(end.h*new_h0-state.h*old_h0).sum(-1),
            residual,entropy_change,theta-theta0,report.energy_change,ever_active,iteration+1)
        return end,result
