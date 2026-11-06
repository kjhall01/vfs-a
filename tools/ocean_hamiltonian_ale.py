"""Column-local full-Hamiltonian closure after potential-enthalpy ALE.

The scalar root is energy-derived; a uniform temperature increment within each
column is the explicit allocation policy, not a unique consequence of H. No
cross-column redistribution, salt/momentum change, clipping or forward AD.
"""
from dataclasses import dataclass, replace
import torch
from ocean_entropy_candidate import validate
from enthalpy_ale_candidate import EnthalpyRemapper, EnthalpyRemapReport


def column_terms(energy, physical):
    """Exact local decomposition of the existing quadrature Hamiltonian.

    K_c=sum_k h_ck * (dK/dh)_ck uses the actual interpolation transpose.
    Its sum is the original edge kinetic energy, including the coast mask.
    The returned temperature derivative holds thickness, salt and velocity fixed.
    """
    base=energy.base;h=physical.h
    mass=energy.mass_area(physical)
    node,weight=base._quadrature(h)
    z=base.height(physical)[...,None]+h[...,None]*(node-.5)
    psi,pt,_,_=base.parcel_potential(physical.temperature()[...,None],physical.salinity()[...,None],z)
    ref=energy.reference.evaluate(physical.salinity(),physical.temperature())
    terms=dict(kinetic=(h*base.mechanical.kinetic_thickness_gradient(physical)).sum(-1),
        free_surface=(.5*base.gravity*mass*physical.column.eta(h).square()).sum(-1),
        dynamic_enthalpy=(mass*h*(psi*weight).sum(-1)).sum(-1),
        reference_thermal=(mass*h*ref['h0']).sum(-1))
    derivative=(mass*h*(ref['h_theta']+(pt*weight).sum(-1))).sum(-1)
    return terms,derivative


@dataclass(frozen=True)
class HamiltonianRemapReport(EnthalpyRemapReport):
    column_energy_residual: torch.Tensor
    column_entropy_change: torch.Tensor
    temperature_increment: torch.Tensor
    provisional_energy_change: dict


class HamiltonianRemapper:
    """Full-H energy projection with fixed h, salt, passives and momentum.

    Solve H_c(T_provisional + alpha_c)=H_c(before) using analytic derivatives
    and a bracket wholly inside the caloric domain. Uniform alpha is a declared
    within-column allocation policy. Only the remap's defect is corrected, never
    accumulated timestep error. Reject nonmonotone/infeasible/entropy-losing roots.
    """
    def __init__(self, *, method='plm'):
        self.remapper=EnthalpyRemapper(method=method)

    def remap_with_report(self,model,state,target_h,energy):
        provisional,report=self.remapper.remap_with_report(model,state,target_h,energy)
        before=validate(model,state,energy)
        after=provisional.physical_state(energy.reference)
        old,_=column_terms(energy,before)
        mass=energy.mass_area(state);wet=mass[:,0]>0
        theta=after.temperature();reference=energy.reference
        eps=torch.finfo(theta.dtype).eps
        scale=sum(v.abs() for v in old.values()).clamp_min(1)
        tolerance=256*eps*scale
        if provisional is state:
            zero=state.h.sum(-1)*0
            return state,HamiltonianRemapReport(**report.__dict__,column_energy_residual=zero,
                column_entropy_change=zero,temperature_increment=zero,provisional_energy_change=report.energy_change)
        def evaluate(alpha):
            physical=after.with_tensors(after.u_e, after.h, after.h*(theta+alpha[:,None]), after.salt_h, after.tracer_h)
            terms,derivative=column_terms(energy,physical)
            # Subtract each component before adding; avoid cancellation against
            # one total background energy. Same Hamiltonian, no alternative EOS.
            residual=sum(terms[k]-old[k] for k in old)
            return residual,derivative,physical
        margin=1024*eps
        lower=(-2.5/reference.to_90-theta).amax(-1)+margin
        upper=(40./reference.to_90-theta).amin(-1)-margin
        # Land is unchanged and has no thermal degree of freedom to solve.
        lower=torch.where(wet,lower,torch.zeros_like(lower));upper=torch.where(wet,upper,torch.zeros_like(upper))
        flo,_,_=evaluate(lower);fhi,_,_=evaluate(upper)
        if ((flo>tolerance)|(fhi < -tolerance)).any():
            raise ValueError('column Hamiltonian target outside thermal-domain bracket')
        alpha=torch.zeros_like(lower)
        for _ in range(10):
            residual,derivative,_=evaluate(alpha)
            if ((derivative<=0)&wet).any():
                raise ValueError('nonpositive column thermal energy derivative')
            lower=torch.where(residual<0,alpha,lower);upper=torch.where(residual>0,alpha,upper)
            proposal=alpha-residual/torch.where(wet,derivative,torch.ones_like(derivative))
            alpha=torch.where((proposal>=lower)&(proposal<=upper),proposal,.5*(lower+upper))
        residual,_,physical=evaluate(alpha)
        eta=reference.evaluate(physical.salinity(),physical.temperature(),entropy_derivatives=True)['entropy']
        end=replace(provisional,entropy_h=torch.where(wet[:,None],provisional.h*eta,state.entropy_h))
        final=validate(model,end,energy)
        final_terms,_=column_terms(energy,final)
        residual=sum(final_terms[k]-old[k] for k in old)
        if (residual.abs()>tolerance).any():
            raise FloatingPointError('column Hamiltonian solve failed roundoff-scaled residual check')
        entropy_change=(mass*(end.entropy_h-state.entropy_h)).sum(-1)
        entropy_scale=(mass*state.entropy_h.abs()).sum(-1).clamp_min(1)
        if (entropy_change < -512*eps*entropy_scale).any():
            raise ValueError('energy-closing remap decreases column entropy beyond roundoff')
        old_h0=reference.evaluate(before.salinity(),before.temperature())['h0']
        new_h0=reference.evaluate(final.salinity(),final.temperature())['h0']
        residuals=dict(report.content_residuals)
        residuals['entropy_h']=(end.entropy_h-state.entropy_h).sum(-1)
        terms0,terms1=energy.terms(state),energy.terms(end)
        result=HamiltonianRemapReport(residuals,report.momentum_residual,
            {k:terms1[k]-terms0[k] for k in terms0},(final.temp_h-before.temp_h).sum(-1),
            report.changed_columns,entropy_change.sum(),(end.h*new_h0-state.h*old_h0).sum(-1),
            residual,entropy_change,alpha,report.energy_change)
        return end,result
