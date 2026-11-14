"""Temperature-coordinate ALE: existing enthalpy remap and adaptive projection.

Research translation of the supported algorithm. Same minimum-change step,
energy/entropy constraints and tolerances; no entropy-to-T reconstructions.
"""
from dataclasses import dataclass
import torch
from vfs.ocean.ale import remap_contents, _geometry
from vfs.ocean.stratified.remap import minimum_change_step, AdaptiveThermalReport
from vfs.ocean.stratified.enthalpy_remap import EnthalpyRemapReport
from vfs.ocean.stratified.energy_columns import column_terms
from fast_thermo import Caloric
from model import TemperatureOcean


class TemperatureEnergy:
    def __init__(self,ocean):
        self.owner=ocean;self.base=ocean.energy.base;self.reference=ocean.reference
        self.mass_area=ocean.energy.mass_area
    def terms(self,state):
        result=self.base.terms(state)
        ref=self.reference.evaluate(state.salinity(),state.temperature())
        result['reference_thermal']=(self.mass_area(state)*state.h*ref['h0']).sum()
        result['total']=result['total']+result['reference_thermal']
        return result


def entropy_content(energy,state):
    return state.h*energy.reference.evaluate(state.salinity(),state.temperature())['entropy']


def validate(model,state,energy):
    TemperatureOcean(energy.owner).fields(state)
    return state


class TemperatureEnthalpyRemapper:
    def __init__(self, *, method='plm'):
        if method not in ('pcm', 'plm'):
            raise ValueError('method must be pcm or plm')
        self.method = method

    def remap_with_report(self, model, state, target_h, energy):
        physical = validate(model, state, energy)
        if target_h.shape != state.h.shape:
            raise ValueError('target must preserve layer layout')
        domain = model.domain
        if domain is not None:
            target_h = torch.where(domain.wet_cell[:, None], target_h, state.h)
        state.column.validate_thickness(target_h, min_thickness=model.dynamics.min_thickness)
        _geometry(state.h, target_h)
        changed = (target_h != state.h).any(-1)
        fields = ['temp_h', 'salt_h']
        reference = energy.reference
        enthalpy_h = state.h*reference.evaluate(physical.salinity(), physical.temperature())['h0']
        if torch.equal(target_h, state.h):
            end = state
        else:
            contents = [enthalpy_h[...,None], state.salt_h[...,None]]
            if state.tracer_h is not None:
                contents.append(state.tracer_h)
            mapped = remap_contents(state.h, torch.cat(contents,-1), target_h, method=self.method)
            # Retain untouched columns exactly, including land placeholders.
            original = torch.cat(contents,-1)
            mapped = torch.where(changed[:,None,None], mapped, original)
            old_edge = model.mesh.cell_to_edge(state.h)
            new_edge = model.mesh.cell_to_edge(target_h)
            momentum = remap_contents(old_edge, old_edge*state.u_e, new_edge, method=self.method)
            active = (new_edge != old_edge).any(-1)
            if domain is not None:
                active = active & domain.active_edge
            u = torch.where(active[:,None], momentum/new_edge, state.u_e)
            theta = Caloric(reference,mapped[...,1]/target_h).inverse(mapped[...,0]/target_h,physical.temperature(),'h0')
            temp_h=torch.where(changed[:,None],target_h*theta,state.temp_h)
            end=state.with_tensors(u,target_h,temp_h,mapped[...,1],
                                  None if state.tracer_h is None else mapped[...,2:])
        after_physical = validate(model, end, energy)
        residuals = {k:(getattr(end,k)-getattr(state,k)).sum(-1) for k in ['h']+fields}
        if state.tracer_h is not None:
            residuals['tracer_h'] = (end.tracer_h-state.tracer_h).sum(-2)
        old_edge, new_edge = model.mesh.cell_to_edge(state.h), model.mesh.cell_to_edge(end.h)
        factor = model.dynamics.rho0*model.mesh.dc_edge*model.mesh.dv_edge
        momentum_error = factor*(new_edge*end.u_e-old_edge*state.u_e).sum(-1)
        if domain is not None:
            momentum_error = momentum_error*domain.active_edge
        before, after = energy.terms(state), energy.terms(end)
        final_enthalpy = end.h*reference.evaluate(after_physical.salinity(), after_physical.temperature())['h0']
        enthalpy_residual = (final_enthalpy-enthalpy_h).sum(-1)
        report = EnthalpyRemapReport(residuals, momentum_error,
            {k:after[k]-before[k] for k in before},
            (after_physical.temp_h-physical.temp_h).sum(-1), int(changed.sum()),
            (energy.mass_area(state)*(entropy_content(energy,end)-entropy_content(energy,state))).sum(), enthalpy_residual)
        return end, report


class TemperatureRemapper:
    def __init__(self, *, method='plm'):
        self.remapper=TemperatureEnthalpyRemapper(method=method)

    def remap_with_report(self,model,state,target_h,energy):
        provisional,report=self.remapper.remap_with_report(model,state,target_h,energy)
        before=validate(model,state,energy);after=validate(model,provisional,energy)
        old,_=column_terms(energy,before);mass=energy.mass_area(state);wet=mass[:,0]>0
        base=energy.base;reference=energy.reference;theta0=after.temperature();eps=torch.finfo(theta0.dtype).eps
        depth=after.h.sum(-1);weights=after.h/depth[:,None]
        column_mass=mass[:,0]*depth;safe_mass=torch.where(wet,column_mass,torch.ones_like(column_mass))
        energy_scale=sum(v.abs() for v in old.values()).clamp_min(1)
        entropy_scale=(mass*entropy_content(energy,state).abs()).sum(-1).clamp_min(1)
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
            entropy_change=(mass*(after.h*ref['entropy']-entropy_content(energy,state))).sum(-1)
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
        end=provisional.with_tensors(provisional.u_e,provisional.h,torch.where(wet[:,None],after.h*theta,state.temp_h),provisional.salt_h,provisional.tracer_h)
        final=validate(model,end,energy);final_terms,_=column_terms(energy,final)
        residual=sum(final_terms[k]-old[k] for k in old)
        entropy_change=(mass*(entropy_content(energy,end)-entropy_content(energy,state))).sum(-1)
        if (residual.abs()>energy_tolerance).any():raise FloatingPointError('adaptive thermal projection failed column energy residual check')
        if (entropy_change < -entropy_tolerance).any():raise ValueError('adaptive thermal projection decreases column entropy')
        old_h0=reference.evaluate(before.salinity(),before.temperature())['h0']
        new_h0=reference.evaluate(final.salinity(),final.temperature())['h0']
        residuals=dict(report.content_residuals);residuals['entropy_h']=(entropy_content(energy,end)-entropy_content(energy,state)).sum(-1)
        terms0,terms1=energy.terms(state),energy.terms(end)
        result=AdaptiveThermalReport(residuals,report.momentum_residual,
            {k:terms1[k]-terms0[k] for k in terms0},(final.temp_h-before.temp_h).sum(-1),
            report.changed_columns,entropy_change.sum(),(end.h*new_h0-state.h*old_h0).sum(-1),
            residual,entropy_change,theta-theta0,report.energy_change,ever_active,iteration+1)
        return end,result
