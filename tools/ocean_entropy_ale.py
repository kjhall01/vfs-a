"""Explicit ALE composition for the research entropy/salt ocean.

Conserves entropy content, salt, passives and edge momentum. This is a grid
projection, not an energy-conserving physical mixing closure. Nonlinear energy
and temperature-content changes are reported, never compensated or clipped.
"""
from dataclasses import dataclass, replace
import torch
from vfs.ocean.ale import remap_contents, ReferenceGrid, _geometry
from ocean_entropy_candidate import validate


@dataclass(frozen=True)
class EntropyRemapReport:
    content_residuals: dict
    momentum_residual: torch.Tensor
    energy_change: dict
    temperature_content_change: torch.Tensor
    changed_columns: int


class EntropyRemapper:
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
        fields = ['entropy_h', 'salt_h']
        if torch.equal(target_h, state.h):
            end = state
        else:
            contents = [getattr(state,k)[...,None] for k in fields]
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
            end = replace(state, u_e=u, h=target_h, entropy_h=mapped[...,0],
                          salt_h=mapped[...,1], tracer_h=None if state.tracer_h is None else mapped[...,2:])
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
        report = EntropyRemapReport(residuals, momentum_error,
            {k:after[k]-before[k] for k in before},
            (after_physical.temp_h-physical.temp_h).sum(-1), int(changed.sum()))
        return end, report


class EntropyALEStepper:
    """Caller-owned solver and grid policy; remap only after valid full SSP step.

    Return horizontal work ledgers and the separate projection ledger. No
    hidden schedule, heat reservoir, state repair or automatic timestep retry.
    """
    def __init__(self, solver, target=None, remapper=None):
        self.solver = solver
        self.target = ReferenceGrid() if target is None else target
        self.remapper = EntropyRemapper() if remapper is None else remapper

    def step_with_report(self, model, state, dt, energy):
        moved, stages = self.solver.step(model, state, dt, energy)
        end, report = self.remapper.remap_with_report(model, moved, self.target(moved), energy)
        return end, stages, report
