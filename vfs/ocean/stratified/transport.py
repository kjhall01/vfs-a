"""Local-bound flux targets with an additional mechanical-work budget.

Research composition of the frozen Gibbs/entropy ocean. Rotation and the
Hamiltonian are unchanged. Local extrema are targets for active quantities:
when their correction would require excessive positive mechanical work, blend
back toward the globally admissible parent flux. Passives retain local bounds.
"""
from dataclasses import dataclass, replace
import torch
from .equations import (EntropyState, EntropyEnergy, validate,
    transport_for, contraction)
from . import equations as parent
from vfs.ocean.transport import divergence


def local_target(model, state, dt, eta_high, salt_high, composition_entropy,
                 baseline=None):
    """FCT target using all incident faces and this Euler stage's local bounds.

    Entropy and salt have independent target budgets. Their eventual changes
    are paired through the unchanged energy functional, not post-step clipping.
    Salt/passive antidiffusion never exceeds the parent's per-incidence budget.
    """
    if baseline is None:
        baseline = parent.bounded_transport(model, state, dt, eta_high,
                                             salt_high, composition_entropy)
    low = transport_for(model).advance(state.transport_view(), dt)
    anti = baseline.high_integrated_flux-low.donor_integrated_flux
    amount = low.low_amounts.reshape(state.h.numel(), -1)
    # Sum incoming/outgoing magnitudes separately, without cancellation.
    positive = torch.zeros_like(amount).index_add(0, low.j, anti.clamp_min(0))
    positive = positive.index_add(0, low.i, (-anti).clamp_min(0))
    negative = torch.zeros_like(amount).index_add(0, low.i, anti.clamp_min(0))
    negative = negative.index_add(0, low.j, (-anti).clamp_min(0))
    def ratio(room, demand):
        safe = torch.where(demand > 0, demand, torch.ones_like(demand))
        return torch.where(demand > 0, room.clamp_min(0)/safe,
                           torch.ones_like(demand)).clamp(max=1)
    upper = ratio(low.upper-amount, positive)
    lower = ratio(amount-low.lower, negative)
    beta = torch.where(anti >= 0, torch.minimum(lower[low.i], upper[low.j]),
                       torch.minimum(upper[low.i], lower[low.j]))
    beta = torch.cat([beta[:, :1], torch.minimum(beta[:, 1:], baseline.beta[:, 1:])], -1)
    beta = torch.where(beta < 1, beta*(1-64*torch.finfo(beta.dtype).eps), beta)
    fields = [state.entropy_h[..., None], state.salt_h[..., None]]
    if state.tracer_h is not None:
        fields.append(state.tracer_h)
    q = (torch.cat(fields, -1)/state.h[..., None]).reshape_as(amount)
    active = slice(None) if model.domain is None else model.domain.active_edge
    high = torch.cat([eta_high[active].reshape(-1, 1), salt_high[active].reshape(-1, 1),
                      .5*(q[low.i, 2:]+q[low.j, 2:])], -1)
    donor = torch.where((low.carrier >= 0)[:, None], q[low.i], q[low.j])
    change = (1-beta)*(donor-high)
    flux = low.donor_integrated_flux+beta*anti
    amounts = amount+divergence(low.i, low.j, beta*anti, amount.shape[0])
    target = replace(baseline, amounts=amounts.reshape_as(baseline.amounts),
                     integrated_flux=flux, trace_change=change, beta=beta)
    return target, low, high


def bounded_transport(model, state, dt, eta_high, salt_high, composition_entropy):
    """Isolated prescribed-carrier target; no active work constraint or heating."""
    return local_target(model, state, dt, eta_high, salt_high, composition_entropy)[0]


@dataclass(frozen=True)
class FrontWorkLedger(parent.WorkLedger):
    target_beta: torch.Tensor
    work_fraction: torch.Tensor
    target_face_work: torch.Tensor
    kinetic_budget: torch.Tensor


def evaluate(model, state, dt, energy):
    original_rhs, original = parent.evaluate(model, state, dt, energy)
    physical = state.physical_state(energy.reference)
    eta, salt, face = parent.high_traces(model, physical, energy)
    target, low, high = local_target(model, state, dt, eta, salt,
                                     face['composition_entropy'], original.transport)
    area, nodes = state.mesh.cell_area[:, None], state.h.numel()
    g = energy.gradients(state)
    covector = torch.stack([g[k]/area for k in ('entropy_h', 'salt_h')], -1).reshape(nodes, 2)
    jump = covector[low.j]-covector[low.i]
    delta = target.trace_change[:, :2]-original.transport.trace_change[:, :2]
    added_work = low.carrier*(jump*delta).sum(-1)
    active = slice(None) if model.domain is None else model.domain.active_edge
    kinetic = (.5*g['u_e']*state.u_e)[active].flatten()
    # Allow at most the greater of parent positive work and K_e/dt. The
    # parent's force is already continuous at a single-edge zero crossing.
    # K_e=O(u_e^2), so the added correction cannot restore an O(1) force jump.
    cap = torch.maximum(original.face_work.clamp_min(0), kinetic/dt)
    room = (cap-original.face_work).clamp_min(0)
    safe = torch.where(added_work > 0, added_work, torch.ones_like(added_work))
    fraction = torch.where(added_work > 0, (room/safe).clamp(max=1), torch.ones_like(safe))
    active_change = original.transport.trace_change[:, :2]+fraction[:, None]*delta
    change = torch.cat([active_change, target.trace_change[:, 2:]], -1)
    beta_salt = original.transport.beta[:, 1]+fraction*(target.beta[:, 1]-original.transport.beta[:, 1])
    beta = torch.cat([target.beta[:, :1], beta_salt[:, None], target.beta[:, 2:]], -1)
    anti = original.transport.high_integrated_flux-low.donor_integrated_flux
    amounts = low.low_amounts.reshape(nodes, -1)+divergence(low.i, low.j, beta*anti, nodes)
    flux = (dt*low.carrier)[:, None]*(high+change)
    initial_entropy = (area*state.entropy_h).flatten()
    entropy_amount = initial_entropy+divergence(low.i, low.j, flux[:, 0], nodes)
    amounts = torch.cat([entropy_amount[:, None], amounts[:, 1:]], -1)
    # Store the exact flux corresponding to authoritative salt/passive amounts.
    flux = torch.cat([flux[:, :1], (low.donor_integrated_flux+beta*anti)[:, 1:]], -1)
    ledger = replace(target, amounts=amounts.reshape_as(target.amounts),
                     integrated_flux=flux, trace_change=change, beta=beta)
    per_carrier = (jump*change[:, :2]).sum(-1)
    work = low.carrier*per_carrier
    heat = (-work).clamp_min(0)
    coefficient = g['entropy_h'].flatten()
    source = torch.zeros_like(coefficient).index_add(0, low.i, .5*heat/coefficient[low.i])
    source = source.index_add(0, low.j, .5*heat/coefficient[low.j]).reshape_as(state.h)
    mechanical = torch.where(work > 0, per_carrier, torch.zeros_like(work))
    correction = torch.zeros_like(state.u_e)
    correction[active] = -mechanical.reshape_as(correction[active])/(model.dynamics.rho0*model.mesh.dc_edge[active, None])
    rates = divergence(low.i, low.j, flux/dt, nodes).reshape_as(ledger.amounts)/area[..., None]
    rhs = replace(original_rhs, u_e=original_rhs.u_e-original.force_correction+correction,
                  entropy_h=rates[..., 0]+source, salt_h=rates[..., 1],
                  tracer_h=None if state.tracer_h is None else rates[..., 2:])
    return rhs, FrontWorkLedger(ledger, work, heat, source, correction, original.gibbs_identity,
        target.beta, fraction, original.face_work+added_work, cap)


def euler(model, state, dt, energy):
    rhs, work = evaluate(model, state, dt, energy)
    area, ledger = state.mesh.cell_area[:, None], work.transport
    amounts = ledger.amounts/area[..., None]
    end = replace(state, u_e=state.u_e+dt*rhs.u_e, h=ledger.volume/area,
                  entropy_h=amounts[..., 0]+dt*work.entropy_source,
                  salt_h=amounts[..., 1], tracer_h=None if state.tracer_h is None else amounts[..., 2:])
    validate(model, end, energy)
    return end, work


def step(model, state, dt, energy):
    first, l1 = euler(model, state, dt, energy)
    second, l2 = euler(model, first, dt, energy)
    third, l3 = euler(model, .75*state+.25*second, dt, energy)
    end = (1/3)*state+(2/3)*third
    validate(model, end, energy)
    return end, (l1, l2, l3)
