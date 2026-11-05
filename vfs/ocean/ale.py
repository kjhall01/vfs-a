"""Explicit vertical Lagrangian regrid/remap for the active JMD95 ocean.

Contents are integrated over old/new layer overlaps. Velocity is remapped on
the actual edge thickness I(h), conserving its column-integrated momentum.
The remap preserves inventories, not the nonlinear Hamiltonian. Its energy
change is returned separately; no temperature correction or heat sink is added.
All vertical coordinates below are distances downward from the common surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .jmd95 import JMD95Dynamics, checked_ssprk3


def _edges(h):
    edges = torch.cat((torch.zeros_like(h[..., :1]), h.cumsum(-1)), -1)
    if not (edges.diff(dim=-1) > 0).all().item():
        raise ValueError('layer interfaces are not distinguishable in this dtype')
    return edges


def _geometry(old_h, new_h):
    if old_h.ndim < 1 or new_h.ndim != old_h.ndim or old_h.shape[:-1] != new_h.shape[:-1]:
        raise ValueError('old/new thickness must have matching column dimensions')
    for h in (old_h, new_h):
        if h.shape[-1] < 1 or not h.is_floating_point():
            raise ValueError('thickness must have at least one floating-point layer')
        if h.dtype != old_h.dtype or h.device != old_h.device:
            raise ValueError('old/new thickness must share dtype/device')
        if not torch.isfinite(h).all().item() or not (h > 0).all().item():
            raise ValueError('thickness must be finite and strictly positive')
    old, new = _edges(old_h), _edges(new_h)
    tolerance = 32 * torch.finfo(old_h.dtype).eps * old[..., -1:].abs()
    if ((new[..., -1:] - old[..., -1:]).abs() > tolerance).any().item():
        raise ValueError('remapping must preserve each column total thickness')
    # A common endpoint prevents roundoff-sized uncovered slivers. This changes
    # no supplied thickness; the permitted mismatch is at summation roundoff.
    new = torch.cat((new[..., :-1], old[..., -1:]), -1)
    if not (new.diff(dim=-1) > 0).all().item():
        raise ValueError('new interfaces collapse at the common endpoint')
    return old, new


def _slopes(h, q, method):
    """Bounded, mean-preserving slopes on unequal layers; flat end layers.

    q has shape (..., layer, field). Boundary slopes are zero to avoid new
    extrema without prescribing unknown boundary values. Interior minmod
    slopes receive an endpoint bound based on the actual layer thickness.
    """
    if method == 'pcm' or h.shape[-1] < 3:
        return torch.zeros_like(q)
    distance = .5 * (h[..., :-1] + h[..., 1:])
    secant = q.diff(dim=-2) / distance[..., None]
    left, right = secant[..., :-1, :], secant[..., 1:, :]
    raw = torch.where(left * right > 0, torch.sign(left) * torch.minimum(left.abs(), right.abs()), 0.)
    center = q[..., 1:-1, :]
    low = torch.minimum(center, torch.minimum(q[..., :-2, :], q[..., 2:, :]))
    high = torch.maximum(center, torch.maximum(q[..., :-2, :], q[..., 2:, :]))
    bound = 2 * torch.minimum(center - low, high - center) / h[..., 1:-1, None]
    interior = torch.sign(raw) * torch.minimum(raw.abs(), bound)
    return torch.cat((torch.zeros_like(q[..., :1, :]), interior, torch.zeros_like(q[..., :1, :])), -2)


def remap_contents(old_h, content, new_h, *, method='plm'):
    """Remap h*q conservatively between columns with the same physical extent.

    h: (..., layers); content: (..., layers) or (..., layers, fields).
    The low-level function permits different old/new layer counts. OceanState
    remapping keeps its model-owned layer count. PCM is a reference method;
    PLM uses bounded piecewise-linear reconstruction. No concentration clipping.

    Searchsorted locates the first/last intersected old layers; prefix sums
    integrate complete intervening layers. This avoids an O(layers**2) overlap
    matrix. Single-layer intersections are integrated directly to avoid
    subtracting two large cumulative inventories for a thin destination layer.
    Tensor gradients are retained away from overlap/limiter switching points.
    """
    if method not in ('pcm', 'plm'):
        raise ValueError("method must be 'pcm' or 'plm'")
    old, new = _geometry(old_h, new_h)
    scalar = content.ndim == old_h.ndim
    values = content[..., None] if scalar else content
    if values.ndim != old_h.ndim + 1 or values.shape[:-1] != old_h.shape:
        raise ValueError('content must match thickness with an optional final field axis')
    if values.dtype != old_h.dtype or values.device != old_h.device:
        raise ValueError('content and thickness must share dtype/device')
    if not torch.isfinite(values).all().item():
        raise ValueError('content must be finite')
    q = values / old_h[..., None]
    slope = _slopes(old_h, q, method)
    a, b = new[..., :-1].contiguous(), new[..., 1:].contiguous()
    old_right = old[..., 1:].contiguous()
    first = torch.searchsorted(old_right, a, right=True).clamp(max=old_h.shape[-1]-1)
    last = torch.searchsorted(old_right, b, right=False).clamp(max=old_h.shape[-1]-1)

    def take(field, index):
        return field.gather(-2, index[..., None].expand(*index.shape, field.shape[-1]))

    start, end = old[..., :-1, None], old[..., 1:, None]
    middle = start + .5 * old_h[..., None]
    a, b = a[..., None], b[..., None]
    q_first, q_last = take(q, first), take(q, last)
    s_first, s_last = take(slope, first), take(slope, last)
    mid_first, mid_last = take(middle, first), take(middle, last)
    end_first, start_last = take(end, first), take(start, last)
    # Integral of q_i + slope_i*(z-mid_i) on each partial intersection.
    first_part = (end_first-a) * (q_first + .5*s_first*((end_first-mid_first)+(a-mid_first)))
    last_part = (b-start_last) * (q_last + .5*s_last*((b-mid_last)+(start_last-mid_last)))
    prefix = torch.cat((torch.zeros_like(values[..., :1, :]), values.cumsum(-2)), -2)
    whole = take(prefix, last) - take(prefix, first+1)
    # Use the supplied small width, not a subtraction of large depths.
    single = new_h[..., None] * (q_first + .5*s_first*((a-mid_first)+(b-mid_first)))
    result = torch.where((first == last)[..., None], single, first_part + whole + last_part)
    return result[..., 0] if scalar else result


class ReferenceGrid:
    """Prescribed positive thickness fractions, stretched with total depth.

    fractions is (layers,) or (cells,layers), sums to one in each column,
    and is borrowed without conversion/detachment. With no explicit fractions,
    use OceanColumn.rest_thickness fractions. For a flat resting bottom this is
    a simple z-star-like target. Varying bottom depth retains the chosen local
    fractions; it does not imply globally level interfaces or partial cells.
    """
    def __init__(self, fractions=None):
        self.fractions = fractions

    def __call__(self, state):
        fractions = self.fractions
        if fractions is None:
            rest = state.column.rest_thickness_like(state.h)
            fractions = rest / rest.sum(-1, keepdim=True)
        if not isinstance(fractions, torch.Tensor):
            raise TypeError('fractions must be a tensor')
        if fractions.shape not in ((state.column.n_layers,), state.h.shape):
            raise ValueError('fractions must have shape (layers,) or (cells,layers)')
        if fractions.dtype != state.h.dtype or fractions.device != state.h.device:
            raise ValueError('fractions must match state dtype/device')
        if not torch.isfinite(fractions).all().item() or not (fractions > 0).all().item():
            raise ValueError('fractions must be finite and positive')
        if (fractions.sum(-1)-1).abs().max().item() > 32*torch.finfo(fractions.dtype).eps:
            raise ValueError('fractions must sum to one in every column')
        return state.h.sum(-1, keepdim=True) * fractions


@dataclass(frozen=True)
class RemapReport:
    """Connected tensor diagnostics, with per-column residuals, not only sums.

    energy_change contains joules for kinetic/free_surface/dynamic_enthalpy/
    total. Content residuals have the stored content units (m times q); momentum
    residuals are edge-column rho0*A_e*sum(h_e*u), in kg m/s. Land residuals are
    zero. Energy changes may have either sign; they are not diagnosed heating.
    """
    content_residuals: dict[str, torch.Tensor]
    momentum_residual: torch.Tensor
    energy_change: dict[str, torch.Tensor]


class VerticalRemapper:
    """Fixed-layer-count conservative remapper for JMD95 Ocean states."""
    def __init__(self, *, method='plm'):
        if method not in ('pcm', 'plm'):
            raise ValueError("method must be 'pcm' or 'plm'")
        self.method = method

    def __call__(self, model, state, target_h):
        result, _ = self._remap(model, state, target_h, report=False)
        return result

    def remap_with_report(self, model, state, target_h):
        return self._remap(model, state, target_h, report=True)

    def _remap(self, model, state, target_h, *, report):
        if not isinstance(model.dynamics, JMD95Dynamics):
            raise TypeError('VerticalRemapper requires JMD95Dynamics')
        model.check_state(state)
        model.dynamics.check_state(state)
        if not isinstance(target_h, torch.Tensor) or target_h.shape != state.h.shape:
            raise ValueError('target thickness must match the state layer layout')
        if target_h.dtype != state.h.dtype or target_h.device != state.h.device:
            raise ValueError('target thickness must match state dtype/device')
        domain = model.domain
        if domain is not None:
            # Keep finite land placeholders and all inactive velocities intact.
            target_h = torch.where(domain.wet_cell[:, None], target_h, state.h)
        state.column.validate_thickness(target_h, min_thickness=model.dynamics.min_thickness)
        fields = [state.temp_h[..., None], state.salt_h[..., None]]
        if state.tracer_h is not None:
            fields.append(state.tracer_h)
        content = remap_contents(state.h, torch.cat(fields, -1), target_h, method=self.method)
        temp_h, salt_h = content[..., 0], content[..., 1]
        tracer_h = None if state.tracer_h is None else content[..., 2:]
        edge_old, edge_new = model.mesh.cell_to_edge(state.h), model.mesh.cell_to_edge(target_h)
        momentum = remap_contents(edge_old, edge_old*state.u_e, edge_new, method=self.method)
        u = momentum / edge_new
        if domain is not None:
            temp_h = torch.where(domain.wet_cell[:, None], temp_h, state.temp_h)
            salt_h = torch.where(domain.wet_cell[:, None], salt_h, state.salt_h)
            if tracer_h is not None:
                tracer_h = torch.where(domain.wet_cell[:, None, None], tracer_h, state.tracer_h)
            u = torch.where(domain.active_edge[:, None], u, state.u_e)
        result = state.with_tensors(u, target_h, temp_h, salt_h, tracer_h)
        model.check_state(result)
        model.dynamics.check_state(result)
        if not report:
            return result, None
        residuals = {name: (getattr(result,name)-getattr(state,name)).sum(-1)
                     for name in ('h', 'temp_h', 'salt_h')}
        if tracer_h is not None:
            residuals['tracer_h'] = (tracer_h-state.tracer_h).sum(-2)
        factor = model.dynamics.rho0 * model.mesh.dc_edge * model.mesh.dv_edge
        momentum_error = factor * (edge_new*u-edge_old*state.u_e).sum(-1)
        if domain is not None:
            momentum_error = momentum_error * domain.active_edge
        before, after = model.dynamics.energy.terms(state), model.dynamics.energy.terms(result)
        energy_change = {name: after[name]-before[name] for name in before}
        return result, RemapReport(residuals, momentum_error, energy_change)


class ALEStepper:
    """Checked SSPRK3 followed by one explicit vertical remap per full step.

    This callable fits Ocean(..., stepper=ALEStepper()). No hidden call counter
    or retained report/graph: use step_with_report for diagnostics. For a less
    frequent remap, call checked_ssprk3 and VerticalRemapper explicitly on the
    caller's schedule. Invalid full Euler stages fail before any remap. The
    combined algorithm's order is not asserted to equal SSPRK3's order.
    """
    def __init__(self, target=None, remapper=None):
        self.target = ReferenceGrid() if target is None else target
        self.remapper = VerticalRemapper() if remapper is None else remapper

    def __call__(self, model, state, dt):
        moved = checked_ssprk3(model, state, dt)
        return self.remapper(model, moved, self.target(moved))

    def step_with_report(self, model, state, dt):
        moved = checked_ssprk3(model, state, dt)
        return self.remapper.remap_with_report(model, moved, self.target(moved))
