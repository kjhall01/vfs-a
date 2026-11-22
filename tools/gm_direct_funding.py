"""Experimental direct local funding for the fixed-thickness donor-parcel GM step.

Scalar roots are calculated on detached local patches. The accepted transport
and its finite energy budget always pass through the original VFS operator.
This is a forward prototype; derivatives of the selected scales are omitted.
"""
import math

import numpy as np
import torch

from gm_parcel_limiter import incident_gm_interfaces, limited_gm_fluxes


def _patches(nodes):
    """Connect elementary circulations that share any of their four cell-layers."""
    parents = list(range(len(nodes)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(nodes):
        for node in row:
            other = owners.setdefault(node, index)
            parents[root(index)] = root(other)
    groups = {}
    for index in range(len(nodes)):
        groups.setdefault(root(index), []).append(index)
    return groups.values()


def _feasible_quadratics(constant, linear, quadratic):
    """Intersection with [0,1] for concave scalar quadratics >= 0."""
    lo, hi = 0., 1.
    for a, b, c in zip(constant, linear, quadratic):
        if a >= 0 and a + b + c >= 0 and c <= 0:
            continue
        if c > 0:  # An alternative EOS could have a nonconcave feasible set.
            return None
        if abs(c) <= np.finfo(float).eps * max(abs(a), abs(b)):
            if b > 0:
                lo = max(lo, -a / b)
            elif b < 0:
                hi = min(hi, -a / b)
            elif a < 0:
                return None
        else:
            discriminant = b*b - 4*c*a
            if discriminant < 0:
                return None
            q = -.5 * (b + math.copysign(math.sqrt(discriminant), b))
            r1, r2 = (q/c, a/q) if q else (0., 0.)
            lo, hi = max(lo, min(r1, r2)), min(hi, max(r1, r2))
        if lo > hi:
            return None
    return lo, hi


def solve_patch_scale(q, contribution, left, right, volume, temp, salt, eddy,
                      coeff, temp_full, reserve_full, dt, config):
    """Find the least removal along this patch's common scale, in specific units.

    Affected carriers are q - removal*contribution. At a carrier sign change
    switch to the other donor. Within each sign region all transports are affine
    in removal, so only the local Roquet CT-squared term adds curvature.
    """
    a, b, c, d = coeff
    thermal_mu = 2*a*temp_full + b
    face_coeff = .5 * (coeff[:, left] + coeff[:, right])
    af, bf, cf, df = face_coeff
    endpoints = np.r_[left, right]
    rate = dt * np.r_[-1/volume[left], 1/volume[right]]
    n = len(volume)

    def per_donor(donor):
        T, S = temp[donor], salt[donor]
        potential = (af*T + bf)*T + cf*S + df
        # cp*CT and advected TKE cancel between the transported Hamiltonian
        # and donor energy flux. The advected eddy inventory remains available.
        e_face = potential + eddy[donor]
        dT, dS = T - config.CT0, S - config.SA0
        thermal = np.tile(dT, 2)
        energy = (np.tile(e_face, 2) - thermal_mu[endpoints]*thermal
                  - c[endpoints]*np.tile(dS, 2))
        return np.stack((thermal, energy)) * rate

    from_left, from_right = per_donor(left), per_donor(right)
    incoming = np.minimum(q, 0)
    outgoing = np.maximum(q, 0)
    # The buffer is relative to available pre-step inventory and is applied
    # only to the failing layers. It is never added to the model inventory.
    margin = np.where(reserve_full < 0, np.minimum(1.e-7, .02*eddy), 0.)

    def evaluate(removal):
        trial = q - removal * contribution
        delta = (from_left * np.tile(np.maximum(trial, 0) - outgoing, 2)
                 + from_right * np.tile(np.minimum(trial, 0) - incoming, 2))
        dT, linear_credit = (np.bincount(endpoints, weights=x, minlength=n) for x in delta)
        return dT, reserve_full - margin + linear_credit - a*dT*dT

    switches = q[contribution != 0] / contribution[contribution != 0]
    breaks = np.r_[0., np.unique(switches[(switches > 0) & (switches < 1)]), 1.]
    T0, R0 = evaluate(0.)
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        T1, R1 = evaluate(upper)
        curvature = -a*(T1-T0)**2
        interval = _feasible_quadratics(R0, R1-R0-curvature, curvature)
        if interval is not None:
            lo, hi = interval
            # Stay on the feasible side of the computed boundary.
            x = min(hi, lo + 1.e-8)
            return 1. - (lower + x*(upper-lower))
        T0, R0 = T1, R1
    return None


def _cpu(tensor):
    return tensor.detach().cpu().numpy()


def _direct_factors(model, state, transported, reserve, metrics, geom, mask, dt):
    s = state.ocean
    nz = s.h.shape[1]
    edges, levels = mask.nonzero(as_tuple=True)
    if edges.numel() == 0:
        return None
    i, j, ell = metrics[3], metrics[4], metrics[6]
    left, right = i[edges], j[edges]
    upper_l, upper_r = left*nz + levels, right*nz + levels
    corners = torch.stack((upper_l, upper_l+1, upper_r, upper_r+1), dim=1)
    # Each psi interface contributes a signed volume transport around one
    # four-face loop (two adjacent cells and two adjacent layers).
    nh = geom[2].numel()
    faces = torch.stack((edges*nz+levels, edges*nz+levels+1,
                         nh+left*(nz-1)+levels, nh+right*(nz-1)+levels), dim=1)
    amount = ell[edges, 0] * geom[1][edges, levels+1]
    factors = torch.ones_like(geom[1][:, 1:-1])
    coefficients = model.ocean.energy.base.coefficients(s)
    patch_scales = []
    for indices in _patches(_cpu(corners)):
        patch = torch.as_tensor(indices, device=edges.device)
        face, offsets = torch.unique(faces[patch].flatten(), return_inverse=True)
        value = amount[patch]
        dq = torch.zeros_like(geom[6][face]).index_add(
            0, offsets, torch.stack((value, -value, -value, value), dim=1).flatten())
        nodes, local = torch.unique(torch.cat((geom[4][face], geom[5][face])),
                                    return_inverse=True)
        il, ir = local.chunk(2)
        def take(field):
            return field.flatten()[nodes]
        h = take(s.h)
        inputs = (geom[6][face], dq, il, ir, s.mesh.cell_area[nodes//nz]*h,
                  take(s.temp_h)/h, take(s.salt_h)/h, take(state.eddy_h)/h,
                  torch.stack([take(p) for p in coefficients]),
                  take(transported.ocean.temp_h)/h, take(reserve))
        scale = solve_patch_scale(*map(_cpu, inputs), dt, model.config)
        if scale is None or not math.isfinite(scale):
            return None
        factors[edges[patch], levels[patch]] = scale
        patch_scales.append(scale)
    return factors, patch_scales


class _Unfunded(FloatingPointError):
    def __init__(self, reserve, transported):
        super().__init__("GM requires unavailable local eddy energy")
        self.reserve, self.transported = reserve, transported


def make_direct_gm_class(GMRedi, divergence, *, events=None):
    """Return an optional GM operator; events receives one dict per intervention."""
    class DirectGM(GMRedi):
        _funding_active = False
        _factor = None

        def _metrics(self, model, state):
            if not self._funding_active:
                return super()._metrics(model, state)
            if self._cached_metrics is None:
                self._cached_metrics = super()._metrics(model, state)
            return self._cached_metrics

        def _geometry(self, model, state, *, part="all", metrics=None, gm_diffusivity=None):
            if part != "gm" or self._factor is None:
                g = super()._geometry(model, state, part=part, metrics=metrics,
                                      gm_diffusivity=gm_diffusivity)
                if part == "gm" and self._funding_active:
                    self._proposal = g
                return g
            g = self._proposal
            limited = limited_gm_fluxes(g[1], self._factor, metrics, state.ocean.h, divergence)
            return (g[0], *limited, g[-2], g[-1])

        def _settle(self, model, before, transported, energy_inflow, name):
            try:
                return super()._settle(model, before, transported, energy_inflow, name)
            except FloatingPointError as err:
                if name != "GM" or "requires unavailable local eddy energy" not in str(err):
                    raise
                H = model.hamiltonian
                work = H.finite_change(before, transported) - energy_inflow
                mass = H.mass_area(before.ocean)
                reserve = (transported.eddy_h - work / torch.where(
                    mass > 0, mass, torch.ones_like(mass))) / before.ocean.h
                raise _Unfunded(reserve, transported) from None

        def apply(self, model, state, dt):
            self.last_event = None
            self._funding_active = True
            self._cached_metrics = self._proposal = self._factor = None
            try:
                return self._funded_apply(model, state, dt)
            finally:
                self._funding_active = False
                self._cached_metrics = self._proposal = self._factor = None

        def _funded_apply(self, model, state, dt):
            attempts = 1
            try:
                return super().apply(model, state, dt)
            except _Unfunded as err:
                reserve, transported = err.reserve, err.transported

            metrics, g = self._cached_metrics, self._proposal
            def incident(balance):
                return incident_gm_interfaces(balance < 0, metrics[3], metrics[4])
            first = incident(reserve)
            solution = _direct_factors(model, state, transported, reserve, metrics, g, first, dt)
            scales = []
            if solution is not None:
                self._factor, scales = solution
                attempts += 1
                try:
                    end, report = super().apply(model, state, dt)
                except FloatingPointError as err:
                    # Verify *all* native constraints. A donor switch, roundoff
                    # or neighboring layers can invalidate a patch prediction.
                    reason = str(err)
                else:
                    self._record(scales, attempts, None)
                    return end, report
            else:
                reason = "No jointly funded scale on one or more patches."

            # Replay the original halving policy from the original proposal,
            # including expansion onto newly unfunded layers. Reuse geometry
            # because the state at the start of every attempt is identical.
            self._factor = torch.ones_like(g[1][:, 1:-1])
            active = first
            for _ in range(20):
                self._factor = torch.where(active, .5*self._factor, self._factor)
                attempts += 1
                try:
                    end, report = super().apply(model, state, dt)
                except _Unfunded as err:
                    active = incident(err.reserve)
                else:
                    self._record(scales, attempts, reason)
                    return end, report
            raise FloatingPointError("Direct GM and 20 fallback funding passes exhausted.")

        def _record(self, scales, attempts, fallback):
            row = dict(gm_budget_attempts=attempts, patch_count=len(scales), scales=scales,
                       fallback=fallback is not None, fallback_reason=fallback)
            if events is not None:
                f = self._factor
                q = (self._cached_metrics[6] * self._proposal[1][:, 1:-1]).abs()
                row.update(locally_scaled_interfaces=int((f < 1).sum()),
                           abs_psi_cut_m3_s=float((q*(1-f)).sum()),
                           abs_psi_scaled_m3_s=float(q[f < 1].sum()),
                           abs_psi_total_m3_s=float(q.sum()))
                events.append(row)
            self.last_event = row

    return DirectGM
