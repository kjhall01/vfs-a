"""Finite species-amount budgets on the CP carrier graph.

Extracted from checkpoint 54 without algebraic changes. Amounts here are
pressure-weighted contents times cell area; divide by gravity for kg.
The limiter bounds dry air and all three water species together over every
incident horizontal/vertical face. It is not a local-extremum limiter.
"""

import math
import torch
from .conservative import v
from .moist_cp import interface_pressure_mass


def graph(d, s):
    """Connect CP nodes and return pressure-weighted carrier amounts/fluxes.

    For C cells, E edges, L layers and I=L+1 interfaces: mass (C,I), Pa m²;
    i,j (H,) integer node IDs; f (H,), Pa m²/s; H=E*I+C*L. Positive flux
    goes i -> j. Horizontal faces (E,I) precede vertical faces (C,L).
    Divide amounts/fluxes by gravity for kg and kg/s. This staggering matches
    entropy/water on cell interfaces, with velocity still on edge layers.
    """
    a = d._adapter(s)
    ctx = d.kernel._transport_primitives(a)
    m = s.mesh
    mass = v(ctx["dm"]) * m.cell_area[:, None]
    n = mass.shape[1]
    c0, c1 = m.cellsOnEdge.T
    k = torch.arange(n)
    i = (c0[:, None] * n + k).flatten()
    j = (c1[:, None] * n + k).flatten()
    layer = v(ctx["mass_flux"]) * m.dv_edge[:, None]
    # Layer-edge transport -> CP interfaces; half weights at top/bottom.
    h = torch.cat(
        [0.5 * layer[:, :1], 0.5 * (layer[:, :-1] + layer[:, 1:]), 0.5 * layer[:, -1:]],
        -1,
    )
    ids = torch.arange(m.nCells * n).reshape(m.nCells, n)
    # Down-column links join adjacent interfaces. omega already includes the
    # hybrid-coordinate continuity response; multiply by horizontal cell area.
    i = torch.cat([i, ids[:, :-1].flatten()])
    j = torch.cat([j, ids[:, 1:].flatten()])
    f = torch.cat([h.flatten(), (v(ctx["omega"]) * m.cell_area[:, None]).flatten()])
    return mass, i, j, f


def divergence(i, j, flux, nodes):
    """Net incoming amount/rate per node, with the same units as flux.

    Subtract each oriented face at i and add it at j: global sums cancel.
    Trailing axes (e.g. species) are carried unchanged; no area division here.
    """
    shape = (nodes, *flux.shape[1:])
    return (
        torch.zeros(shape, dtype=flux.dtype, device=flux.device)
        .index_add(0, i, -flux)
        .index_add(0, j, flux)
    )


def bounded_transfer(m0, Q0, i, j, G, J, floors=None, *, safety=0.9, target_mass=None):
    """Return species amounts and one common limited flux for all species.

    m0/target_mass (N,) and Q0 (N,4) are Pa m²; G (H,) and J (H,4)
    are time-integrated carrier/species transfers, also Pa m². Species order
    is dry/vapor/liquid/ice. floors (4,) are dimensionless lower bounds.
    Result (N,4) is the Euler ledger, limited (H,4) its integrated transfer;
    beta (H,) is a common dimensionless face factor for all four species.

    Requires m0>0, m1=m0+B G>0 and sum_a J_a=G. A donor remap along the
    straight mass path supplies the low-order state. Only that cheap graph
    remap may subcycle; no wave solve or full RHS is subcycled. Both endpoints
    use the SAME integrated carrier as the high-order stage.

    FCT bounds all species simultaneously using total negative demand over
    every incident face. Positivity + the species sum also bounds q_a<=1.
    This is not a local-extremum or entropy-monotonicity limiter.
    target_mass optionally supplies the actual stage mass from an inexact
    wave solve; bounds account for its residual, without inventing a flux.
    Returned dry amounts remain ledger amounts; physical dry is target_mass
    minus the returned water sum. Reject if the low-order budget is invalid.
    """
    if not 0 < safety < 1:
        raise ValueError("safety must lie between zero and one")
    if not all(bool(torch.isfinite(x).all()) for x in (m0, Q0, G, J)):
        raise ValueError("nonfinite transfer")
    if Q0.ndim != 2 or J.shape != (G.numel(), Q0.shape[1]):
        raise ValueError("incompatible species/flux shapes")
    nodes = m0.numel()
    eps = torch.finfo(Q0.dtype).eps
    floors = (
        Q0.new_zeros(Q0.shape[1])
        if floors is None
        else torch.as_tensor(floors, dtype=Q0.dtype, device=Q0.device)
    )
    if bool((floors < 0).any()) or bool(floors.sum() >= 1):
        raise ValueError("invalid fraction floors")
    if not torch.allclose(Q0.sum(-1), m0, rtol=128 * eps, atol=0):
        raise ValueError("initial species must sum to carrier")
    scale = G.abs().max().clamp_min(torch.finfo(G.dtype).tiny)
    if not torch.allclose(J.sum(-1), G, rtol=128 * eps, atol=128 * eps * scale):
        raise ValueError("species fluxes must sum to carrier flux")
    dm = divergence(i, j, G, nodes)
    m1 = m0 + dm
    if bool((m0 <= 0).any() or (m1 <= 0).any()):
        raise ValueError("nonpositive endpoint carrier mass")
    tol = 128 * eps * Q0.abs().amax(0)
    if bool((Q0 - m0[:, None] * floors < -tol).any()):
        raise ValueError("initial species below floor")
    target = m1 if target_mass is None else target_mass
    if (
        target.shape != m1.shape
        or not bool(torch.isfinite(target).all())
        or bool((target <= 0).any())
    ):
        raise ValueError("invalid target carrier mass")
    # Account explicitly for the tiny carrier residual of an inexact wave
    # solve. Q_d is the ledger dry amount; physical dry = Q_d+target-m1.
    lower = target[:, None] * floors
    lower = lower.clone()
    lower[:, 0] += m1 - target
    high = Q0 + divergence(i, j, J, nodes)
    # Exact recovery, including fluxes, whenever the proposed amounts are safe.
    if bool((high >= lower).all()):
        return (
            high,
            J,
            dict(beta=torch.ones_like(G), substeps=0, limited_faces=0, mass=m1),
        )
    out = divergence(i, j, torch.zeros_like(G), nodes)
    out.index_add_(0, i, G.clamp_min(0))
    out.index_add_(0, j, (-G).clamp_min(0))
    steps = max(1, math.ceil(float((out / torch.minimum(m0, m1)).max()) / safety))
    # Donor reference on the straight carrier path. low is a running amount
    # ledger; lowflux is cumulative transfer; lowface is the mean donor fraction.
    low = Q0
    lowflux = torch.zeros_like(J)
    lowface = torch.zeros_like(J)
    for k in range(steps):
        mass = m0 + (k / steps) * dm
        q = low / mass[:, None]
        donor = torch.where((G >= 0)[:, None], q[i], q[j])
        lowface = lowface + donor / steps
        transfer = (G / steps)[:, None] * donor
        low = low + divergence(i, j, transfer, nodes)
        lowflux = lowflux + transfer
    anti = J - lowflux
    # anti (H,4), Pa m², is the correction to donor transfer. demand (N,4)
    # sums all outgoing correction amounts, including horizontal/vertical faces.
    demand = (
        torch.zeros_like(Q0)
        .index_add(0, i, anti.clamp_min(0))
        .index_add(0, j, (-anti).clamp_min(0))
    )
    # Only a roundoff margin is admitted; never hide a failed low-order budget.
    if bool((low - lower < -tol).any()):
        raise RuntimeError("donor reference violates bounds")
    allowance = (low - lower).clamp_min(0)
    # Available amount above the lower bound / total demand gives node ratios.
    # A single minimum over species and the losing endpoints preserves the
    # sum-to-carrier constraint while limiting every incident-face demand.
    ratios = torch.where(
        demand > 0, allowance / demand.clamp_min(torch.finfo(Q0.dtype).tiny), 1
    ).clamp(max=1)
    beta = torch.minimum(
        torch.where(anti > 0, ratios[i], 1), torch.where(anti < 0, ratios[j], 1)
    ).amin(-1)
    beta = torch.where(beta < 1, beta * (1 - 64 * eps), beta)
    limited = lowflux + beta[:, None] * anti
    # Keep this donor-plus-correction ledger expression: recomputing from a
    # diagnosed tendency or reconstructed fractions changes cancellation.
    result = low + divergence(i, j, beta[:, None] * anti, nodes)
    return (
        result,
        limited,
        dict(
            beta=beta,
            substeps=steps,
            limited_faces=int((beta < 1).sum()),
            mass=m1,
            low_face=lowface,
        ),
    )


def limited_traces(m, q, sh, qh, eta, i, j, F, dt, *, target=None, floors=None):
    """Limit the ACTUAL high flux; pair its change without dividing by F.

    m (N,) is Pa m²; q (N,4) and qh (H,4) are species fractions. sh (H,)
    is Gibbs specific entropy, J/(kg K). eta (H,3), J/(kg K), is the face
    composition derivative for vapor/liquid/ice. F (H,) is Pa m²/s; dt is s.
    Return entropy/fraction traces plus the authoritative amounts/flux ledger.

    eta is the arithmetic face mean of ds/dq at fixed T,p. It attaches the
    existing composition-entropy convention to dq relative to Gibbs, not
    relative to an unrelated arithmetic high flux. This convention closes
    work with the transpose; it is not an entropy-production theorem.
    """
    if dt <= 0:
        raise ValueError("positive budget timestep required")
    G = dt * F
    J = G[:, None] * qh
    # Make implicit dry flux exactly complementary to the three water fluxes.
    J = torch.cat([(G - J[:, 1:].sum(-1))[:, None], J[:, 1:]], -1)
    Q, flux, info = bounded_transfer(
        m, m[:, None] * q, i, j, G, J, floors, target_mass=target
    )
    if info["limited_faces"]:
        # Recover the trace CHANGE from beta/donor fractions, not flux/F:
        # zero carrier faces remain well-defined. Q/flux stay authoritative.
        dq = (1 - info["beta"][:, None]) * (info["low_face"] - qh)
    else:
        dq = torch.zeros_like(qh)
    ds = (eta * dq[:, 1:]).sum(-1)
    # dq (H,4) is dimensionless; ds (H,) is J/(kg K). They describe the
    # limiter-minus-Gibbs change used by evaluate's matched work correction.
    info.update(
        amounts=Q,
        integrated_flux=flux,
        high_integrated_flux=J,
        delta_q=dq,
        delta_s=ds,
        high_q=qh,
        trace_flux_error=(flux - G[:, None] * (qh + dq)).abs().max(),
    )
    return sh + ds, qh + dq, info
