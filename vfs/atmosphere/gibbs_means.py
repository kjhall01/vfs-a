"""Analytic mixture face means and fixed-temperature composition derivatives.

Positive dry-air and vapor partial densities are required. Condensates may
vanish. These routines retain checkpoint-54 expressions and branch conditions.
"""

import torch


def logmean(a, b):
    """Positive logarithmic mean, with a cancellation-safe equal-state limit."""
    if bool((a <= 0).any() or (b <= 0).any()):
        raise ValueError("logarithmic mean requires positive inputs")
    avg = 0.5 * (a + b)
    x = (b - a) / (b + a)
    x2 = x * x
    # x/atanh(x) = 1-x^2/3-4*x^4/45-44*x^6/945+O(x^8).
    series = 1 - x2 / 3 - 4 * x2 * x2 / 45 - 44 * x2 * x2 * x2 / 945
    close = x2 < 1e-6
    den = torch.where(close, torch.ones_like(a), torch.log(b) - torch.log(a))
    exact = (b - a) / den
    return torch.where(close, avg * series, exact)


def partial_chemical(t, T, p, q):
    """Species chemical potentials (last axis dry/vapor/liquid/ice), J/kg.

    T/p have the same leading shape, K/Pa; q adds a length-four fraction axis.
    partial holds gas partial pressures (Pa), sa partial entropies J/(kg K),
    ha partial enthalpies J/kg. Condensates have zero gas constants.
    """
    cp = T.new_tensor([t.cp_d, t.cp_v, t.cp_l, t.cp_i])
    R = T.new_tensor([t.R_d, t.R_v, 0.0, 0.0])
    href = T.new_tensor([0.0, t.L_v_ref, 0.0, -t.L_f_ref])
    pref = T.new_tensor([t.p_d_ref, t.p_v_ref, 1.0, 1.0])
    partial = p[..., None] * q * R / t.gas_constant(q)[..., None]
    safe = torch.where(R > 0, partial, torch.ones_like(partial))
    sa = (
        cp * torch.log(T[..., None] / t.T_ref)
        + href / t.T_ref
        - R * torch.log(safe / pref)
    )
    ha = cp * (T[..., None] - t.T_ref) + href
    return ha - T[..., None] * sa


def gibbs_face(t, Ti, pi, qi, Tj, pj, qj):
    """Return s*, q*, alpha* such that Delta mu_d+s*Delta T+
    q_water*dot(Delta(mu_water-mu_d)) = alpha*Delta p.

    Called on horizontal edges × CP interfaces: endpoint T (E,I) K,
    p (E,I) Pa, q (E,I,4) kg/kg. Returns specific entropy (E,I) J/(kg K),
    species fractions (E,I,4) and specific volume alpha (E,I) m³/kg.
    rhoi/rhoj and di/dj/dstar are mixture/partial densities, kg/m³.
    eta here is mean mu/T, J/(kg K), not the composition derivative below.

    Gas densities use logarithmic means; condensates use arithmetic means.
    The latter is free in this identity because their gas constants are zero.
    """
    Ri = t.gas_constant(qi)
    Rj = t.gas_constant(qj)
    rhoi = pi / (Ri * Ti)
    rhoj = pj / (Rj * Tj)
    di = rhoi[..., None] * qi
    dj = rhoj[..., None] * qj
    dstar = torch.cat(
        [logmean(di[..., :2], dj[..., :2]), 0.5 * (di[..., 2:] + dj[..., 2:])], -1
    )
    density = dstar.sum(-1)
    q = dstar / density[..., None]
    mu_i = partial_chemical(t, Ti, pi, qi)
    mu_j = partial_chemical(t, Tj, pj, qj)
    eta = 0.5 * (mu_i / Ti[..., None] + mu_j / Tj[..., None])
    cp = t.heat_capacity_p(q)
    R = t.gas_constant(q)
    cv = cp - R
    # cp/R/cv are J/(kg K); k is the reference enthalpy offset (J/kg).
    # The resulting entropy trace satisfies the face Gibbs identity above.
    k = t.enthalpy_reference(q) - cp * t.T_ref
    Tavg = 0.5 * (Ti + Tj)
    s = (
        Tavg * cv / logmean(Ti, Tj)
        + Tavg * k / (Ti * Tj)
        - (q * eta).sum(-1)
        + 0.5 * (rhoi * Ri + rhoj * Rj) / density
    )
    return s, q, 1 / density


def composition_entropy_derivatives(t, T, p, q):
    """ds/d(q_v,q_l,q_i) at fixed T,p, with q_d=1-sum(q_water).
    Analytic ideal-mixture partial entropies; requires positive dry/vapor.

    T (C,I) K, p (C,I) Pa, q (C,I,4) kg/kg -> (C,I,3) J/(kg K).
    Each water-species change replaces dry air; subtracting sd enforces that
    composition convention. evaluate averages these derivatives onto graph faces.
    """
    R = t.gas_constant(q)
    pd = p * q[..., 0] * t.R_d / R
    pv = p * q[..., 1] * t.R_v / R
    logT = torch.log(T / t.T_ref)
    sd = t.cp_d * logT - t.R_d * torch.log(pd / t.p_d_ref)
    sv = t.cp_v * logT + t.L_v_ref / t.T_ref - t.R_v * torch.log(pv / t.p_v_ref)
    sl = t.cp_l * logT
    si = t.cp_i * logT - t.L_f_ref / t.T_ref
    return torch.stack([sv - sd, sl - sd, si - sd], -1)
