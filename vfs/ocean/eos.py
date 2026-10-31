"""Equation-of-state helpers for VFS-O."""

from __future__ import annotations

import torch

from .constants import OCEAN_ALPHA_T, OCEAN_BETA_S, OCEAN_RHO0, OCEAN_S0, OCEAN_T0


def linear_seawater_density(
    temperature: torch.Tensor,
    salinity: torch.Tensor,
    *,
    rho0: float = OCEAN_RHO0,
    alpha: float = OCEAN_ALPHA_T,
    beta: float = OCEAN_BETA_S,
    T0: float = OCEAN_T0,
    S0: float = OCEAN_S0,
) -> torch.Tensor:
    """Return linear Boussinesq seawater density in kg m-3."""

    return float(rho0) * (1.0 - float(alpha) * (temperature - float(T0)) + float(beta) * (salinity - float(S0)))


def jmd95_bulk_modulus(salinity: torch.Tensor, temperature: torch.Tensor, pressure_dbar: torch.Tensor) -> torch.Tensor:
    """Return Jackett-McDougall-style secant bulk modulus.

    Inputs are practical salinity, potential temperature in degree C, and
    pressure in dbar. The polynomial internally uses pressure in bar, matching
    the common MITgcm JMD95 convention.
    """

    s = salinity
    t = temperature
    p = 0.1 * pressure_dbar
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * torch.sqrt(s.clamp_min(0.0))
    p2 = p * p
    bulk = 1.965933e04 + 1.444304e02 * t - 1.706103e00 * t2 + 9.648704e-03 * t3 - 4.190253e-05 * t4
    bulk = bulk + s * (5.284855e01 - 3.101089e-01 * t + 6.283263e-03 * t2 - 5.084188e-05 * t3)
    bulk = bulk + s3o2 * (3.886640e-01 + 9.085835e-03 * t - 4.619924e-04 * t2)
    bulk = bulk + p * (3.186519e00 + 2.212276e-02 * t - 2.984642e-04 * t2 + 1.956415e-06 * t3)
    bulk = bulk + p * s * (6.704388e-03 - 1.847318e-04 * t + 2.059331e-07 * t2)
    bulk = bulk + p * s3o2 * 1.480266e-04
    bulk = bulk + p2 * (2.102898e-04 - 1.202016e-05 * t + 1.394680e-07 * t2)
    bulk = bulk + p2 * s * (-2.040237e-06 + 6.128773e-08 * t + 6.207323e-10 * t2)
    return bulk


def jmd95_seawater_density(
    salinity: torch.Tensor,
    temperature: torch.Tensor,
    pressure_dbar: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable JMD95 in-situ seawater density in kg m-3."""

    s = salinity
    t = temperature
    p = pressure_dbar
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * torch.sqrt(s.clamp_min(0.0))
    rho = 999.842594 + 6.793952e-02 * t - 9.095290e-03 * t2 + 1.001685e-04 * t3 - 1.120083e-06 * t4 + 6.536332e-09 * t4 * t
    rho = rho + s * (8.244930e-01 - 4.089900e-03 * t + 7.643800e-05 * t2 - 8.246700e-07 * t3 + 5.387500e-09 * t4)
    rho = rho + s3o2 * (-5.724660e-03 + 1.022700e-04 * t - 1.654600e-06 * t2)
    rho = rho + 4.831400e-04 * s * s
    pressure_bar = 0.1 * p
    return rho / (1.0 - pressure_bar / jmd95_bulk_modulus(s, t, p))
