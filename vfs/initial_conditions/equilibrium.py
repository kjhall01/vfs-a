"""Prescribed Held--Suarez equilibrium shared by initialization and thermal forcing."""
from dataclasses import dataclass
import torch
from vfs.geometry.column import CharneyPhillipsColumn

@dataclass(frozen=True)
class HeldSuarezConstants:
    """Standard dry Held-Suarez forcing constants."""

    p0: float = 1.0e5
    T0: float = 315.0
    T_min: float = 200.0
    delta_T_y: float = 60.0
    delta_theta_z: float = 10.0
    sigma_b: float = 0.7
    k_a: float = 1.0 / (40.0 * 86400.0)
    k_s: float = 1.0 / (4.0 * 86400.0)
    k_f: float = 1.0 / 86400.0

    @property
    def T_surface_equator(self) -> float:
        return self.T0

    @property
    def k_a_per_day(self) -> float:
        return self.k_a * 86400.0

    @property
    def k_s_per_day(self) -> float:
        return self.k_s * 86400.0

    @property
    def k_f_per_day(self) -> float:
        return self.k_f * 86400.0


def _lat_for_pressure(lat: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    lat_aligned = lat
    while lat_aligned.ndim < pressure.ndim:
        lat_aligned = lat_aligned[..., None]
    return lat_aligned


def held_suarez_equilibrium_temperature(
    column: CharneyPhillipsColumn,
    constants: HeldSuarezConstants,
    lat: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    """Held-Suarez radiative-equilibrium temperature at pressure points."""

    lat = _lat_for_pressure(lat, pressure)
    sigma = (pressure / float(constants.p0)).clamp_min(1.0e-12)
    raw = (
        float(constants.T0)
        - float(constants.delta_T_y) * torch.sin(lat).square()
        - float(constants.delta_theta_z) * torch.log(sigma) * torch.cos(lat).square()
    ) * sigma.pow(column.kappa)
    floor = torch.as_tensor(constants.T_min, dtype=raw.dtype, device=raw.device)
    return torch.maximum(raw, floor)


