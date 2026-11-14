"""Instantaneous Hamiltonian backscatter with local vertex power funding."""

from dataclasses import dataclass
from vfs.ocean.stratified.tke_model import TKEStepReport
import math
import torch
from eddy_energy import EddyState


@dataclass(frozen=True)
class BackscatterReport:
    kinetic_power: torch.Tensor  # W
    reservoir_power: torch.Tensor  # W
    local_kinetic_power: torch.Tensor  # W/cell/layer
    local_power_residual: torch.Tensor  # W/cell/layer
    power_allowance: torch.Tensor  # W/cell/layer
    vertex_scale: torch.Tensor  # [vertex, layer], no global minimum


@dataclass(frozen=True)
class BackscatterStepReport(TKEStepReport):
    backscatter: tuple = ()  # reports at the three SSPRK3 RHS evaluations


@dataclass(frozen=True)
class EddyBackscatter:
    """Negative rotational viscosity, paired with a local reservoir tendency.

    Positive viscosity [m²/s] is its magnitude. Local available power is
    fraction*E/timescale [W]. There is no timestep in this constitutive law.
    """

    viscosity: float
    timescale: float
    fraction: float

    def __post_init__(self):
        if (
            not all(
                math.isfinite(x)
                for x in (self.viscosity, self.timescale, self.fraction)
            )
            or self.viscosity < 0
            or self.timescale <= 0
            or not 0 <= self.fraction <= 1
        ):
            raise ValueError(
                "finite viscosity >= 0, timescale > 0 and 0 <= fraction <= 1 required"
            )

    def tendency(self, model, state):
        """Return (du/dt, d(h e)/dt, power report); never modify the state."""
        from roquet import RoquetOcean

        if not isinstance(model, RoquetOcean) or not isinstance(state, EddyState):
            raise TypeError("backscatter requires RoquetOcean and EddyState")
        model.check(state)
        s, H = state.ocean, model.hamiltonian
        mesh, base = s.mesh, H.base.mechanical
        mass = H.mass_area(s).expand_as(s.h)
        allowance = mass * state.eddy_h * (self.fraction / self.timescale)
        he = mesh.cell_to_edge(s.h)
        hv = he[mesh.edgesOnVertex].mean(dim=1)
        u = base.velocity(s)
        stress = self.viscosity * hv * mesh.curl(u)
        # Resolve the adjoint into its two endpoint contributions. This is
        # the same incidence/sign convention as mesh.curl_adjoint.
        vertices = mesh.verticesOnEdge
        endpoint_edges = mesh.edgesOnVertex[vertices]
        signs = (
            mesh.edgeSignOnVertex[vertices]
            * (
                endpoint_edges
                == torch.arange(mesh.nEdges, device=u.device)[:, None, None]
            )
        ).sum(-1)
        force_parts = (
            signs[..., None]
            * stress[vertices]
            / mesh.dv_edge[:, None, None]
            / he[:, None, :]
        )
        domain = model.ocean.domain
        if domain is not None:
            force_parts = force_parts * domain.active_edge[:, None, None]
        metric = base.metric(s)
        # Sum positive incidence powers BEFORE cancellations. Scaling every
        # contributing vertex by <= a cell's ratio bounds its positive debit.
        parts = metric[:, None, :] * u[:, None, :] * force_parts
        demand = base.interpolation_transpose(mesh, parts.clamp_min(0).sum(1))
        safe = torch.where(demand > 0, demand, torch.ones_like(demand))
        ratio = torch.where(
            demand > 0, (allowance / safe).clamp(max=1), torch.ones_like(demand)
        )
        # Cell -> incident edge -> incident vertex, strictly a local stencil.
        edge_ratio = ratio[mesh.cellsOnEdge].amin(1)
        scale = edge_ratio[mesh.edgesOnVertex].amin(1)
        scale = torch.where(
            scale < 1, scale * (1 - 64 * torch.finfo(u.dtype).eps), scale
        )
        force = mesh.curl_adjoint(stress * scale) / he
        if domain is not None:
            force = domain.apply_edge(force)
        local = H.kinetic_power(state, force)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        reservoir_rhs = -local / safe_mass
        reservoir_power = mass * reservoir_rhs
        if not all(
            torch.isfinite(x).all() for x in (force, reservoir_rhs, allowance, demand)
        ):
            raise FloatingPointError("nonfinite backscatter tendency")
        tolerance = 256 * torch.finfo(u.dtype).eps * (allowance + local.abs())
        if (local > allowance + tolerance).any():
            raise FloatingPointError("local backscatter power exceeds allowance")
        return (
            force,
            reservoir_rhs,
            BackscatterReport(
                local.sum(),
                reservoir_power.sum(),
                local,
                local + reservoir_power,
                allowance,
                scale,
            ),
        )
