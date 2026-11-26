"""Instantaneous Hamiltonian backscatter with local vertex power funding."""

from dataclasses import dataclass
from vfs.ocean.stratified.tke_model import TKEStepReport
import math
import torch
from vfs.ocean.helpers import nonzero_divisor
from vfs.ocean.temperature.eddy_energy import EddyState


@dataclass(frozen=True)
class BackscatterReport:
    kinetic_power: torch.Tensor  # W
    reservoir_power: torch.Tensor  # W
    local_kinetic_power: torch.Tensor  # W/cell/layer
    local_power_residual: torch.Tensor  # W/cell/layer
    power_allowance: torch.Tensor  # W/cell/layer
    vertex_scale: torch.Tensor  # [vertex, layer], no global minimum
    biharmonic_kinetic_power: torch.Tensor  # W, nonpositive globally
    biharmonic_reservoir_power: torch.Tensor  # W, nonnegative
    local_energy_redistribution: torch.Tensor  # W/cell/layer, sums to zero


@dataclass(frozen=True)
class BackscatterStepReport(TKEStepReport):
    backscatter: tuple = ()  # reports at the three SSPRK3 RHS evaluations


@dataclass(frozen=True)
class EddyBackscatter:
    """Funded negative viscosity and an optional budgeted biharmonic sink.

    Positive viscosity [m²/s] is its magnitude. Local available power is
    fraction*E/timescale [W]. There is no timestep in this constitutive law.
    """

    viscosity: float
    timescale: float
    fraction: float
    biharmonic_viscosity: float = 0.0  # m^4/s; zero preserves the original path

    def __post_init__(self):
        if (
            not all(
                math.isfinite(x)
                for x in (
                    self.viscosity,
                    self.timescale,
                    self.fraction,
                    self.biharmonic_viscosity,
                )
            )
            or self.biharmonic_viscosity < 0
            or self.viscosity < 0
            or self.timescale <= 0
            or not 0 <= self.fraction <= 1
        ):
            raise ValueError(
                "finite viscosities >= 0, timescale > 0 and 0 <= fraction <= 1 required"
            )

    def tendency(self, model, state, *, transported_eddy_h=None, dt=None):
        """Return (du/dt, d(h e)/dt, power report); never modify the state."""
        from vfs.ocean.temperature.roquet import RoquetOcean

        if not isinstance(model, RoquetOcean) or not isinstance(state, EddyState):
            raise TypeError("backscatter requires RoquetOcean and EddyState")
        model.check(state)
        s, H = state.ocean, model.hamiltonian
        mesh, base = s.mesh, H.base.mechanical
        mass = H.mass_area(s).expand_as(s.h)
        allowance = mass * state.eddy_h * (self.fraction / self.timescale)
        if transported_eddy_h is not None:
            if dt is None or not math.isfinite(dt) or dt <= 0:
                raise ValueError("post-transport backscatter funding requires positive finite dt")
            # Advection and backscatter spend the same inventory in an Euler
            # stage. Tighten the local power bound only where transport matters.
            # Leave summation room when source and transport tendencies combine.
            free_power = mass * transported_eddy_h / dt
            margin = 32 * torch.finfo(s.h.dtype).eps * mass * state.eddy_h / dt
            allowance = torch.where(free_power <= allowance,
                                    (free_power - margin).clamp_min(0.), allowance)
        domain = model.ocean.domain
        # Land holds a finite dummy column. Its thickness must not contribute
        # to the vertex viscosity seen by open faces along a partial-cell coast.
        thickness = domain.apply_cell(s.h) if s.column.active_layers is not None else s.h
        he = mesh.cell_to_edge(thickness)
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
            / nonzero_divisor(he)[:, None, :]
        )
        if domain is not None:
            force_parts = force_parts * domain.layer_edge_mask(u)[:, None, :]
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
        force = mesh.curl_adjoint(stress * scale) / nonzero_divisor(he)
        if domain is not None:
            force = domain.apply_layer_edge(force)
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
        # L is positive and self-adjoint in the instantaneous kinetic metric:
        # L v = mask C*(h_v C(mask v))/h_e. Therefore -nu4 L(Lu)
        # removes nu4 <Lu,Lu>_M from resolved KE even with variable thickness.
        # Mask between BOTH applications: coast projection is part of L.
        sink_power = local.new_zeros(())
        gain = torch.zeros_like(local)
        redistribution = torch.zeros_like(local)
        if self.biharmonic_viscosity > 0:

            def rotational_laplacian(v):
                if domain is not None:
                    v = domain.apply_layer_edge(v)
                out = mesh.curl_adjoint(hv * mesh.curl(v)) / nonzero_divisor(he)
                return out if domain is None else domain.apply_layer_edge(out)

            lap_u = rotational_laplacian(u)
            sink = -self.biharmonic_viscosity * rotational_laplacian(lap_u)
            sink_local = H.kinetic_power(state, sink)
            # Positive quadratic dissipation is deposited locally via I^T.
            # Signed u*F also contains conservative energy transport. Retain
            # that redistribution explicitly instead of debiting empty cells.
            gain = base.interpolation_transpose(
                mesh, self.biharmonic_viscosity * metric * lap_u.square()
            )
            redistribution = sink_local + gain
            force = force + sink
            local = local + sink_local
            reservoir_rhs = reservoir_rhs + gain / safe_mass
            reservoir_power = mass * reservoir_rhs
            sink_power = sink_local.sum()
            if not all(torch.isfinite(x).all() for x in (force, reservoir_rhs, gain)):
                raise FloatingPointError("nonfinite biharmonic tendency")
        return (
            force,
            reservoir_rhs,
            BackscatterReport(
                local.sum(),
                reservoir_power.sum(),
                local,
                local + reservoir_power - redistribution,
                allowance,
                scale,
                sink_power,
                gain.sum(),
                redistribution,
            ),
        )
