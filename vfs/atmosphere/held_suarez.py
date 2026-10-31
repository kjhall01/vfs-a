"""Held–Suarez sources, extracted unchanged from the PR #8 forcing equations.

Use with an explicit sum of the conservative and forcing RHS at every RK stage;
no registry, process framework, hidden operator split or dycore replacement.
"""
import torch
from torch import nn
from vfs.atmosphere.state import State, interface_mass
from ..initial_conditions.equilibrium import (
    HeldSuarezConstants, held_suarez_equilibrium_temperature,
)

__all__ = ["HeldSuarezForcing", "HeldSuarezConstants"]


class HeldSuarezForcing(nn.Module):
    """Prescribed dry thermal relaxation and drag, separate from dynamics.

    No geometry or prognostic state is owned here. Optional Teq (K), kT and kv
    (s⁻¹) are prescribed buffers, with the inherited broadcast/conversion rules.
    Teq/kT broadcast to cell/interfaces; kv must carry cell/full axes for
    edge averaging (a scalar kv is not supported by the original equations).
    The tendency is not energy conserving: drag removes kinetic energy without
    recycling it as heat. It adds no pressure or tracer tendency.
    """

    def __init__(
        self,
        constants: HeldSuarezConstants | None = None,
        *,
        Teq: torch.Tensor | None = None,
        kT: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.constants = constants or HeldSuarezConstants()
        self.register_buffer("Teq", None if Teq is None else torch.as_tensor(Teq))
        self.register_buffer("kT", None if kT is None else torch.as_tensor(kT))
        self.register_buffer("kv", None if kv is None else torch.as_tensor(kv))

    def _cell_lat(self, state: State) -> torch.Tensor:
        mesh = state._require_mesh()
        return torch.asin(torch.clamp(mesh.cell_xyz[:, 2:3], -1.0, 1.0))

    def _optional_tensor(self, value: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def equilibrium_temperature(self, state: State, pressure: torch.Tensor) -> torch.Tensor:
        Teq = self._optional_tensor(self.Teq, pressure)
        if Teq is not None:
            return Teq
        return held_suarez_equilibrium_temperature(state._require_column(), self.constants, self._cell_lat(state), pressure)

    def thermal_relaxation_rate(self, state: State, sigma: torch.Tensor) -> torch.Tensor:
        kT = self._optional_tensor(self.kT, sigma)
        if kT is not None:
            return kT
        c = self.constants
        ramp = ((sigma - float(c.sigma_b)) / (1.0 - float(c.sigma_b))).clamp(0.0, 1.0)
        return float(c.k_a) + (float(c.k_s) - float(c.k_a)) * ramp * torch.cos(self._cell_lat(state)).pow(4)

    def drag_rate(self, state: State, sigma: torch.Tensor) -> torch.Tensor:
        kv = self._optional_tensor(self.kv, sigma)
        if kv is not None:
            return kv
        c = self.constants
        ramp = ((sigma - float(c.sigma_b)) / (1.0 - float(c.sigma_b))).clamp(0.0, 1.0)
        return float(c.k_f) * ramp

    def tendency(self, state: State) -> State:
        """Return wind m/s² (edge/full), theta-content Pa K/s (cell/interface).

        Relax CP theta toward Teq/Exner using pressure-mass weights. Apply
        full-level Rayleigh rate averaged to edges to damp normal wind. Rates
        ramp below sigma_b; latitude dependence strengthens low-level thermal
        relaxation near the equator. Existing pressure safeguards are unchanged.
        """
        theta = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)

        p_i = state.interface_pressure()
        sigma_i = (p_i / state.ps.clamp_min(1.0)).clamp_min(1.0e-12)
        theta_eq = self.equilibrium_temperature(state, p_i) / (
            state.exner_interface() / state._require_column().c_p
        ).clamp_min(1.0e-12)
        dmtheta = dm * self.thermal_relaxation_rate(state, sigma_i) * (theta_eq - theta)

        p_full = state.full_pressure()
        sigma_full = (p_full / state.ps.clamp_min(1.0)).clamp_min(1.0e-12)
        du = -state.cell_to_edge(self.drag_rate(state, sigma_full)) * state.u_e
        tracer = None if state.tracer_m is None else torch.zeros_like(state.tracer_m)
        return State(
            du,
            dmtheta,
            torch.zeros_like(state.ps),
            tracer,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def forward(self, state: State) -> State:
        return self.tendency(state)

    def tendency_terms(self, state: State) -> dict[str, State]:
        """Separate thermal relaxation and drag for instantaneous budget work.

        Contract either result with dynamics.boundary_energy_rate_closed(state,
        term) for watts, separately from the conservative RHS contribution.
        These are instantaneous rates, not an exact finite-step energy ledger.
        """
        rhs = self.tendency(state)
        return {
            "thermal_relaxation": State(
                torch.zeros_like(rhs.u_e), rhs.mtheta_i, rhs.ps, rhs.tracer_m,
                mesh=state.mesh, column=state.column, attrs=state._attrs_copy(),
            ),
            "rayleigh_drag": State(
                rhs.u_e, torch.zeros_like(rhs.mtheta_i), rhs.ps, rhs.tracer_m,
                mesh=state.mesh, column=state.column, attrs=state._attrs_copy(),
            ),
        }
