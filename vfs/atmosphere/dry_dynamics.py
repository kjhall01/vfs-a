"""Dry hydrostatic primitive-equation dynamics on the VFS atmosphere state."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..core.state import State
from .constants import GRAVITY, OMEGA
from . import dry_energy
from . import dry_momentum
from .dry_strategies import (
    DryThermodynamics,
    EdgeUpwindVerticalMomentumTransport,
    ScaleNormalizedCoordinateWork,
    TerrainAwarePressureGradient,
    UpwindTracerTransport,
)
from . import dry_terms
from . import dry_thermo
from . import dry_tracers
from . import dry_transport


@register_component("dry_hydrostatic")
class DryHydrostaticDynamics(nn.Module):
    """Dry adiabatic hydrostatic dynamics on a State-owned Voronoi C-grid."""

    # The tendency assembly below remains the entry point. Equation helpers
    # are ordinary functions in their owning modules, explicitly bound here.
    # No multiple-inheritance resolution or dynamic method injection is used.
    # dry_transport: explicit ownership, preserving subclass overrides.
    _transport_primitives = dry_transport._transport_primitives
    ps_tendency_terms = dry_transport.ps_tendency_terms
    _layer_thickness = dry_transport._layer_thickness
    _interface_mass = dry_transport._interface_mass
    _layer_divergence = dry_transport._layer_divergence
    _ps_tendency = dry_transport._ps_tendency
    _vertical_mass_flux = dry_transport._vertical_mass_flux
    _omega_full = dry_transport._omega_full

    # dry_momentum: explicit ownership, preserving subclass overrides.
    _trisk_rotation = dry_momentum._trisk_rotation
    _u_vertical = dry_momentum._u_vertical
    _momentum_diffusion_tendency = dry_momentum._momentum_diffusion_tendency
    _momentum_diffusion_term_name = dry_momentum._momentum_diffusion_term_name
    momentum_tendency_terms = dry_momentum.momentum_tendency_terms
    _pgf_components_for_state = dry_momentum._pgf_components_for_state
    _pgf_sigma_correction_for_state = dry_momentum._pgf_sigma_correction_for_state
    _pgf_term_for_state = dry_momentum._pgf_term_for_state
    _u_tendency = dry_momentum._u_tendency

    # dry_thermo: explicit ownership, preserving subclass overrides.
    _cell_interface_energy_weight = dry_thermo._cell_interface_energy_weight
    _cell_full_energy_weight = dry_thermo._cell_full_energy_weight
    _dry_energy_gradient_mtheta_for_coord_work = dry_thermo._dry_energy_gradient_mtheta_for_coord_work
    _dry_energy_gradient_ps_local_terms = dry_thermo._dry_energy_gradient_ps_local_terms
    _dry_energy_gradient_ps_terms = dry_thermo._dry_energy_gradient_ps_terms
    _dry_energy_gradient_ps = dry_thermo._dry_energy_gradient_ps
    _coordinate_work_unscaled_reference_impl = dry_thermo._coordinate_work_unscaled_reference_impl
    _coordinate_work_scale_normalized_impl = dry_thermo._coordinate_work_scale_normalized_impl
    _coordinate_work_scale_normalized_from_gradient = dry_thermo._coordinate_work_scale_normalized_from_gradient
    _mtheta_vertical = dry_thermo._mtheta_vertical
    _energy_gradient_mtheta_for_coord_work = dry_thermo._energy_gradient_mtheta_for_coord_work
    _canonical_mtheta_coordinate_work_unscaled_reference = dry_thermo._canonical_mtheta_coordinate_work_unscaled_reference
    _canonical_mtheta_coordinate_work = dry_thermo._canonical_mtheta_coordinate_work
    _mtheta_tendency = dry_thermo._mtheta_tendency
    _mtheta_advection_tendency = dry_thermo._mtheta_advection_tendency
    mtheta_tendency_terms = dry_thermo.mtheta_tendency_terms

    # dry_tracers: explicit ownership, preserving subclass overrides.
    _tracer_tendency_upwind = dry_tracers._tracer_tendency_upwind
    _tracer_tendency = dry_tracers._tracer_tendency
    tracer_tendency_terms = dry_tracers.tracer_tendency_terms

    # dry_terms: explicit ownership, preserving subclass overrides.
    _zero_state_like = dry_terms._zero_state_like
    _state_with_tendency = dry_terms._state_with_tendency
    _directional_energy_rate = dry_terms._directional_energy_rate
    edge_metric = dry_terms.edge_metric
    pressure_fields = dry_terms.pressure_fields
    thermo_fields = dry_terms.thermo_fields
    wind_fields = dry_terms.wind_fields
    energy_terms = dry_terms.energy_terms
    tracer_terms = dry_terms.tracer_terms
    static_stability = dry_terms.static_stability
    static_instability_penalty = dry_terms.static_instability_penalty
    vertical_structure_terms = dry_terms.vertical_structure_terms
    mass_flux = dry_terms.mass_flux
    rotated_mass_flux = dry_terms.rotated_mass_flux
    rotation_term = dry_terms.rotation_term
    pgf_term = dry_terms.pgf_term
    bernoulli_term = dry_terms.bernoulli_term
    mass_transport_term = dry_terms.mass_transport_term
    mtheta_transport_term = dry_terms.mtheta_transport_term
    tracer_transport_term = dry_terms.tracer_transport_term
    coordinate_work_term = dry_terms.coordinate_work_term
    tendency_terms = dry_terms.tendency_terms
    terms_total = dry_terms.terms_total
    force_terms = dry_terms.force_terms
    force_balance_terms = dry_terms.force_balance_terms
    rotation_work_terms = dry_terms.rotation_work_terms
    _edge_extensive_to_cell_layer = dry_terms._edge_extensive_to_cell_layer
    _pressure_work_column_residual_stats = dry_terms._pressure_work_column_residual_stats
    pressure_work_local_terms = dry_terms.pressure_work_local_terms
    pressure_work_local_attribution_terms = dry_terms.pressure_work_local_attribution_terms
    pressure_work_edge_exchange_terms = dry_terms.pressure_work_edge_exchange_terms
    pressure_work_tangential_stencil_terms = dry_terms.pressure_work_tangential_stencil_terms
    pressure_work_residual_structure_terms = dry_terms.pressure_work_residual_structure_terms
    energy_exchange_terms = dry_terms.energy_exchange_terms
    pressure_work_terms = dry_terms.pressure_work_terms
    cfl_terms = dry_terms.cfl_terms
    tracer_no_feedback_terms = dry_terms.tracer_no_feedback_terms
    divergence = dry_terms.divergence
    vorticity = dry_terms.vorticity
    omega = dry_terms.omega
    tendency_scale_terms = dry_terms.tendency_scale_terms
    adjustment_shock_terms = dry_terms.adjustment_shock_terms
    mesh_health_terms = dry_terms.mesh_health_terms
    operator_identity_terms = dry_terms.operator_identity_terms

    # dry_energy: explicit ownership, preserving subclass overrides.
    total_mass = dry_energy.total_mass
    total_tracer_mass = dry_energy.total_tracer_mass
    total_kinetic_energy = dry_energy.total_kinetic_energy
    kinetic_energy_gradient_u = dry_energy.kinetic_energy_gradient_u
    _kinetic_energy_edge_weight = dry_energy._kinetic_energy_edge_weight
    kinetic_energy_gradient_u_terms = dry_energy.kinetic_energy_gradient_u_terms
    scalar_energy_gradients = dry_energy.scalar_energy_gradients
    scalar_pressure_energy_gradient_terms = dry_energy.scalar_pressure_energy_gradient_terms
    scalar_pressure_energy_gradient_local_terms = dry_energy.scalar_pressure_energy_gradient_local_terms
    scalar_energy_rate_closed = dry_energy.scalar_energy_rate_closed
    scalar_energy_rate_terms_closed = dry_energy.scalar_energy_rate_terms_closed
    scalar_pressure_energy_rate_terms_closed = dry_energy.scalar_pressure_energy_rate_terms_closed
    wind_energy_rate_terms_closed = dry_energy.wind_energy_rate_terms_closed
    energy_rate_closed = dry_energy.energy_rate_closed
    _dry_total_potential_energy = dry_energy._dry_total_potential_energy
    _dry_total_internal_energy = dry_energy._dry_total_internal_energy
    total_potential_energy = dry_energy.total_potential_energy
    total_internal_energy = dry_energy.total_internal_energy
    total_energy = dry_energy.total_energy

    recipe_name = "canonical"

    def __init__(
        self,
        omega: float = OMEGA,
        gravity: float = GRAVITY,
        validate_state: bool = True,
        *,
        tracer_transport=None,
        coordinate_work=None,
        thermodynamics=None,
        pressure_gradient=None,
        momentum_transport=None,
        momentum_diffusion=None,
        pgf_correction_mode: str = "none",
    ) -> None:
        """Create a dry dynamics operator.

        The base dynamics composes concrete strategy objects. Named component
        subclasses own recipe strings and registry identifiers.
        """
        super().__init__()

        self.rotation_rate = float(omega)
        self.gravity = float(gravity)
        self.validate_state = bool(validate_state)
        self.tracer_transport = (
            UpwindTracerTransport() if tracer_transport is None else tracer_transport
        )
        self.coordinate_work = (
            ScaleNormalizedCoordinateWork() if coordinate_work is None else coordinate_work
        )
        self.thermodynamics = DryThermodynamics() if thermodynamics is None else thermodynamics
        base_pressure_gradient = (
            TerrainAwarePressureGradient(self.thermodynamics)
            if pressure_gradient is None
            else pressure_gradient
        )
        if str(pgf_correction_mode) != "none":
            raise ValueError("learned PGF corrections were removed; pgf_correction_mode must be 'none'")
        self.pressure_gradient = base_pressure_gradient
        self.momentum_transport = (
            EdgeUpwindVerticalMomentumTransport()
            if momentum_transport is None
            else momentum_transport
        )
        self.momentum_diffusion = momentum_diffusion
        self._validate_strategy_interfaces()

    @staticmethod
    def _require_strategy_methods(strategy, name: str, methods: tuple[str, ...]) -> None:
        missing = [method for method in methods if not callable(getattr(strategy, method, None))]
        if missing:
            got = type(strategy).__name__
            required = ", ".join(methods)
            absent = ", ".join(missing)
            raise TypeError(
                f"{name} must be a concrete strategy object with methods {required}; "
                f"got {got} missing {absent}"
            )

    def _validate_strategy_interfaces(self) -> None:
        self._require_strategy_methods(self.tracer_transport, "tracer_transport", ("tendency",))
        self._require_strategy_methods(self.coordinate_work, "coordinate_work", ("tendency",))
        self._require_strategy_methods(
            self.thermodynamics,
            "thermodynamics",
            (
                "pgf_components",
                "energy_gradient_mtheta_for_coord_work",
                "thermo_fields",
                "total_potential_energy",
                "total_internal_energy",
            ),
        )
        self._require_strategy_methods(self.pressure_gradient, "pressure_gradient", ("pgf_components",))
        self._require_strategy_methods(self.momentum_transport, "momentum_transport", ("tendency",))
        if self.momentum_diffusion is not None:
            self._require_strategy_methods(self.momentum_diffusion, "momentum_diffusion", ("tendency",))

    def check_state_compatible(self, state: State) -> None:
        """Dry dynamics requires a self-describing State."""
        if state.mesh is None:
            raise ValueError("DryHydrostaticDynamics requires State.mesh")
        if state.column is None:
            raise ValueError("DryHydrostaticDynamics requires State.column")

    # ------------------------------------------------------------------
    # Diagnostic quantities
    # ------------------------------------------------------------------

    def _validate_mass_geometry(
        self,
        state: State,
        dp: torch.Tensor,
        dm: torch.Tensor | None = None,
    ) -> None:
        """Raise when the hybrid column geometry becomes nonphysical."""
        if not self.validate_state:
            return
        ps = state.ps
        p_top = float(state.column.A_interface[0].item())

        if not torch.isfinite(ps).all().item():
            raise FloatingPointError("surface pressure is not finite")
        if ps.min().item() <= p_top:
            raise FloatingPointError(
                f"surface pressure fell below model top pressure: "
                f"ps_min={ps.min().item():.3e}, p_top={p_top:.3e}"
            )

        if not torch.isfinite(dp).all().item():
            raise FloatingPointError("layer thickness is not finite")
        if dp.min().item() <= 0.0:
            raise FloatingPointError(f"nonpositive layer thickness: dp_min={dp.min().item():.3e}")

        if dm is not None:
            if not torch.isfinite(dm).all().item():
                raise FloatingPointError("interface mass is not finite")
            if dm.min().item() <= 0.0:
                raise FloatingPointError(f"nonpositive interface mass: dm_min={dm.min().item():.3e}")


    def tendency(self, state: State) -> State:
        """
        Compute all prognostic tendencies from the current state.

        Computation order
        -----------------
        1. dp, dm       — layer/interface mass (from ps)
        2. layer_div    — horizontal mass-flux divergence per layer
        3. dps_dt       — sum of layer_div (ps budget)
        4. mu           — vertical mass flux at interfaces (from layer_div, dps_dt)
        5. omega        — full-level vertical mass flux (from mu)
        6. du_dt        — horizontal + vertical momentum tendency
        7. dtheta_dt    — horizontal + vertical theta tendency

        Each quantity is computed once and reused where needed.

        Parameters
        ----------
        state : State

        Returns
        -------
        State of tendencies (du_dt, dtheta_dt, dps_dt).
        """
        self.check_state_compatible(state)
        attrs = dict(state.attrs or {})
        attrs.setdefault("gravity", self.gravity)
        u_e = state.u_e

        # --- Shared mass/transport primitives ---
        ctx = self._transport_primitives(state)
        dp = ctx["dp"]                                      # [..., nCells, N]
        theta_i = ctx["theta"]                              # [..., nCells, N+1]
        dps_dt = ctx["dps_dt"]                              # [..., nCells, 1]
        mu = ctx["mu"]                                      # [..., nCells, N+1]
        omega = ctx["omega"]                                # [..., nCells, N]

        # --- Momentum tendency (horizontal + vertical) ---
        du_dt = (
            self._u_tendency(state, u_e, theta_i)
            + self._u_vertical(state, u_e, mu, dp, dp_e=ctx.get("dp_e"))
            + self._momentum_diffusion_tendency(state)
        )                                                  # [..., nEdges, N]

        # --- mtheta tendency: horizontal + vertical + coordinate work ---
        # The coordinate work correction restores energy conservation when
        # using donor-cell vertical scalar transport.
        dmtheta_dt = (
            self._mtheta_advection_tendency(
                state,
                theta_i,
                ctx,
            )
            + self._canonical_mtheta_coordinate_work(state, theta_i, omega, primitives=ctx)
        )                                                  # [..., nCells, N+1]

        tracer_dt = None
        if state.tracer_m is not None:
            q = state.tracers()
            assert q is not None
            tracer_dt = self._tracer_tendency(state, u_e, q, dp, mu)

        return State(
            u_e=du_dt,
            mtheta_i=dmtheta_dt,
            ps=dps_dt,
            tracer_m=tracer_dt,
            mesh=state.mesh,
            column=state.column,
            attrs=attrs,
        )


    def forward(self, state: State) -> State:
        """Alias for tendency() — makes dynamics usable as an nn.Module."""
        return self.tendency(state)

    @property
    def diagnostics(self):
        """Detached reporting facade for canonical dynamics diagnostics."""
        if not hasattr(self, "_diagnostics"):
            from ..diagnostics import DryHydrostaticDiagnostics

            self._diagnostics = DryHydrostaticDiagnostics(self)
        return self._diagnostics


__all__ = ["DryHydrostaticDynamics"]
