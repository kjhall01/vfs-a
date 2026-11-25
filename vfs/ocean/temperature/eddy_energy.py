"""One Roquet Hamiltonian for resolved flow, turbulent TKE and mesoscale energy.

The new independent prognostic content is h*e, not a temperature adjustment or
an alias for boundary-layer TKE. At fixed contents its energy adds no pressure
or momentum derivative. Its conjugate per unit mass is exactly one.
"""

from dataclasses import dataclass
import torch
from vfs.ocean.helpers import nonzero_divisor
from vfs.ocean.stratified import TKEState


@dataclass(frozen=True)
class EddyState(TKEState):
    eddy_h: torch.Tensor  # cell × layer, h*e [m³/s²]; e [J/kg]

    def specific_eddy_energy(self):
        return self.eddy_h / nonzero_divisor(self.ocean.h)


class RoquetHamiltonian:
    """Full H, its fixed-geometry discrete gradient and finite work.

    `resolved` is the existing TemperatureEnergy with the actual Roquet layer
    integral and calorics. There is no separate EOS or approximate PE here.
    """

    def __init__(self, resolved):
        self.resolved = resolved
        self.base = resolved.base
        self.mass_area = resolved.mass_area

    def terms(self, state):
        terms = dict(self.resolved.terms(state.ocean))
        mass = self.mass_area(state.ocean)
        terms["turbulent_kinetic"] = (mass * state.tke_h).sum()
        terms["mesoscale_eddy"] = (
            (mass * state.eddy_h).sum()
            if isinstance(state, EddyState)
            else state.tke_h.new_zeros(())
        )
        terms["total"] = (
            terms["total"] + terms["turbulent_kinetic"] + terms["mesoscale_eddy"]
        )
        return terms

    def kinetic_power(self, state, acceleration):
        """Instantaneous dH/du · acceleration, allocated to cells [W]."""
        mechanical = self.base.mechanical
        s = state.ocean
        edge_power = mechanical.metric(s)*mechanical.velocity(s)*acceleration
        return mechanical.interpolation_transpose(s.mesh, edge_power)

    def kinetic_change(self, before, after):
        """Finite KE change [J/cell/layer] from this Hamiltonian's own metric.

        Thickness is fixed. Edge work is allocated by the same geometric
        interpolation transpose used for the kinetic thickness derivative.
        Summing it gives the full KE change without subtracting large H totals.
        """
        s, t = before.ocean, after.ocean
        if (
            s.mesh is not t.mesh
            or s.column is not t.column
            or (s.h is not t.h and not torch.equal(s.h, t.h))
        ):
            raise ValueError("finite kinetic work requires fixed geometry")
        mechanical = self.base.mechanical
        u, v = mechanical.velocity(s), mechanical.velocity(t)
        delta = v - u
        edge_work = mechanical.metric(s) * (u + 0.5 * delta) * delta
        return mechanical.interpolation_transpose(s.mesh, edge_work)

    @staticmethod
    def _fixed_geometry(before, after):
        s, t = before.ocean, after.ocean
        if (
            s.mesh is not t.mesh
            or s.column is not t.column
            or (s.h is not t.h and not torch.equal(s.h, t.h))
            or (s.u_e is not t.u_e and not torch.equal(s.u_e, t.u_e))
        ):
            raise ValueError(
                "finite tracer work requires unchanged geometry and velocity"
            )
        if isinstance(before, EddyState) != isinstance(after, EddyState):
            raise TypeError("reservoir must be present on both endpoints")

    def midpoint_conjugates(self, before, after):
        """dH/d(amount) divided by rho0; amounts are volume*q.

        Order: CT, SA, optional passives, turbulent TKE, optional mesoscale e.
        The midpoint rule integrates this quadratic Hamiltonian exactly.
        """
        thermal, saline = self._thermal_conjugates(before, after)
        s = before.ocean
        fields = [thermal[..., None], saline[..., None]]
        if s.tracer_h is not None:
            fields.append(torch.zeros_like(s.tracer_h))
        fields.append(torch.ones_like(s.h)[..., None])
        if isinstance(before, EddyState):
            fields.append(torch.ones_like(s.h)[..., None])
        return torch.cat(fields, -1)

    def _thermal_conjugates(self, before, after):
        self._fixed_geometry(before, after)
        s, t = before.ocean, after.ocean
        a, b, c, _ = self.base.coefficients(s)
        return self.base.config.cp + a * (s.temperature() + t.temperature()) + b, c

    def finite_change(self, before, after, *, include_eddy=True):
        """Local finite Hamiltonian change [J], without subtracting large H totals."""
        thermal, saline = self._thermal_conjugates(before, after)
        s, t = before.ocean, after.ocean
        work = thermal * (t.temp_h - s.temp_h) + saline * (t.salt_h - s.salt_h)
        work = work + (after.tke_h - before.tke_h)
        if include_eddy and isinstance(before, EddyState):
            work = work + (after.eddy_h - before.eddy_h)
        return self.mass_area(s) * work


@dataclass(frozen=True)
class EddyMixingLength:
    """GM diffusivity c L sqrt(2 f E) [m²/s], with explicitly supplied scales.

    E is total mesoscale energy; f is an assumed kinetic fraction, not a second
    prognostic reservoir. This is a mixing-length closure, not GEOMETRIC.
    No diffusivity/energy floor, seed, clipping or automatic funding limiter.
    """

    length: float
    efficiency: float
    kinetic_fraction: float

    def __post_init__(self):
        import math

        if (
            not all(
                math.isfinite(x)
                for x in (self.length, self.efficiency, self.kinetic_fraction)
            )
            or self.length <= 0
            or self.efficiency <= 0
            or not 0 < self.kinetic_fraction <= 1
        ):
            raise ValueError(
                "positive finite mixing length/efficiency and 0 < kinetic_fraction <= 1 required"
            )

    def diffusivity(self, state, i, j):
        # Average specific energy onto the same active edges as GM's slopes.
        e = state.specific_eddy_energy()
        edge = 0.5 * (e[i] + e[j])
        return self.from_edge_energy(edge)

    def from_edge_energy(self, edge):
        if not torch.isfinite(edge).all() or (edge < 0).any():
            raise ValueError("finite nonnegative eddy energy required")
        # sqrt has a singular derivative at zero. Evaluate only its positive
        # branch and explicitly select the zero subgradient at exactly E=0.
        positive = edge > 0
        speed = torch.sqrt(
            2
            * self.kinetic_fraction
            * torch.where(positive, edge, torch.ones_like(edge))
        )
        kappa = (
            self.efficiency
            * self.length
            * torch.where(positive, speed, torch.zeros_like(speed))
        )
        if not torch.isfinite(kappa).all():
            raise ValueError("nonfinite energy-dependent GM diffusivity")
        return kappa


@dataclass(frozen=True)
class EddyDissipationReport:
    dissipated_energy: torch.Tensor  # J, nonnegative reservoir debit
    thermal_energy_gain: torch.Tensor  # J, includes thermodynamic PE change
    local_energy_residual: torch.Tensor  # J per cell/layer


@dataclass(frozen=True)
class EddyDissipation:
    """Exact exponential reservoir decay to full-H thermal energy.

    Timescale [s] is explicit and uncalibrated. All dissipated mesoscale energy
    heats the same cell/layer; turbulent TKE and momentum are unchanged.
    """

    timescale: float

    def __post_init__(self):
        import math

        if not math.isfinite(self.timescale) or self.timescale <= 0:
            raise ValueError("positive finite eddy dissipation timescale required")

    def apply(self, model, state, dt):
        import math
        from vfs.ocean.temperature.roquet import RoquetOcean, RoquetWorkspace

        if not isinstance(model, RoquetOcean) or not isinstance(state, EddyState):
            raise TypeError("eddy dissipation requires RoquetOcean and EddyState")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("positive finite dt required")
        model.check(state)
        # expm1 preserves small physical decay fractions; no explicit sink CFL.
        removed = state.eddy_h * (-math.expm1(-dt / self.timescale))
        remaining = state.eddy_h - removed
        s = state.ocean
        # This existing algebraic quadratic solve includes cp*CT AND the exact
        # Roquet potential-energy change; adding removed/(h*cp) would not close H.
        heated, _, _ = RoquetWorkspace(model.ocean, s).heat(s, s, removed / nonzero_divisor(s.h))
        end = EddyState(heated, state.tke_h, remaining)
        model._check_eddy(end)  # other fields validated by the matched heat solve
        H = model.hamiltonian
        thermal = H.finite_change(state, end, include_eddy=False)
        return end, EddyDissipationReport(
            (H.mass_area(s) * (state.eddy_h - end.eddy_h)).sum(),
            thermal.sum(),
            H.finite_change(state, end),
        )
