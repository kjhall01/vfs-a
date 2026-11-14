"""One Roquet Hamiltonian for resolved flow, turbulent TKE and mesoscale energy.

The new independent prognostic content is h*e, not a temperature adjustment or
an alias for boundary-layer TKE. At fixed contents its energy adds no pressure
or momentum derivative. Its conjugate per unit mass is exactly one.
"""

from dataclasses import dataclass
import torch
from vfs.ocean.stratified import TKEState


@dataclass(frozen=True)
class EddyState(TKEState):
    eddy_h: torch.Tensor  # cell × layer, h*e [m³/s²]; e [J/kg]

    def specific_eddy_energy(self):
        return self.eddy_h / self.ocean.h


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
