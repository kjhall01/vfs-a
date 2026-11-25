"""Thermal ice phase Hamiltonian relative to the all-liquid freezing state.

Energy is in J/m² and trial frozen mass m is in kg/m² of horizontal cell.
This expression is specific to fresh ice, the linear CT liquidus, the Roquet
polynomial and the two-category footprint law used by ThermalPhase.
"""
import torch

from vfs.ice.ocean_ice import safe


class ReducedPhaseEnergy:
    """Cache column coefficients once for bisection and Newton evaluations."""

    def __init__(self, phase, initial, rows):
        water = initial.ocean.ocean
        column = water.column
        eos = phase.H.model.config
        p = phase.H.slab.parameters
        self.rho = column.rho0
        self.old_mass = phase.H.mass(initial)[rows]
        top = water.h[rows, :1]
        self.all_liquid = top + self.old_mass / self.rho
        eta = column.eta(water.h)[rows]
        bottom = eta - top
        surface = eta + self.old_mass / self.rho
        salt = water.salt_h[rows, :1]
        ct = -p.freezing_slope * salt  # conserved h*CT on the liquidus
        h0 = self.all_liquid

        # Integrate the Roquet parcel potential over [bottom, bottom+h].
        # Its density polynomial along this path is f0 + f1/h + f2/h²
        # before the thermobaric term. The two required height moments are
        # mean(z) = bottom+h/2 and mean(z²) = mean(z)²+h²/12.
        f0 = -eos.s * eos.SA0 - eos.t * eos.CT0 - eos.tt * eos.CT0**2
        f1 = eos.s * salt + eos.t * ct
        f2 = eos.tt * ct.square()
        moment = (bottom + 0.5 * h0).square() + h0.square() / 12
        moment_slope = bottom + 2 * h0 / 3
        anomaly = ct - eos.CT0 * h0
        g = column.gravity
        self.linear = g * (f0 * surface + 0.5 * f1
                           - 0.5 * eos.tz * (anomaly * moment_slope - eos.CT0 * moment))
        self.quadratic = g * (0.5 * f0 - 0.5 * eos.tz * (anomaly / 3 - eos.CT0 * moment_slope))
        self.cubic = g * eos.tz * eos.CT0 / 6
        self.reciprocal = -g * bottom * f2 / h0

        # Completing the hydrostatic square leaves C*sum(V_k²/a_k).
        # Liquid sensible heat and the square in eta+m/rho are constant.
        area = initial.ice.area[rows]
        volume = initial.ice.volume[rows]
        share = volume / safe(self.old_mass)  # category volume per kg of solid
        coefficient = phase.H.coefficient(initial)
        self.category_linear = coefficient * share * (volume / safe(area))
        self.bare_fraction = (p.density * volume - phase.initial_snow[rows]) / safe(p.density * volume)
        self.mass_scale = safe(self.old_mass)
        cover = area.sum(-1, keepdim=True)
        full_area = area / safe(cover)
        self.grow_linear = torch.where(self.old_mass > 0, self.category_linear.sum(-1, keepdim=True),
                                       coefficient * phase.nucleation_thickness / p.density)
        self.grow_quadratic = torch.where(self.old_mass > 0,
                                         (coefficient * share.square() / safe(full_area)).sum(-1, keepdim=True),
                                         coefficient / p.density**2)
        self.full_cover_mass = torch.where(self.old_mass > 0, self.old_mass / safe(cover),
                                           p.density * phase.nucleation_thickness)
        self.cold = phase.initial.cold_content[rows].sum(-1, keepdim=True)
        self.latent_heat = p.latent_heat

    def value(self, mass):
        """E(m) - E(0). Factor differences before evaluation near complete melt."""
        d = -mass / self.rho
        h = self.all_liquid + d
        ocean = d * (self.linear + d * (self.quadratic + d * self.cubic) + self.reciprocal / h)
        ratio = mass / self.mass_scale
        # On melt, snow loss holds area fixed until the bare ice starts to
        # retreat. On growth, thickness is fixed until the cover reaches one.
        melt = (self.category_linear * torch.maximum(ratio, self.bare_fraction)).sum(-1, keepdim=True)
        grow = torch.where(mass < self.full_cover_mass, self.grow_linear, self.grow_quadratic * mass)
        geometry = torch.where(mass < self.old_mass, melt, grow)
        cold = self.cold * ratio.clamp(max=1)
        return ocean + mass * (geometry - self.latent_heat) + cold

    def derivative(self, mass):
        """dE/dm (J/kg) with the same one-sided conventions as ThermalPhase."""
        d = -mass / self.rho
        h = self.all_liquid + d
        ocean = -(self.linear + d * (2 * self.quadratic + 3 * d * self.cubic)
                  + self.reciprocal * self.all_liquid / h.square()) / self.rho
        ratio = mass / self.mass_scale
        melt = (self.category_linear * torch.where(ratio < self.bare_fraction,
                                                   self.bare_fraction, 2 * ratio)).sum(-1, keepdim=True)
        grow = torch.where(mass < self.full_cover_mass, self.grow_linear, 2 * self.grow_quadratic * mass)
        geometry = torch.where(mass < self.old_mass, melt, grow)
        cold = torch.where(mass <= self.old_mass, self.cold / self.mass_scale, 0.0)
        return ocean + geometry - self.latent_heat + cold

    def decreasing_slope(self, mass):
        slope = self.derivative(mass)
        if not (slope < 0).all():
            raise ValueError("nonmonotone phase-energy relation")
        return slope

    def refine(self, mass, residual, upper):
        """Keep the bracketed iterate if Newton would cross a physical endpoint."""
        trial = mass - residual / self.decreasing_slope(mass)
        outside = ((trial < 0) | (trial > upper)) & torch.isfinite(trial)
        return torch.where(outside, mass, trial)
