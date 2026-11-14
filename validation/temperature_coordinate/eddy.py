"""Research GM/Redi transport on the existing Roquet CT/SA state.

Fixed geometry during this explicit operator; arbitrary positive layer thickness.
GM is a discrete divergence-free circulation, including interlayer transport.
Redi uses an adjoint small-slope operator in layer coordinates. Slopes follow
surface-referenced potential density, NOT exact neutral directions. GM and Redi
exchange actual finite Hamiltonian work with an explicit mesoscale reservoir.
Neither momentum backscatter nor reservoir dissipation is supplied.
"""

from dataclasses import dataclass
import math
import torch
from vfs.ocean.transport import divergence
from eddy_energy import EddyState


@dataclass(frozen=True)
class EddyReport:
    slope: torch.Tensor
    streamfunction: torch.Tensor
    horizontal_transport: torch.Tensor
    vertical_transport: torch.Tensor
    volume_residual: torch.Tensor
    gm_outflow_fraction: torch.Tensor
    redi_step_number: torch.Tensor
    inventory_change: torch.Tensor
    redi_variance_change: torch.Tensor
    energy_change: torch.Tensor
    gm_energy_change: torch.Tensor
    redi_energy_change: torch.Tensor
    redi_slope: torch.Tensor
    reservoir_change: torch.Tensor
    gm_reservoir_exchange: torch.Tensor
    redi_reservoir_exchange: torch.Tensor
    gm_energy_flux_divergence: torch.Tensor
    redi_energy_flux_divergence: torch.Tensor
    gm_local_energy_residual: torch.Tensor
    redi_local_energy_residual: torch.Tensor


class GMRedi:
    """Optional first-order split step; coefficients in m²/s, N² floor in s⁻².

    GM uses donor advection. Redi dissipates tracer variance but its rotated
    stencil has no maximum principle: reject negative passives/TKE or invalid
    CT/SA rather than clipping. Coefficients are prescribed, not calibrated.
    Every call rebuilds geometry/slopes from its input; no hidden state/cache.
    """

    def __init__(
        self, kappa_gm=0.0, kappa_redi=0.0, *, max_slope=0.01, min_n2=1e-8, cfl=0.8
    ):
        values = (kappa_gm, kappa_redi, max_slope, min_n2, cfl)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("finite eddy parameters required")
        if (
            min(kappa_gm, kappa_redi) < 0
            or max_slope <= 0
            or min_n2 <= 0
            or not 0 < cfl < 1
        ):
            raise ValueError("invalid eddy parameters")
        self.kappa_gm, self.kappa_redi = kappa_gm, kappa_redi
        self.max_slope, self.min_n2, self.cfl = max_slope, min_n2, cfl

    @staticmethod
    def _vertical_rows(z):
        """Three-point derivative rows, exact for linear profiles on uneven z.

        End layers use one-sided secants; interior rows average two secants.
        Columns are ordered top to bottom, with z positive UP.
        """
        cells, layers = z.shape
        ids = torch.arange(z.numel(), device=z.device).reshape_as(z)
        upper = torch.cat((ids[:, :1], ids[:, :-1]), 1)
        lower = torch.cat((ids[:, 1:], ids[:, -1:]), 1)
        upz = torch.cat((z[:, :1], z[:, :-1]), 1)
        loz = torch.cat((z[:, 1:], z[:, -1:]), 1)
        wu = torch.ones_like(z) * 0.5
        wl = torch.ones_like(z) * 0.5
        wu[:, 0] = 0
        wl[:, 0] = 1
        wu[:, -1] = 1
        wl[:, -1] = 0
        du = torch.where(wu > 0, z - upz, torch.ones_like(z))
        dl = torch.where(wl > 0, loz - z, torch.ones_like(z))
        a = -wu / du
        c = wl / dl
        return torch.stack((upper, ids, lower), -1), torch.stack((a, -a - c, c), -1)

    @staticmethod
    def _metrics(model, state):
        """Geometry shared only within this fixed-h/u GM–Redi call."""
        s = state.ocean
        h, mesh = s.h, s.mesh
        z = model.ocean.energy.base.height(s)
        vertical_ids, vertical_weights = GMRedi._vertical_rows(z)
        domain = model.ocean.domain
        active = (
            torch.ones(mesh.nEdges, dtype=torch.bool, device=h.device)
            if domain is None
            else domain.active_edge
        )
        i, j = mesh.cellsOnEdge[active].unbind(1)
        dc, ell = mesh.dc_edge[active, None], mesh.dv_edge[active, None]
        ids = torch.arange(h.numel(), device=h.device).reshape_as(h)
        return z, vertical_ids, vertical_weights, i, j, dc, ell, ids, (z[j] - z[i]) / dc

    def _geometry(self, model, state, *, part="all", metrics=None):
        s = state.ocean
        mesh = s.mesh
        h = s.h
        layers = h.shape[1]
        if layers < 2:
            raise ValueError("GMRedi requires at least two layers")
        if metrics is None:
            metrics = self._metrics(model, state)
        z, vertical_ids, vertical_weights, i, j, dc, ell, ids, zn = metrics
        # A SINGLE reference pressure prevents pressure-compression gradients
        # from masquerading as water-mass gradients. Thermobaricity is omitted.
        b = (
            -s.column.gravity
            / s.column.rho0
            * model.config.anomaly(s.temperature(), s.salinity(), torch.zeros_like(z))
        )
        bz = (b.flatten()[vertical_ids] * vertical_weights).sum(-1)
        bz_edge = 0.5 * (bz[i] + bz[j])
        bn = (b[j] - b[i]) / dc
        stable = bz_edge > self.min_n2
        safe = torch.where(stable, bz_edge, torch.ones_like(bz_edge))
        raw = zn - bn / safe
        slope = torch.where(
            stable, raw.clamp(-self.max_slope, self.max_slope), torch.zeros_like(raw)
        )
        # Relative slope in layer-index coordinates. This metric term is what
        # cancels spurious lateral mixing on a tilted computational grid.
        relative = slope - zn
        rows = weights = measure = None
        if part != "gm":
            rows = torch.cat(
                (
                    ids[i][..., None],
                    ids[j][..., None],
                    vertical_ids[i],
                    vertical_ids[j],
                ),
                -1,
            ).reshape(-1, 8)
            weights = torch.cat(
                (
                    -torch.ones_like(h[i])[..., None] / dc[..., None],
                    torch.ones_like(h[j])[..., None] / dc[..., None],
                    0.5 * relative[..., None] * vertical_weights[i],
                    0.5 * relative[..., None] * vertical_weights[j],
                ),
                -1,
            ).reshape(-1, 8)
            # Suppress the ENTIRE rotated diffusion row in weak/unstable layers.
            measure = (
                self.kappa_redi * ell * dc * 0.5 * (h[i] + h[j]) * stable
            ).flatten()
        if part == "redi":
            return slope, None, None, None, None, None, None, rows, weights, measure
        hi, hj = ids[i].flatten(), ids[j].flatten()
        # Psi at interfaces; exact zero at closed upper/lower boundaries.
        zero = torch.zeros_like(slope[:, :1])
        psi = torch.cat(
            (zero, self.kappa_gm * 0.5 * (slope[:, :-1] + slope[:, 1:]), zero), 1
        )
        horizontal = ell * psi.diff(dim=1)
        hdiv = divergence(hi, hj, horizontal.flatten(), h.numel()).reshape_as(h)
        vertical = torch.cat(
            (
                torch.zeros_like(h[:, :1]),
                hdiv.cumsum(1)[:, :-1],
                torch.zeros_like(h[:, :1]),
            ),
            1,
        )
        vi, vj = ids[:, :-1].flatten(), ids[:, 1:].flatten()
        ii = torch.cat((hi, vi))
        jj = torch.cat((hj, vj))
        carrier = torch.cat((horizontal.flatten(), vertical[:, 1:-1].flatten()))
        return slope, psi, horizontal, vertical, ii, jj, carrier, rows, weights, measure

    @staticmethod
    def _energy_change(model, before, after):
        return model.hamiltonian.finite_change(before, after).sum()

    @staticmethod
    def _contents(state):
        s = state.ocean
        fields = [s.temp_h[..., None], s.salt_h[..., None]]
        if s.tracer_h is not None:
            fields.append(s.tracer_h)
        fields.extend((state.tke_h[..., None], state.eddy_h[..., None]))
        return torch.cat(fields, -1)

    def _unpack(self, model, before, values, initial=None):
        s, h = before.ocean, before.ocean.h
        initial = (
            self._contents(before) / h[..., None]
            if initial is None
            else initial.reshape(*h.shape, -1)
        )
        delta = values.reshape_as(initial) - initial
        passive = (
            None if s.tracer_h is None else s.tracer_h + h[..., None] * delta[..., 2:-2]
        )
        ocean = s.with_tensors(
            s.u_e,
            h,
            s.temp_h + h * delta[..., 0],
            s.salt_h + h * delta[..., 1],
            passive,
        )
        end = EddyState(
            ocean, before.tke_h + h * delta[..., -2], before.eddy_h + h * delta[..., -1]
        )
        if (
            (end.tke_h < 0).any()
            or (end.eddy_h < 0).any()
            or (passive is not None and (passive < 0).any())
        ):
            raise FloatingPointError(
                "negative passive/TKE/reservoir: rotated Redi is not positivity preserving"
            )
        model.check(end)
        return end

    @staticmethod
    def _settle(model, before, transported, energy_inflow, name):
        """Local discrete first law: delta H = energy inflow + conversion.

        The flux has zero global divergence. Exchange the remaining LOCAL work
        with h*e; do not repair a global scalar budget or borrow distant energy.
        The specified midpoint energy flux defines the spatial work allocation.
        """
        H = model.hamiltonian
        work = H.finite_change(before, transported) - energy_inflow
        mass = H.mass_area(before.ocean)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        eddy_h = transported.eddy_h - work / safe_mass
        if (eddy_h < 0).any():
            raise FloatingPointError(
                name
                + " requires unavailable local eddy energy; reduce dt or supply a physical reservoir initial condition"
            )
        end = EddyState(transported.ocean, transported.tke_h, eddy_h)
        # Transported CT/SA/TKE were already checked; exchange changes only e.
        model._check_eddy(end)
        exchange = (mass * (end.eddy_h - transported.eddy_h)).sum()
        residual = H.finite_change(before, end) - energy_inflow
        return end, exchange, residual

    def apply(self, model, state, dt):
        from roquet import RoquetOcean

        if not isinstance(model, RoquetOcean):
            raise TypeError("research GMRedi requires RoquetOcean CT/SA")
        if not isinstance(state, EddyState):
            raise TypeError("GMRedi requires an explicit EddyState reservoir")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("positive finite dt required")
        model.check(state)
        s, h = state.ocean, state.ocean.h
        volume = (s.mesh.cell_area[:, None] * h).flatten()
        rho0 = s.column.rho0
        metrics = self._metrics(model, state)
        geom = self._geometry(model, state, part="gm", metrics=metrics)
        slope, psi, horizontal, vertical, i, j, carrier, _, _, _ = geom
        initial = (self._contents(state) / h[..., None]).reshape(h.numel(), -1)
        outgoing = (
            torch.zeros_like(volume)
            .index_add(0, i, carrier.clamp_min(0))
            .index_add(0, j, (-carrier).clamp_min(0))
        )
        fraction = dt * outgoing / volume
        if (fraction > self.cfl).any():
            raise FloatingPointError("GM outflow exceeds CFL; reduce dt")
        # The circulation has zero divergence, so subtracting the EOS reference
        # CT/SA changes no tendency. It fixes the energy-flux datum and avoids
        # transporting a large arbitrary thermal/salt background in the ledger.
        reference = initial.new_zeros(initial.shape[-1])
        reference[0], reference[1] = model.config.CT0, model.config.SA0
        anomaly = initial - reference
        donor = torch.where((carrier >= 0)[:, None], anomaly[i], anomaly[j])
        transfer = dt * carrier[:, None] * donor
        gm_values = initial + divergence(i, j, transfer, h.numel()) / volume[:, None]
        gm_transport = self._unpack(model, state, gm_values, initial)
        mu = model.hamiltonian.midpoint_conjugates(state, gm_transport).reshape_as(
            initial
        )
        energy_flux = rho0 * (0.5 * (mu[i] + mu[j]) * transfer).sum(-1)
        gm_inflow = divergence(i, j, energy_flux, h.numel()).reshape_as(h)
        gm, gm_exchange, gm_residual = self._settle(
            model, state, gm_transport, gm_inflow, "GM"
        )

        # Redi follows the UPDATED density surfaces, not the pre-GM slopes.
        redi_geometry = self._geometry(model, gm, part="redi", metrics=metrics)
        redi_slope = redi_geometry[0]
        rows, weights, measure = redi_geometry[-3:]
        values = (self._contents(gm) / h[..., None]).reshape_as(initial)
        # Batched row dot products avoid materializing a second
        # edge×layer×8×tracer array just to reduce it immediately.
        gradient = torch.bmm(weights[:, None, :], values[rows]).squeeze(1)
        flux = measure[:, None] * gradient
        transfers = -dt * weights[..., None] * flux[:, None, :]
        rate_amount = torch.zeros_like(values).index_add(
            0, rows.flatten(), transfers.reshape(-1, values.shape[-1])
        )
        del transfers  # the transpose application above no longer needs this array
        bound = (
            torch.zeros_like(volume).index_add(
                0,
                rows.flatten(),
                (
                    weights.abs() * measure[:, None] * weights.abs().sum(1)[:, None]
                ).flatten(),
            )
            / volume
        )
        step_number = dt * bound
        if (step_number > self.cfl).any():
            raise FloatingPointError("Redi explicit step exceeds bound; reduce dt")
        final_values = values + rate_amount / volume[:, None]
        redi_transport = self._unpack(model, gm, final_values, values)
        mu = model.hamiltonian.midpoint_conjugates(gm, redi_transport).reshape_as(
            initial
        )
        # A single conjugate per row defines a conservative energy flux over
        # the SAME rotated stencil; sum(weights)=0. Its remainder is conversion.
        mu_face = 0.5 * (mu[rows[:, 0]] + mu[rows[:, 1]])
        # Factor the row scalar product before distributing across its eight
        # stencil entries. Avoid another edge×layer×8×tracer temporary.
        energy_transfers = rho0 * (-dt * weights * (flux * mu_face).sum(-1)[:, None])
        redi_inflow = (
            torch.zeros_like(volume)
            .index_add(0, rows.flatten(), energy_transfers.flatten())
            .reshape_as(h)
        )
        end, redi_exchange, redi_residual = self._settle(
            model, gm, redi_transport, redi_inflow, "Redi"
        )
        H = model.hamiltonian
        mass = H.mass_area(s)
        report = EddyReport(
            slope,
            psi,
            horizontal,
            vertical,
            divergence(i, j, carrier, h.numel()).reshape_as(h),
            fraction.reshape_as(h),
            step_number.reshape_as(h),
            (
                s.mesh.cell_area[:, None, None]
                * (self._contents(end) - self._contents(state))
            ).sum((0, 1)),
            (
                0.5
                * volume[:, None]
                * (final_values - values)
                * (final_values + values)
            ).sum(0),
            H.finite_change(state, end).sum(),
            H.finite_change(state, gm, include_eddy=False).sum(),
            H.finite_change(gm, end, include_eddy=False).sum(),
            redi_slope,
            (mass * (end.eddy_h - state.eddy_h)).sum(),
            gm_exchange,
            redi_exchange,
            gm_inflow,
            redi_inflow,
            gm_residual,
            redi_residual,
        )
        return end, report
