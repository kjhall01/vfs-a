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
from vfs.ocean.helpers import nonzero_divisor
from vfs.ocean.transport import divergence
from vfs.ocean.temperature.eddy_energy import EddyState, EddyMixingLength


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
    gm_diffusivity: torch.Tensor | None = None  # active edge × layer, feedback only
    redi_limited_fluxes: int = 0  # row/positive-field fluxes reduced, opt-in only
    gm_limiting_passes: int = 0
    gm_limited_interfaces: int = 0


class LocalEddyEnergyError(FloatingPointError):
    """Negative eddy inventory after settling the finite work in a cell-layer."""

    def __init__(
        self, name, bad, detail="reduce dt or supply a physical reservoir initial condition",
        *, remaining=None,
    ):
        super().__init__(f"{name} requires unavailable local eddy energy; {detail}")
        self.bad = bad
        self.remaining = remaining


@torch.no_grad()
def _funded_factors(model, before, after, failure, metrics):
    """Largest affordable halving along the current local GM transport direction.

    With fixed donors the inflow and tracer increments scale linearly. The only
    quadratic term in the remaining h*e is -a*(delta hCT)**2/h. Neighboring
    layers can request different factors, so the actual endpoint must be checked.
    """
    h = before.ocean.h
    bad = failure.bad
    start = before.eddy_h[bad]
    curvature = model.hamiltonian.base.coefficients(before.ocean)[0]
    delta_ct = after.ocean.temp_h[bad] - before.ocean.temp_h[bad]
    quadratic = -curvature[bad] * delta_ct.square() / h[bad]
    fractions = 0.5 ** torch.arange(1, 21, dtype=h.dtype, device=h.device)
    remaining = failure.remaining[bad]
    budget = (1 - fractions) * start[:, None] + fractions * remaining[:, None]
    budget = budget + (fractions * (fractions - 1)) * quadratic[:, None]
    best = torch.where(budget >= 0, fractions, 0.).amax(1)
    layer = torch.ones_like(h)
    layer[bad] = torch.where(best > 0, best, 0.5)
    return GMRedi._interface_min(layer, metrics)


class _RediGradient:
    """R = horizontal difference + relative slope * edge average * D_z.

    Values have shape (cell, layer, field). The transpose reverses these
    operations exactly; no edge x layer x stencil x field tensor is formed.
    Absolute weights retain the original uncoalesced explicit-step bound.
    """

    def __init__(self, metrics, relative):
        z, self.ids, self.weights, self.i, self.j, self.dc, _, _, _ = metrics
        self.shape = z.shape
        self.relative = relative

    def vertical(self, values):
        # This stencil has exactly three entries. Unpack them rather than
        # launching a Python loop; keep the original left-to-right sum.
        flat = values.reshape(-1, values.shape[-1])
        i0, i1, i2 = self.ids.unbind(-1)
        w0, w1, w2 = self.weights.unbind(-1)
        return (
            w0[..., None] * flat[i0]
            + w1[..., None] * flat[i1]
            + w2[..., None] * flat[i2]
        )

    def apply(self, values):
        dz = self.vertical(values)
        return (values[self.j] - values[self.i]) / self.dc[
            ..., None
        ] + 0.5 * self.relative[..., None] * (dz[self.i] + dz[self.j])

    def face_mean(self, values):
        return 0.5 * (values[self.i] + values[self.j])

    def transpose(self, flux, *, absolute=False):
        # Horizontal difference transpose uses the existing conservative ledger.
        horizontal = flux / self.dc[..., None]
        if absolute:
            result = flux.new_zeros((*self.shape, flux.shape[-1]))
            result = result.index_add(0, self.i, horizontal).index_add(
                0, self.j, horizontal
            )
        else:
            result = divergence(self.i, self.j, horizontal, self.shape[0])
        slope = self.relative.abs() if absolute else self.relative
        contribution = 0.5 * slope[..., None] * flux
        column = (
            torch.zeros_like(result)
            .index_add(0, self.i, contribution)
            .index_add(0, self.j, contribution)
        )
        weights = self.weights.abs() if absolute else self.weights
        flat = result.reshape(-1, flux.shape[-1])
        for k in range(3):
            flat = flat.index_add(
                0,
                self.ids[..., k].flatten(),
                (weights[..., k, None] * column).reshape_as(flat),
            )
        return flat.reshape_as(result)

    def step_bound(self, measure):
        # |R|^T M |R| 1, using the same eight entries as the original stencil,
        # including repeated center entries rather than coalescing them.
        vertical_sum = self.weights.abs().sum(-1)
        row_sum = 2 / self.dc + 0.5 * self.relative.abs() * (
            vertical_sum[self.i] + vertical_sum[self.j]
        )
        return self.transpose((measure * row_sum)[..., None], absolute=True)[..., 0]

    def positive_flux_scale(self, transfer, amounts):
        """Locally fund row fluxes from nonnegative cell/layer inventories.

        |R|^T |transfer| bounds every node's total possible withdrawal before
        cancellations. A row takes the smallest allowance of its donor entries,
        independently for each field. Scaling the WHOLE row retains R^T's zero
        column sum and the negative-semidefinite diffusion form R^T M alpha R.
        No tracer amount is clipped, and no nonlocal minimum is taken.
        """
        demand = self.transpose(transfer.abs(), absolute=True)
        safe = torch.where(demand > 0, demand, torch.ones_like(demand))
        ratio = torch.minimum(torch.ones_like(amounts), amounts / safe)
        ratio = torch.where(demand > 0, ratio, torch.ones_like(ratio))
        # Leave summation room only on limited fluxes; this is not a state floor.
        ratio = torch.where(
            ratio < 1, ratio * (1 - 64 * torch.finfo(ratio.dtype).eps), ratio
        )
        scale = torch.ones_like(transfer)

        def fund(weight, allowance):
            nonlocal scale
            scale = torch.minimum(
                scale,
                torch.where(
                    weight[..., None] * transfer < 0, allowance, torch.ones_like(scale)
                ),
            )

        fund(-1 / self.dc, ratio[self.i])
        fund(1 / self.dc, ratio[self.j])
        flat = ratio.reshape(-1, ratio.shape[-1])
        for cells in (self.i, self.j):
            for k in range(3):
                fund(
                    0.5 * self.relative * self.weights[cells, :, k],
                    flat[self.ids[cells, :, k]],
                )
        return scale


class GMRedi:
    """Optional first-order split step; coefficients in m²/s, N² floor in s⁻².

    GM uses donor advection. The default rotated Redi stencil has no maximum
    principle: reject negative stores rather than clipping. Opt-in positive_redi
    funds local row fluxes from SA/passive/TKE/eddy inventories; the same limited
    flux supplies the energy ledger. This preserves positivity and variance
    dissipation, not a full maximum principle. Coefficients are not calibrated.
    The midpoint GM energy flux is the historical default. donor_parcel
    carries the upwind parcel's energy. funded_donor_parcel uses this flux
    and halves incident interface streamfunctions if local GM work exceeds
    the eddy reservoir. quadratic_funded_donor_parcel predicts a local dyadic
    scale from the Roquet energy curve and verifies the full trial before Redi.
    """

    energy_schemes = (
        "midpoint", "donor_parcel", "funded_donor_parcel", "quadratic_funded_donor_parcel"
    )

    def __init__(
        self,
        kappa_gm=0.0,
        kappa_redi=0.0,
        *,
        max_slope=0.01,
        min_n2=1e-8,
        cfl=0.8,
        energy_closure=None,
        positive_redi=False,
        gm_energy_scheme="midpoint",
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
        if energy_closure is not None:
            if not isinstance(energy_closure, EddyMixingLength):
                raise TypeError("energy_closure must be EddyMixingLength")
            if kappa_gm != 0:
                raise ValueError("choose prescribed kappa_gm OR energy_closure")
        self.energy_closure = energy_closure
        if type(positive_redi) is not bool:
            raise ValueError("positive_redi must be boolean")
        self.positive_redi = positive_redi
        if gm_energy_scheme not in self.energy_schemes:
            raise ValueError("Unknown GM energy scheme: " + str(gm_energy_scheme))
        self.gm_energy_scheme = gm_energy_scheme
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
    def _interface_min(layer, metrics):
        if hasattr(metrics, "interface_min"):
            return metrics.interface_min(layer)
        i, j = metrics[3:5]
        sides = torch.minimum(layer[i], layer[j])
        return torch.minimum(sides[:, :-1], sides[:, 1:])

    def _gm_diffusivity(self, state, metrics):
        if self.energy_closure is None:
            return None
        return self.energy_closure.diffusivity(state, metrics[3], metrics[4])

    @staticmethod
    def _metrics(model, state):
        """Geometry shared only within this fixed-h/u GM–Redi call."""
        s = state.ocean
        h, mesh = s.h, s.mesh
        z = model.hamiltonian.base.height(s)
        vertical_ids, vertical_weights = GMRedi._vertical_rows(z)
        domain = model.domain
        active = (
            torch.ones(mesh.nEdges, dtype=torch.bool, device=h.device)
            if domain is None
            else domain.active_edge
        )
        i, j = mesh.cellsOnEdge[active].unbind(1)
        dc, ell = mesh.dc_edge[active, None], mesh.dv_edge[active, None]
        ids = torch.arange(h.numel(), device=h.device).reshape_as(h)
        return z, vertical_ids, vertical_weights, i, j, dc, ell, ids, (z[j] - z[i]) / dc

    def _geometry(self, model, state, *, part="all", metrics=None, gm_diffusivity=None):
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
        operator = measure = None
        if part != "gm":
            operator = _RediGradient(metrics, relative)
            # Suppress the ENTIRE rotated diffusion row in weak/unstable layers.
            measure = self.kappa_redi * ell * dc * 0.5 * (h[i] + h[j]) * stable
        if part == "redi":
            return slope, None, None, None, None, None, None, operator, measure
        # Psi at interfaces; exact zero at closed upper/lower boundaries.
        zero = torch.zeros_like(slope[:, :1])
        if self.energy_closure is None:
            # Preserve the prescribed-coefficient expression and reduction order.
            interior = self.kappa_gm * 0.5 * (slope[:, :-1] + slope[:, 1:])
        else:
            kappa = (
                self.energy_closure.diffusivity(state, i, j)
                if gm_diffusivity is None
                else gm_diffusivity
            )
            transport = kappa * slope
            interior = 0.5 * (transport[:, :-1] + transport[:, 1:])
        psi = torch.cat((zero, interior, zero), 1)
        return slope, psi, *self._carriers(psi, metrics, h), operator, measure

    @staticmethod
    def _carriers(psi, metrics, h):
        """Pair horizontal and vertical bolus fluxes from a shared streamfunction."""
        i, j, ell, ids = metrics[3], metrics[4], metrics[6], metrics[7]
        hi, hj = ids[i].flatten(), ids[j].flatten()
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
        return horizontal, vertical, ii, jj, carrier

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
            self._contents(before) / nonzero_divisor(h)[..., None]
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
        end = EddyState.from_layers(
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
        The selected energy flux defines the spatial work allocation.
        """
        H = model.hamiltonian
        work = H.finite_change(before, transported) - energy_inflow
        mass = H.mass_area(before.ocean)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        eddy_h = transported.eddy_h - work / safe_mass
        bad = eddy_h < 0
        if bad.any():
            raise LocalEddyEnergyError(name, bad, remaining=eddy_h.detach())
        end = EddyState.from_layers(transported.ocean, transported.tke_h, eddy_h)
        # Transported CT/SA/TKE were already checked; exchange changes only e.
        model._check_eddy(end)
        exchange = (mass * (end.eddy_h - transported.eddy_h)).sum()
        residual = H.finite_change(before, end) - energy_inflow
        return end, exchange, residual

    def _gm_transport(self, model, state, dt, initial, volume, i, j, carrier):
        h = state.ocean.h
        rho0 = state.ocean.column.rho0
        outgoing = (
            torch.zeros_like(volume)
            .index_add(0, i, carrier.clamp_min(0))
            .index_add(0, j, (-carrier).clamp_min(0))
        )
        fraction = dt * outgoing / nonzero_divisor(volume)
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
        gm_values = initial + divergence(i, j, transfer, h.numel()) / nonzero_divisor(volume)[:, None]
        gm_transport = self._unpack(model, state, gm_values, initial)
        if self.gm_energy_scheme == "midpoint":
            mu = model.hamiltonian.midpoint_conjugates(state, gm_transport).reshape_as(initial)
            energy_flux = rho0 * (0.5 * (mu[i] + mu[j]) * transfer).sum(-1)
        else:
            # Transport upwind CT, SA and reservoirs. For its Roquet potential,
            # evaluate this same parcel in the two adjacent layer geometries.
            donor_ids = torch.where(carrier >= 0, i, j)
            CT, SA = initial[donor_ids, 0], initial[donor_ids, 1]
            a, b, c, d = [
                0.5 * (p.flatten()[i] + p.flatten()[j])
                for p in model.hamiltonian.base.coefficients(state.ocean)
            ]
            potential = (a * CT + b) * CT + c * SA + d
            caloric = model.config.cp * (CT - model.config.CT0)
            reservoir = initial[donor_ids, -2:].sum(-1)
            energy_flux = rho0 * dt * carrier * (caloric + potential + reservoir)
        gm_inflow = divergence(i, j, energy_flux, h.numel()).reshape_as(h)
        return gm_transport, gm_inflow, fraction

    def apply(self, model, state, dt):
        from vfs.ocean.model import Ocean

        if not isinstance(model, Ocean):
            raise TypeError("research GMRedi requires Ocean CT/SA")
        if not isinstance(state, EddyState):
            raise TypeError("GMRedi requires an explicit EddyState reservoir")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("positive finite dt required")
        model.check(state)
        s, h = state.ocean, state.ocean.h
        volume = (s.mesh.cell_area[:, None] * h).flatten()
        rho0 = s.column.rho0
        metrics = self._metrics(model, state)
        gm_diffusivity = self._gm_diffusivity(state, metrics)
        geom = self._geometry(
            model, state, part="gm", metrics=metrics, gm_diffusivity=gm_diffusivity
        )
        slope, psi, horizontal, vertical, i, j, carrier, _, _ = geom
        initial = (self._contents(state) / nonzero_divisor(h)[..., None]).reshape(h.numel(), -1)
        # Retry from the same state. Halving an entire closed circulation
        # preserves zero divergence, including its two vertical return paths.
        # Only a local GM energy deficit can initiate funding retries.
        proposed = psi
        factors = None
        first_halving = None
        halvings = 0
        passes = 0
        while True:
            try:
                transported, gm_inflow, fraction = self._gm_transport(
                    model, state, dt, initial, volume, i, j, carrier
                )
                gm, gm_exchange, gm_residual = self._settle(
                    model, state, transported, gm_inflow, "GM"
                )
                break
            except (FloatingPointError, ValueError) as exc:
                if first_halving is not None:
                    if passes == 1 and isinstance(exc, LocalEddyEnergyError):
                        # Recompute once along the actual, partly reduced circulation.
                        correction = _funded_factors(model, state, transported, exc, metrics)
                        factors = (factors * correction).clamp_min(0.5**20)
                    else:
                        # The interacting layers still need repair. Replay the original
                        # halving sequence with its full budget of twenty attempts.
                        factors, first_halving = first_halving, None
                else:
                    if not isinstance(exc, LocalEddyEnergyError) or self.gm_energy_scheme not in (
                        "funded_donor_parcel", "quadratic_funded_donor_parcel"
                    ):
                        raise
                    if halvings == 20:
                        raise LocalEddyEnergyError(
                            "GM", exc.bad, "exhausted 20 local streamfunction halvings"
                        ) from exc
                    incident = ~self._interface_min(~exc.bad, metrics)
                    if not incident.any():
                        raise
                    if factors is None:
                        factors = torch.ones_like(psi[:, 1:-1])
                    factors = torch.where(incident, 0.5 * factors, factors)
                    halvings += 1
                    if passes == 0 and self.gm_energy_scheme == "quadratic_funded_donor_parcel":
                        first_halving = factors
                        factors = _funded_factors(model, state, transported, exc, metrics)
            passes += 1
            psi = torch.cat((proposed[:, :1], proposed[:, 1:-1] * factors, proposed[:, -1:]), 1)
            horizontal, vertical, i, j, carrier = self._carriers(psi, metrics, h)

        # Redi follows the UPDATED density surfaces, not the pre-GM slopes.
        redi_geometry = self._geometry(model, gm, part="redi", metrics=metrics)
        redi_slope = redi_geometry[0]
        operator, measure = redi_geometry[-2:]
        values = (self._contents(gm) / nonzero_divisor(h)[..., None]).reshape_as(initial)
        column_values = values.reshape(*h.shape, -1)
        gradient = operator.apply(column_values)
        flux = measure[..., None] * gradient
        bound = operator.step_bound(measure).flatten() / nonzero_divisor(volume)
        step_number = dt * bound
        if (step_number > self.cfl).any():
            raise FloatingPointError("Redi explicit step exceeds bound; reduce dt")
        limited_fluxes = 0
        if self.positive_redi:
            # CT is signed; SA, passives, TKE and mesoscale energy are positive
            # inventories. Use the SAME limited flux in both transport and H.
            scale = operator.positive_flux_scale(
                -dt * flux[..., 1:],
                volume.reshape(*h.shape, 1) * column_values[..., 1:],
            )
            limited_fluxes = int((scale < 1).sum())
            flux = torch.cat((flux[..., :1], flux[..., 1:] * scale), -1)
        rate_amount = operator.transpose(-dt * flux).reshape_as(values)
        final_values = values + rate_amount / nonzero_divisor(volume)[:, None]
        redi_transport = self._unpack(model, gm, final_values, values)
        mu = model.hamiltonian.midpoint_conjugates(gm, redi_transport).reshape_as(
            initial
        )
        # A single conjugate per row defines a conservative energy flux over
        # the SAME rotated stencil; sum(weights)=0. Its remainder is conversion.
        column_mu = mu.reshape_as(column_values)
        mu_face = operator.face_mean(column_mu)
        energy_flux = rho0 * (-dt * (flux * mu_face).sum(-1))
        redi_inflow = operator.transpose(energy_flux[..., None])[..., 0]
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
            gm_diffusivity,
            limited_fluxes,
            passes,
            0 if factors is None else int((factors < 1).sum()),
        )
        return end, report
