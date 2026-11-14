"""Research GM/Redi transport on the existing Roquet CT/SA state.

Fixed geometry during this explicit operator; arbitrary positive layer thickness.
GM is a discrete divergence-free circulation, including interlayer transport.
Redi uses an adjoint small-slope operator in layer coordinates. Slopes follow
surface-referenced potential density, NOT exact neutral directions. No energy
compensation, resolved momentum forcing, or prognostic eddy energy is supplied.
"""

from dataclasses import dataclass
import math
import torch
from vfs.ocean.transport import divergence
from vfs.ocean.stratified import TKEState


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

    def _geometry(self, model, state):
        s = state.ocean
        mesh = s.mesh
        h = s.h
        layers = h.shape[1]
        if layers < 2:
            raise ValueError("GMRedi requires at least two layers")
        z = model.ocean.energy.base.height(s)
        vertical_ids, vertical_weights = self._vertical_rows(z)
        domain = model.ocean.domain
        active = (
            torch.ones(mesh.nEdges, dtype=torch.bool, device=h.device)
            if domain is None
            else domain.active_edge
        )
        edges = mesh.cellsOnEdge[active]
        i, j = edges.unbind(1)
        dc = mesh.dc_edge[active, None]
        ell = mesh.dv_edge[active, None]
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
        zn = (z[j] - z[i]) / dc
        stable = bz_edge > self.min_n2
        safe = torch.where(stable, bz_edge, torch.ones_like(bz_edge))
        raw = zn - bn / safe
        slope = torch.where(
            stable, raw.clamp(-self.max_slope, self.max_slope), torch.zeros_like(raw)
        )
        # Relative slope in layer-index coordinates. This metric term is what
        # cancels spurious lateral mixing on a tilted computational grid.
        relative = slope - zn
        ids = torch.arange(h.numel(), device=h.device).reshape_as(h)
        hi, hj = ids[i].flatten(), ids[j].flatten()
        rows = torch.cat(
            (ids[i][..., None], ids[j][..., None], vertical_ids[i], vertical_ids[j]), -1
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
        measure = (self.kappa_redi * ell * dc * 0.5 * (h[i] + h[j]) * stable).flatten()
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
        """Exact finite quadratic Roquet work at fixed h/u; no large subtraction."""
        s, t = before.ocean, after.ocean
        a, b, c, _ = model.ocean.energy.base.coefficients(s)
        dth = t.temp_h - s.temp_h
        dsh = t.salt_h - s.salt_h
        work = (
            model.config.cp + a * (s.temperature() + t.temperature()) + b
        ) * dth + c * dsh
        return (model.energy.mass_area(s) * (work + after.tke_h - before.tke_h)).sum()

    def apply(self, model, state, dt):
        from roquet import RoquetOcean

        if not isinstance(model, RoquetOcean):
            raise TypeError("research GMRedi requires RoquetOcean CT/SA")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("positive finite dt required")
        model.check(state)
        s = state.ocean
        h = s.h
        volume = (s.mesh.cell_area[:, None] * h).flatten()
        geom = self._geometry(model, state)
        slope, psi, horizontal, vertical, i, j, carrier, rows, weights, measure = geom
        q = [s.temperature()[..., None], s.salinity()[..., None]]
        if s.tracer_h is not None:
            q.append(s.tracer_h / h[..., None])
        q.append((state.tke_h / h)[..., None])
        initial = torch.cat(q, -1).reshape(h.numel(), -1)
        outgoing = torch.zeros_like(volume)
        outgoing = outgoing.index_add(0, i, carrier.clamp_min(0)).index_add(
            0, j, (-carrier).clamp_min(0)
        )
        fraction = dt * outgoing / volume
        if (fraction > self.cfl).any():
            raise FloatingPointError("GM outflow exceeds CFL; reduce dt")
        donor = torch.where((carrier >= 0)[:, None], initial[i], initial[j])
        gm = (
            initial
            + dt
            * divergence(i, j, carrier[:, None] * donor, h.numel())
            / volume[:, None]
        )
        # -G^T W G is conservative and negative semidefinite in volume norm.
        gradient = (weights[..., None] * gm[rows]).sum(1)
        flux = measure[:, None] * gradient
        redi_rate = torch.zeros_like(gm).index_add(
            0,
            rows.flatten(),
            (-weights[..., None] * flux[:, None, :]).reshape(-1, gm.shape[-1]),
        )
        # Gershgorin upper bound on spectral radius of M^-1 G^T W G.
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
        final = gm + dt * redi_rate / volume[:, None]

        def unpack(values):
            # Increment original contents, preserving exact identity for inactive
            # cells/operators instead of round-tripping contents through h*q.
            delta = (values - initial).reshape(*h.shape, -1)
            passive = (
                None
                if s.tracer_h is None
                else s.tracer_h + h[..., None] * delta[..., 2:-1]
            )
            out = s.with_tensors(
                s.u_e,
                h,
                s.temp_h + h * delta[..., 0],
                s.salt_h + h * delta[..., 1],
                passive,
            )
            result = TKEState(out, state.tke_h + h * delta[..., -1])
            if (result.tke_h < 0).any() or (
                passive is not None and (passive < 0).any()
            ):
                raise FloatingPointError(
                    "negative passive/TKE: rotated Redi is not positivity preserving"
                )
            model.check(result)
            return result

        gm_state = unpack(gm)
        end = unpack(final)
        eg = self._energy_change(model, state, gm_state)
        er = self._energy_change(model, gm_state, end)
        report = EddyReport(
            slope,
            psi,
            horizontal,
            vertical,
            divergence(i, j, carrier, h.numel()).reshape_as(h),
            fraction.reshape_as(h),
            step_number.reshape_as(h),
            (volume[:, None] * (final - initial)).sum(0),
            (0.5 * volume[:, None] * (final - gm) * (final + gm)).sum(0),
            eg + er,
            eg,
            er,
        )
        return end, report
