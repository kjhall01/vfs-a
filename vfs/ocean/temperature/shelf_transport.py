"""Resolved CT/SA transport across physical wet intersections in the shelf campaign."""

from dataclasses import dataclass

import torch

from vfs.ocean.temperature.model import TemperatureOcean
from vfs.ocean.partial_cells import layer_interfaces
from vfs.ocean.transport import HorizontalTransport, TransportLedger, divergence


@dataclass(frozen=True)
class SplitFluxGraph:
    left: torch.Tensor
    right: torch.Tensor
    velocity_slot: torch.Tensor
    area: torch.Tensor

    @classmethod
    def build(cls, state, domain):
        """Intersect the two physical columns with their staggered velocity layers."""
        h, mesh = state.h, state.mesh
        edge = torch.where(domain.active_edge)[0]
        a, b = mesh.cellsOnEdge[edge].T
        interfaces = layer_interfaces(domain.depth[:, 0], h)
        left, right = interfaces[a], interfaces[b]
        # The momentum metric uses the same area-weighted interpolation of h.
        middle = mesh.cell_to_edge(interfaces)[edge].contiguous()
        top = torch.maximum(left[:, :1], right[:, :1])
        bed = torch.minimum(left[:, -1:], right[:, -1:])
        # A weighted mean of equal endpoints can round a ULP outside the
        # two columns. Pin the end bins to the shared opening so every
        # emitted interval has a valid velocity owner on this very edge.
        middle = torch.cat(
            (
                torch.minimum(middle[:, :1], top),
                middle[:, 1:-1],
                torch.maximum(middle[:, -1:], bed),
            ),
            1,
        )
        if domain.active_edge_layers is not None:
            # The deepest shared velocity bin owns the remainder of the opening.
            middle = torch.cat((middle[:, :1], torch.where(
                domain.active_edge_layers[edge, 1:], middle[:, 1:-1].clamp(max=bed), bed
            ), bed), 1)
        ends = (
            torch.cat((left, middle, right), 1).clamp(min=top, max=bed).sort(1).values
        )
        widths = ends.diff(dim=1)
        row, slot = torch.where(widths > 0)
        starts = ends[:, :-1].contiguous()

        def owner(z, ids):
            index = (
                torch.searchsorted(z.contiguous(), starts, right=True)[row, slot] - 1
            )
            return ids[row] * h.shape[1] + index

        return cls(
            owner(left, a),
            owner(right, b),
            owner(middle, edge),
            mesh.dv_edge[edge[row]] * widths[row, slot],
        )

    def force(self, jump, metric):
        """Negative adjoint of opening volume flux in the native kinetic metric."""
        impulse = metric.new_zeros(metric.numel()).index_add(
            0, self.velocity_slot, -self.area * jump
        )
        return impulse.reshape_as(metric) / torch.where(metric > 0, metric, 1.0)


@dataclass(frozen=True)
class ShelfTransportReport:
    transport: TransportLedger
    faces: SplitFluxGraph
    metric: torch.Tensor


class ShelfTemperatureOcean(TemperatureOcean):
    """Use one split carrier graph for continuity, bounded CT/SA and pressure work."""

    def __init__(self, ocean):
        super().__init__(ocean)
        self.matched_openings = False
        self.limiter = HorizontalTransport(
            ocean.mesh,
            domain=ocean.domain,
            min_thickness=ocean.base.dynamics.min_thickness,
        )

    def tendency_with_report(self, state, dt):
        self.fields(state)
        faces = SplitFluxGraph.build(state, self.ocean.domain)
        s, mesh = state, state.mesh
        cell_area = mesh.cell_area[:, None]
        volume = (cell_area * s.h).flatten()
        i, j = faces.left, faces.right
        flux = faces.area * s.u_e.flatten()[faces.velocity_slot]
        ledger = self.limiter.advance_graph(
            s, dt, (volume, i, j, flux), caloric_reference_K=273.15
        )
        base = self.ocean.energy.base
        gradients = base.gradients(s)
        metric = base.mechanical.metric(s)

        def jump(field):
            q = (gradients[field] / cell_area).flatten()
            return q[j] - q[i]

        q = torch.stack((s.temperature(), s.salinity()), -1).reshape(-1, 2)
        trace = 0.5 * (q[i] + q[j]) + ledger.trace_change[:, :2]
        # Roquet caloric enthalpy is linear in CT, hence its conjugate has zero jump.
        force = faces.force(
            jump("h") + trace[:, 0] * jump("temp_h") + trace[:, 1] * jump("salt_h"),
            metric,
        )
        rotation = self.ocean.base.dynamics.mechanical.rotation_acceleration(s)
        dh = divergence(i, j, flux, s.h.numel()).reshape_as(s.h) / cell_area
        dq = (
            divergence(i, j, ledger.integrated_flux / dt, s.h.numel()).reshape_as(
                ledger.amounts
            )
            / cell_area[..., None]
        )
        rhs = s.with_tensors(
            rotation + force,
            dh,
            dq[..., 0],
            dq[..., 1],
            None if s.tracer_h is None else dq[..., 2:],
        )
        return rhs, ShelfTransportReport(ledger, faces, metric)

    def loading_acceleration(self, state, load, report):
        # A column load is the same pressure conjugate at every wet level.
        faces = report.faces
        n = state.h.shape[1]
        p = load.flatten()
        return faces.force(p[faces.right // n] - p[faces.left // n], report.metric)

    def euler(self, state, dt, *, extra_tendency=None, transported_source=None):
        rhs, report = self.tendency_with_report(state, dt)
        end = report.transport.install(state, velocity=state.u_e + dt * rhs.u_e)
        if extra_tendency is not None or transported_source is not None:
            du, dq = transported_source(state, end) if transported_source is not None else extra_tendency(state)
            end = end.with_tensors(
                end.u_e + dt * du, end.h, end.temp_h, end.salt_h, end.tracer_h + dt * dq
            )
        end = self.finish_euler(state, end)
        self.fields(end)
        return end

    def _finish_faces(self, end, parents):
        """Remap only edges that change support. Keep ordinary RK arithmetic elsewhere."""
        from .shelf_momentum import project_changed_faces
        from .roquet import roquet_heat

        mechanical = self.ocean.energy.base.mechanical
        new = mechanical.edge_thickness(end, end.h)
        wet = new > 0
        u = torch.where(wet, end.u_e, 0.)
        # Each Euler or RK donor supplies its own physically supported velocity.
        # Reconstruct the blend on transitioning edges, including its wet parent.
        old_widths = [mechanical.edge_thickness(state, state.h) for state, _, _ in parents]
        changed = torch.stack([((old > 0) != wet).any(1) for old in old_widths]).any(0)
        edges = torch.where(changed)[0]
        if edges.numel():
            mixed = torch.zeros_like(new[edges])
            shear = torch.zeros_like(mixed)
            for old, (_, velocity, weight) in zip(old_widths, parents):
                changed_parent = ((old[edges] > 0) != wet[edges]).any(1, keepdim=True)
                mapped, heat = project_changed_faces(old[edges], new[edges], velocity[edges])
                native = torch.where(wet[edges], velocity[edges], 0.)
                mixed = mixed + weight * torch.where(changed_parent, mapped, native)
                shear = shear + weight * torch.where(changed_parent, heat, 0.)
            u = u.index_copy(0, edges, mixed)
            # Deposit only shear lost by the face projection; ordinary RK
            # truncation and the work of a moving free surface remain separate.
            mesh = end.mesh
            joules = mechanical.rho0 * (mesh.dc_edge[edges] * mesh.dv_edge[edges])[:, None] * shear
            edge_heat = torch.zeros_like(new).index_copy(0, edges, joules)
            cell_heat = mechanical.interpolation_transpose(mesh, edge_heat)
            mass = self.ocean.energy.mass_area(end) * end.h
            specific = cell_heat / torch.where(mass > 0, mass, 1.)
            end, _, _ = roquet_heat(self.ocean, end, specific, end)
        return end.with_tensors(u, end.h, end.temp_h, end.salt_h, end.tracer_h)

    def finish_euler(self, initial, end):
        if not self.matched_openings:
            return end
        # The force impulse is already included on the old wet support.
        return self._finish_faces(end, [(initial, end.u_e, 1.)])

    def finish_blend(self, first, second, end, weights):
        if not self.matched_openings:
            return end
        return self._finish_faces(end, [(first, first.u_e, weights[0]),
                                        (second, second.u_e, weights[1])])
