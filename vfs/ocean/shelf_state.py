"""Native partial-cell inventories and split-face velocity on a moving ocean grid."""

from dataclasses import dataclass, replace

import torch

from .continuity_momentum import transport_with_budget
from .partial_cell_eddy import ShelfRedi, advance_gm_roquet
from .partial_cells import WetFaces, remap_active_contents, remap_roquet
from .transport import HorizontalTransport


@dataclass(frozen=True)
class ShelfState:
    h: torch.Tensor
    temp_h: torch.Tensor
    salt_h: torch.Tensor
    tke_h: torch.Tensor
    eddy_h: torch.Tensor
    velocity: torch.Tensor
    cell_velocity: torch.Tensor
    tracer_h: torch.Tensor | None = None

    def transport(self, mesh, depth, dt, rho0, *, scheme="donor"):
        """Bounded material transport. TKE settles the donor momentum budget."""
        faces = WetFaces.from_columns(depth, self.h, mesh.cellsOnEdge)
        step, settlement = transport_with_budget(
            mesh, depth, self.h, self.velocity, self.cell_velocity, self.tke_h, dt, rho0
        )
        # CT, SA and all material inventories use the same signed water volumes
        # as the changing-face momentum update. Pack EKE as the last passive.
        contents = self.eddy_h[..., None]
        if self.tracer_h is not None:
            contents = torch.cat((self.tracer_h, contents), -1)
        area = mesh.cell_area[:, None]
        graph = (
            (area * self.h).flatten(),
            faces.left,
            faces.right,
            faces.volume_flux(self.velocity, mesh.dv_edge),
        )
        limiter = HorizontalTransport(mesh, scheme=scheme, min_thickness=0.0)
        ledger = limiter.advance_graph(
            self,
            dt,
            graph,
            tracer_h=contents,
            velocity=self.velocity,
            caloric_reference_K=273.15,
        )
        fields = ledger.amounts / area[..., None]
        return ShelfState(
            step.h,
            fields[..., 0],
            fields[..., 1],
            settlement.tke_h,
            fields[..., -1],
            step.face.velocity,
            step.cell_velocity,
            None if self.tracer_h is None else fields[..., 2:-1],
        )

    def mesoscale(self, mesh, depth, energy, gm, redi, dt):
        """Advance the GM circulation and then Redi on the current physical openings."""
        faces = WetFaces.from_columns(depth, self.h, mesh.cellsOnEdge)
        divisor = torch.where(self.h > 0, self.h, 1.0)

        def geometry(ht, hs):
            buoyancy = (
                -energy.gravity
                / energy.rho0
                * energy.config.anomaly(ht / divisor, hs / divisor, 0.0)
            )
            return ShelfRedi.build(
                faces, depth, self.h, buoyancy, mesh.dc_edge, mesh.dv_edge, redi
            )

        op = geometry(self.temp_h, self.salt_h)
        carried = self.tke_h[..., None]
        if self.tracer_h is not None:
            carried = torch.cat((carried, self.tracer_h), -1)
        ht, hs, he, carried = advance_gm_roquet(
            faces,
            depth,
            self.h,
            self.temp_h,
            self.salt_h,
            self.eddy_h,
            mesh.cell_area,
            energy,
            mesh.dv_edge,
            gm * op.slope,
            dt,
            carried_h=carried,
        )
        ht, hs, he, carried = geometry(ht, hs).advance_roquet(
            depth, self.h, ht, hs, he, mesh.cell_area, energy, dt, carried_h=carried
        )
        return replace(
            self,
            temp_h=ht,
            salt_h=hs,
            eddy_h=he,
            tke_h=carried[..., 0],
            tracer_h=None if self.tracer_h is None else carried[..., 1:],
        )

    def rezone(self, mesh, depth, target, energy):
        """Conserve the reservoirs and transport donor vectors through the same rezone."""
        remap = remap_roquet(
            depth,
            self.h,
            target,
            self.temp_h,
            self.salt_h,
            self.velocity,
            mesh.cellsOnEdge,
            mesh.cell_area,
            mesh.dc_edge * mesh.dv_edge,
            energy,
        )
        # PCM gives a convex combination of tangent vectors. Component-wise PLM
        # limiters would generally give different weights on the three axes.
        momentum = remap_active_contents(
            self.h, self.h[..., None] * self.cell_velocity, target, method="pcm"
        )
        safe = torch.where(target > 0, target, 1.0)
        return ShelfState(
            target,
            remap.temp_h,
            remap.salt_h,
            remap_active_contents(self.h, self.tke_h, target),
            remap_active_contents(self.h, self.eddy_h, target),
            remap.velocity,
            momentum / safe[..., None],
            None
            if self.tracer_h is None
            else remap_active_contents(self.h, self.tracer_h, target),
        )
