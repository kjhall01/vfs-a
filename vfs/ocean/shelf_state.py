"""Native partial-cell inventories and split-face velocity on a moving ocean grid."""

from dataclasses import dataclass, replace

import torch

from .continuity_momentum import transport_with_budget
from .partial_cell_eddy import ShelfRedi, advance_gm_roquet
from .partial_cells import WetFaces, remap_active_contents, remap_roquet


@dataclass(frozen=True)
class ShelfState:
    h: torch.Tensor
    temp_h: torch.Tensor
    salt_h: torch.Tensor
    tke_h: torch.Tensor
    eddy_h: torch.Tensor
    velocity: torch.Tensor
    cell_velocity: torch.Tensor

    def transport(self, mesh, depth, dt, rho0):
        """Unforced donor transport. TKE settles the split-face kinetic budget."""
        faces = WetFaces.from_columns(depth, self.h, mesh.cellsOnEdge)
        step, settlement = transport_with_budget(
            mesh, depth, self.h, self.velocity, self.cell_velocity, self.tke_h, dt, rho0
        )
        volume = step.transferred_volume
        donor = torch.where(volume >= 0, faces.left, faces.right)
        divisor = torch.where(self.h > 0, self.h, 1.0)

        def carry(content):
            q = (content / divisor).flatten()
            return content + faces.exchange(volume * q[donor]) / mesh.cell_area[:, None]

        return ShelfState(
            step.h,
            carry(self.temp_h),
            carry(self.salt_h),
            settlement.tke_h,
            carry(self.eddy_h),
            step.face.velocity,
            step.cell_velocity,
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
        ht, hs, he = advance_gm_roquet(
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
        )
        ht, hs, he = geometry(ht, hs).advance_roquet(
            depth, self.h, ht, hs, he, mesh.cell_area, energy, dt
        )
        return replace(self, temp_h=ht, salt_h=hs, eddy_h=he)

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
        )
