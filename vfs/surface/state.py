"""Standalone surface-system state container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vfs.field import Field

from .fractions import SurfaceFractions
from .glacier_land_ice import GlacierState
from .icebergs import IcebergState
from ..land import LandSurfaceState
from .runoff_router import RunoffState
from .sea_ice import SeaIceState


_UNSET = object()


@dataclass(frozen=True)
class SurfaceState:
    """Component-owned surface/cryosphere state on the native cell mesh.

    This is intentionally standalone for now. Coupled atmosphere/ocean state
    remains unchanged while the surface-system contracts mature.
    """

    fractions: SurfaceFractions
    land: LandSurfaceState | None = None
    sea_ice: SeaIceState | None = None
    glacier: GlacierState | None = None
    icebergs: IcebergState | None = None
    runoff: RunoffState | None = None
    attrs: dict[str, Any] | None = None

    @property
    def n_cells(self) -> int:
        return int(self.fractions.land_fraction.shape[-2])

    def _attrs_copy(self) -> dict[str, Any]:
        return dict(self.attrs or {})

    def validate(self) -> None:
        self.fractions.validate()
        n_cells = self.n_cells
        for owner, state in (
            ("land", self.land),
            ("sea_ice", self.sea_ice),
            ("glacier", self.glacier),
            ("icebergs", self.icebergs),
            ("runoff", self.runoff),
        ):
            if state is None:
                continue
            for name, value in vars(state).items():
                if not isinstance(value, Field) and not torch.is_tensor(value):
                    continue
                if isinstance(value, Field) and value.location == "edge":
                    # SurfaceState has no mesh reference; edge cardinality is
                    # checked by the momentum closure when it receives its
                    # native coupled mesh.
                    continue
                if value.ndim == 0 or int(value.shape[0]) != n_cells:
                    got = "scalar" if value.ndim == 0 else int(value.shape[0])
                    raise ValueError(f"{owner}.{name} n_cells={got} does not match fractions {n_cells}")

    def with_components(
        self,
        *,
        fractions: SurfaceFractions | None = None,
        land: LandSurfaceState | None | object = _UNSET,
        sea_ice: SeaIceState | None | object = _UNSET,
        glacier: GlacierState | None | object = _UNSET,
        icebergs: IcebergState | None | object = _UNSET,
        runoff: RunoffState | None | object = _UNSET,
        attrs: dict[str, Any] | None = None,
    ) -> "SurfaceState":
        return SurfaceState(
            self.fractions if fractions is None else fractions,
            self.land if land is _UNSET else land,
            self.sea_ice if sea_ice is _UNSET else sea_ice,
            self.glacier if glacier is _UNSET else glacier,
            self.icebergs if icebergs is _UNSET else icebergs,
            self.runoff if runoff is _UNSET else runoff,
            self._attrs_copy() if attrs is None else attrs,
        )

    def to(self, *args, **kwargs) -> "SurfaceState":
        return SurfaceState(
            self.fractions.to(*args, **kwargs),
            None if self.land is None else self.land.to(*args, **kwargs),
            None if self.sea_ice is None else self.sea_ice.to(*args, **kwargs),
            None if self.glacier is None else self.glacier.to(*args, **kwargs),
            None if self.icebergs is None else self.icebergs.to(*args, **kwargs),
            None if self.runoff is None else self.runoff.to(*args, **kwargs),
            self._attrs_copy(),
        )

    def clone(self) -> "SurfaceState":
        return SurfaceState(
            self.fractions.clone(),
            None if self.land is None else self.land.clone(),
            None if self.sea_ice is None else self.sea_ice.clone(),
            None if self.glacier is None else self.glacier.clone(),
            None if self.icebergs is None else self.icebergs.clone(),
            None if self.runoff is None else self.runoff.clone(),
            self._attrs_copy(),
        )

    def detach(self) -> "SurfaceState":
        return SurfaceState(
            self.fractions.detach(),
            None if self.land is None else self.land.detach(),
            None if self.sea_ice is None else self.sea_ice.detach(),
            None if self.glacier is None else self.glacier.detach(),
            None if self.icebergs is None else self.icebergs.detach(),
            None if self.runoff is None else self.runoff.detach(),
            self._attrs_copy(),
        )

    def total_water(self, *, mesh: Any | None = None) -> torch.Tensor:
        total = torch.zeros((), dtype=self.fractions.land_fraction.dtype, device=self.fractions.land_fraction.device)
        if self.land is not None:
            total = total + self.land.total_water(self.fractions, mesh=mesh)
        if self.sea_ice is not None:
            total = total + self.sea_ice.total_water(mesh=mesh)
        if self.glacier is not None:
            total = total + self.glacier.total_water(mesh=mesh)
        if self.icebergs is not None:
            total = total + self.icebergs.total_water()
        if self.runoff is not None:
            total = total + self.runoff.river_storage.sum()
        return total


__all__ = ["SurfaceState"]
