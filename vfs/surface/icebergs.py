"""Passive iceberg reservoir skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from vfs.field import Field
from .conservation import ConservativeTransfer
from .utils import as_cell_column, field_values


@dataclass(frozen=True)
class IcebergState:
    """Per-cell extensive iceberg mass in kg."""

    iceberg_mass: Field | torch.Tensor

    @classmethod
    def uniform(
        cls,
        n_cells: int,
        *,
        iceberg_mass: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "IcebergState":
        return cls(as_cell_column(iceberg_mass, n_cells=n_cells, dtype=dtype, device=device, name="iceberg_mass"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iceberg_mass",
            Field(
                as_cell_column(self.iceberg_mass, n_cells=int(field_values(self.iceberg_mass).numel()), name="iceberg_mass").clamp_min(0.0),
                location="cell",
                vertical_location="surface",
                name="iceberg_mass",
            ),
        )

    def to(self, *args, **kwargs) -> "IcebergState":
        return IcebergState(self.iceberg_mass.to(*args, **kwargs))

    def clone(self) -> "IcebergState":
        return IcebergState(self.iceberg_mass.clone())

    def detach(self) -> "IcebergState":
        return IcebergState(self.iceberg_mass.detach())

    def total_water(self) -> torch.Tensor:
        return self.iceberg_mass.sum()


@dataclass(frozen=True)
class IcebergStepResult:
    state: IcebergState
    melt_mass: torch.Tensor
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]


class IcebergTracer(nn.Module):
    """Passive iceberg storage with a simple melt timescale."""

    recipe_name = "iceberg_tracer"

    def __init__(self, *, melt_timescale: float = 30.0 * 86400.0) -> None:
        super().__init__()
        self.melt_timescale = float(melt_timescale)
        if self.melt_timescale <= 0.0:
            raise ValueError("melt_timescale must be positive")

    def step(
        self,
        state: IcebergState,
        dt: float,
        *,
        source_mass: torch.Tensor | float = 0.0,
        source_component: str | None = "glacier",
    ) -> IcebergStepResult:
        source = as_cell_column(source_mass, like=state.iceberg_mass, name="source_mass").clamp_min(0.0)
        dt_t = torch.as_tensor(float(dt), dtype=state.iceberg_mass.dtype, device=state.iceberg_mass.device)
        pre_melt = state.iceberg_mass + source
        melt_fraction = (dt_t / float(self.melt_timescale)).clamp(0.0, 1.0)
        melt_mass = pre_melt * melt_fraction
        next_state = IcebergState((pre_melt - melt_mass).clamp_min(0.0))
        transfers = [ConservativeTransfer("iceberg_melt", "iceberg", "ocean", water_mass=melt_mass)]
        if source_component is not None:
            transfers.insert(
                0,
                ConservativeTransfer(
                    "iceberg_source",
                    str(source_component),
                    "iceberg",
                    water_mass=source,
                ),
            )
        return IcebergStepResult(
            next_state,
            melt_mass,
            tuple(transfers),
            {
                "iceberg_source_mass": source.sum(),
                "iceberg_melt_mass": melt_mass.sum(),
                "iceberg_water_storage": next_state.total_water(),
            },
        )


register_component("iceberg_tracer", IcebergTracer)


__all__ = ["IcebergState", "IcebergStepResult", "IcebergTracer"]
