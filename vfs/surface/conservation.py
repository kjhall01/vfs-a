"""Conservative transfer ledger for surface and cryosphere coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from .utils import sum_extensive


TRANSFER_QUANTITIES = ("water_mass", "salt_mass", "energy", "momentum_x", "momentum_y")


def _zero_like_transfer(transfers: Iterable["ConservativeTransfer"], quantity: str) -> torch.Tensor:
    for transfer in transfers:
        value = getattr(transfer, quantity)
        if value is not None:
            return torch.zeros((), dtype=value.dtype, device=value.device)
    return torch.zeros((), dtype=torch.float64)


@dataclass(frozen=True)
class ConservativeTransfer:
    """Per-step extensive transfer between named reservoirs.

    Positive values move from ``from_component`` to ``to_component``. The
    stored tensors are already integrated over the timestep and cell area when
    applicable, so ledger sums are conservative extensive budgets rather than
    rates.
    """

    name: str
    from_component: str
    to_component: str
    water_mass: torch.Tensor | None = None
    salt_mass: torch.Tensor | None = None
    energy: torch.Tensor | None = None
    momentum_x: torch.Tensor | None = None
    momentum_y: torch.Tensor | None = None

    def quantity(self, name: str) -> torch.Tensor | None:
        if name not in TRANSFER_QUANTITIES:
            raise ValueError(f"unknown transfer quantity {name!r}")
        return getattr(self, name)

    def summed(self, name: str) -> torch.Tensor:
        value = self.quantity(name)
        if value is None:
            return torch.zeros((), dtype=torch.float64)
        return sum_extensive(value)

    def to(self, *args, **kwargs) -> "ConservativeTransfer":
        values = {
            key: (None if getattr(self, key) is None else getattr(self, key).to(*args, **kwargs))
            for key in TRANSFER_QUANTITIES
        }
        return ConservativeTransfer(self.name, self.from_component, self.to_component, **values)


@dataclass
class TransferLedger:
    """A small signed budget accumulator for conservative transfers."""

    transfers: list[ConservativeTransfer] = field(default_factory=list)

    def add(self, transfer: ConservativeTransfer | None) -> None:
        if transfer is not None:
            self.transfers.append(transfer)

    def extend(self, transfers: Iterable[ConservativeTransfer]) -> None:
        for transfer in transfers:
            self.add(transfer)

    def component_budget(self, quantity: str) -> dict[str, torch.Tensor]:
        zero = _zero_like_transfer(self.transfers, quantity)
        budgets: dict[str, torch.Tensor] = {}
        for transfer in self.transfers:
            value = transfer.quantity(quantity)
            if value is None:
                continue
            total = sum_extensive(value)
            budgets.setdefault(transfer.from_component, zero.clone())
            budgets.setdefault(transfer.to_component, zero.clone())
            budgets[transfer.from_component] = budgets[transfer.from_component] - total
            budgets[transfer.to_component] = budgets[transfer.to_component] + total
        return budgets

    def global_residual(self, quantity: str) -> torch.Tensor:
        budgets = self.component_budget(quantity)
        if not budgets:
            return _zero_like_transfer(self.transfers, quantity)
        total = None
        for value in budgets.values():
            total = value if total is None else total + value
        return total if total is not None else _zero_like_transfer(self.transfers, quantity)

    def storage_residual(
        self,
        *,
        quantity: str,
        component: str,
        initial_storage: torch.Tensor,
        final_storage: torch.Tensor,
        boundary_flux: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        budgets = self.component_budget(quantity)
        transfer_budget = budgets.get(str(component))
        if transfer_budget is None:
            transfer_budget = torch.zeros((), dtype=final_storage.dtype, device=final_storage.device)
        boundary = (
            boundary_flux.to(dtype=final_storage.dtype, device=final_storage.device)
            if torch.is_tensor(boundary_flux)
            else torch.as_tensor(boundary_flux, dtype=final_storage.dtype, device=final_storage.device)
        )
        return sum_extensive(final_storage) - sum_extensive(initial_storage) - transfer_budget - sum_extensive(boundary)

    def summary(self) -> dict[str, Any]:
        rows: dict[str, Any] = {"n_transfers": len(self.transfers)}
        for quantity in TRANSFER_QUANTITIES:
            residual = self.global_residual(quantity)
            rows[f"{quantity}_global_residual"] = residual
            budgets = self.component_budget(quantity)
            for component, value in budgets.items():
                rows[f"{component}_{quantity}_budget"] = value
        return rows


def ledger_from(transfers: Iterable[ConservativeTransfer]) -> TransferLedger:
    ledger = TransferLedger()
    ledger.extend(transfers)
    return ledger


def relative_residual(residual: torch.Tensor, scale: torch.Tensor | float, *, tiny: float | None = None) -> torch.Tensor:
    if not torch.is_tensor(scale):
        scale = torch.as_tensor(scale, dtype=residual.dtype, device=residual.device)
    if tiny is None:
        tiny = torch.finfo(residual.dtype).tiny if residual.is_floating_point() else 1.0e-30
    return residual.abs() / scale.abs().clamp_min(float(tiny))


def storage_budget_summary(
    *,
    quantity: str,
    component: str,
    initial_storage: torch.Tensor,
    final_storage: torch.Tensor,
    transfers: Iterable[ConservativeTransfer],
    boundary_flux: torch.Tensor | float = 0.0,
) -> Mapping[str, torch.Tensor]:
    ledger = ledger_from(transfers)
    residual = ledger.storage_residual(
        quantity=quantity,
        component=component,
        initial_storage=initial_storage,
        final_storage=final_storage,
        boundary_flux=boundary_flux,
    )
    scale = sum_extensive(initial_storage).abs() + sum_extensive(final_storage).abs() + torch.as_tensor(
        1.0,
        dtype=final_storage.dtype,
        device=final_storage.device,
    )
    return {
        "initial_storage": sum_extensive(initial_storage),
        "final_storage": sum_extensive(final_storage),
        "residual": residual,
        "relative_residual": relative_residual(residual, scale),
    }


__all__ = [
    "ConservativeTransfer",
    "TRANSFER_QUANTITIES",
    "TransferLedger",
    "ledger_from",
    "relative_residual",
    "storage_budget_summary",
]
