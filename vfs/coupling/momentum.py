"""Momentum exchange operators for coupled atmosphere/ocean runs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..core.state import DEFAULT_GRAVITY, State


class AirSeaMomentumCoupler(nn.Module):
    """Edge-normal bulk wind-stress coupling between atmosphere and ocean states.

    The coupler does not own either model. It computes the relative wind stress
    from an atmospheric edge wind and the ocean top-layer edge velocity, can
    attach the wind/stress payload expected by ``OceanEkmanSurfacePhysics``,
    and can return an optional atmosphere-side bottom-level velocity tendency.

    The first implementation assumes a shared native edge mesh when extracting
    winds directly from an atmospheric ``State``. Future regridding or learned
    coupling operators should feed ``atmospheric_wind_edge=...`` explicitly.
    """

    recipe_name = "air_sea_momentum_coupler"

    def __init__(
        self,
        *,
        air_density: float = 1.225,
        drag_coefficient: float = 1.3e-3,
        atmosphere_level: int = -1,
    ) -> None:
        super().__init__()
        self.air_density = float(air_density)
        self.drag_coefficient = float(drag_coefficient)
        self.atmosphere_level = int(atmosphere_level)
        if self.air_density < 0.0:
            raise ValueError("air_density must be nonnegative")
        if self.drag_coefficient < 0.0:
            raise ValueError("drag_coefficient must be nonnegative")

    def _level_slice(self, n_levels: int) -> slice:
        level = self.atmosphere_level
        if level < 0:
            level = int(n_levels) + level
        if level < 0 or level >= int(n_levels):
            raise ValueError(
                f"atmosphere_level {self.atmosphere_level} is out of range for {int(n_levels)} levels"
            )
        return slice(level, level + 1)

    @staticmethod
    def _same_native_mesh(atmosphere_state: State, ocean_state: Any) -> None:
        if atmosphere_state.mesh is None or ocean_state.mesh is None:
            raise ValueError("AirSeaMomentumCoupler requires mesh-bound atmosphere and ocean states")
        if atmosphere_state.mesh is not ocean_state.mesh:
            raise ValueError(
                "AirSeaMomentumCoupler can extract atmospheric winds only when "
                "atmosphere and ocean share the same native mesh; pass "
                "atmospheric_wind_edge=... after regridding otherwise"
            )

    @staticmethod
    def _tensor_like(value: Any, like: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=like.dtype, device=like.device)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    @classmethod
    def _edge_column_field(cls, value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = cls._tensor_like(value, like)
        if tensor.ndim == 0:
            return torch.ones_like(like) * tensor
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.shape == like.shape:
            return tensor
        if (
            tensor.ndim == 2
            and int(tensor.shape[0]) == int(like.shape[-2])
            and int(tensor.shape[1]) == 1
        ):
            return tensor
        raise ValueError(f"{name} must be scalar, [nEdges], or [nEdges, 1]; got {tuple(tensor.shape)}")

    def atmospheric_wind_edge(
        self,
        ocean_state: Any,
        atmosphere_state: State | None = None,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
    ) -> torch.Tensor:
        """Return atmospheric edge-normal wind at the ocean surface."""

        like = ocean_state.u_e[..., :1]
        supplied = sum(
            value is not None
            for value in (atmosphere_state, atmospheric_wind_edge, atmospheric_wind_xyz_edge)
        )
        if supplied != 1:
            raise ValueError(
                "pass exactly one of atmosphere_state, atmospheric_wind_edge, or atmospheric_wind_xyz_edge"
            )
        if atmospheric_wind_edge is not None:
            return self._edge_column_field(atmospheric_wind_edge, like, name="atmospheric_wind_edge")
        if atmospheric_wind_xyz_edge is not None:
            wind_xyz = self._tensor_like(atmospheric_wind_xyz_edge, like)
            if (
                wind_xyz.ndim < 2
                or int(wind_xyz.shape[-2]) != int(ocean_state.mesh.nEdges)
                or int(wind_xyz.shape[-1]) != 3
            ):
                raise ValueError(
                    "atmospheric_wind_xyz_edge must have trailing shape "
                    f"({int(ocean_state.mesh.nEdges)}, 3); got {tuple(wind_xyz.shape)}"
                )
            normal = ocean_state.mesh.edge_normal.to(dtype=like.dtype, device=like.device)
            projected = (wind_xyz * normal).sum(dim=-1, keepdim=True)
            return self._edge_column_field(projected, like, name="atmospheric_wind_xyz_edge")
        assert atmosphere_state is not None
        self._same_native_mesh(atmosphere_state, ocean_state)
        level = self._level_slice(int(atmosphere_state.u_e.shape[-1]))
        return self._edge_column_field(atmosphere_state.u_e[..., level], like, name="atmosphere_state.u_e")

    def relative_wind_edge(
        self,
        atmosphere_state: State | None,
        ocean_state: Any,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
    ) -> torch.Tensor:
        """Return edge-normal air-minus-ocean relative wind."""

        wind = self.atmospheric_wind_edge(
            ocean_state,
            atmosphere_state,
            atmospheric_wind_edge=atmospheric_wind_edge,
            atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
        )
        return wind - ocean_state.u_e[..., :1]

    def stress_from_relative_wind(self, relative_wind_edge: torch.Tensor) -> torch.Tensor:
        """Return edge-normal stress on the ocean in N m-2."""

        return (
            float(self.air_density)
            * float(self.drag_coefficient)
            * relative_wind_edge.abs()
            * relative_wind_edge
        )

    def surface_stress_edge(
        self,
        atmosphere_state: State | None,
        ocean_state: Any,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
    ) -> torch.Tensor:
        """Return bulk edge-normal wind stress applied to the ocean."""

        rel = self.relative_wind_edge(
            atmosphere_state,
            ocean_state,
            atmospheric_wind_edge=atmospheric_wind_edge,
            atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
        )
        return self.stress_from_relative_wind(rel)

    def stress_pair(
        self,
        atmosphere_state: State | None,
        ocean_state: Any,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return equal-and-opposite edge stresses for ocean and atmosphere."""

        tau_ocean = self.surface_stress_edge(
            atmosphere_state,
            ocean_state,
            atmospheric_wind_edge=atmospheric_wind_edge,
            atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
        )
        return {
            "stress_on_ocean_edge": tau_ocean,
            "stress_on_atmosphere_edge": -tau_ocean,
        }

    def prepare_ocean_state(
        self,
        ocean_state: Any,
        atmosphere_state: State | None = None,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
        include_stress: bool = True,
    ) -> Any:
        """Return an OceanState carrying the attrs consumed by ocean physics."""

        wind = self.atmospheric_wind_edge(
            ocean_state,
            atmosphere_state,
            atmospheric_wind_edge=atmospheric_wind_edge,
            atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
        )
        out = ocean_state.with_tensors(
            ocean_state.u_e,
            ocean_state.h,
            ocean_state.temp_h,
            ocean_state.salt_h,
            ocean_state.tracer_h,
        )
        out.attrs["atmospheric_wind_edge"] = wind
        if include_stress:
            out.attrs["surface_stress_edge"] = self.stress_from_relative_wind(wind - ocean_state.u_e[..., :1])
        return out

    def coupled_momentum_tendencies(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        ocean_physics: Any,
        *,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
        edge_mask: Any | None = None,
        include_audit: bool = True,
    ) -> dict[str, Any]:
        """Return explicitly coupled atmosphere/ocean surface-momentum tendencies.

        ``ocean_physics`` is expected to consume the prepared ocean state, as
        ``OceanEkmanSurfacePhysics(stress_mode="relative")`` does. If it
        exposes ``wind_stress_edge(state)``, that exact stress is used for the
        atmosphere-side feedback so ocean masks or stress-provider semantics
        stay consistent.
        """

        prepared_ocean = self.prepare_ocean_state(
            ocean_state,
            atmosphere_state,
            atmospheric_wind_edge=atmospheric_wind_edge,
            atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
        )
        stress_provider = getattr(ocean_physics, "wind_stress_edge", None)
        if callable(stress_provider):
            tau_ocean = stress_provider(prepared_ocean)
        else:
            tau_ocean = prepared_ocean.attrs.get("surface_stress_edge")
            if tau_ocean is None:
                tau_ocean = self.surface_stress_edge(
                    atmosphere_state,
                    ocean_state,
                    atmospheric_wind_edge=atmospheric_wind_edge,
                    atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
                )
        tau_ocean = self._edge_column_field(tau_ocean, ocean_state.u_e[..., :1], name="stress_on_ocean_edge")
        if edge_mask is not None:
            mask = self._edge_column_field(edge_mask, tau_ocean, name="momentum_edge_mask").clamp(0.0, 1.0)
            tau_ocean = tau_ocean * mask
        # Supplying the masked stress directly is essential: masking an already
        # computed ocean tendency would also scale unrelated ocean terms. The
        # standard VFS Ekman block exposes this explicit entry point.
        tendency_with_stress = getattr(ocean_physics, "tendency_with_surface_stress", None)
        if edge_mask is None:
            # Preserve the established unmasked owner contract exactly.
            ocean_rhs = ocean_physics.tendency(prepared_ocean)
        else:
            if not callable(tendency_with_stress):
                raise TypeError(
                    "masked coupled momentum requires ocean physics with "
                    "tendency_with_surface_stress(state, stress)"
                )
            ocean_rhs = tendency_with_stress(prepared_ocean, tau_ocean)
        atmosphere_rhs = self.atmospheric_momentum_tendency(
            atmosphere_state,
            stress_on_ocean_edge=tau_ocean,
        )
        out: dict[str, Any] = {
            "ocean_state": prepared_ocean,
            "ocean_rhs": ocean_rhs,
            "atmosphere_rhs": atmosphere_rhs,
            "stress_on_ocean_edge": tau_ocean,
        }
        if include_audit:
            terms = self.momentum_impulse_terms(
                atmosphere_state,
                ocean_state,
                ocean_rhs=ocean_rhs,
                atmosphere_rhs=atmosphere_rhs,
                stress_on_ocean_edge=tau_ocean,
                ocean_density=getattr(ocean_physics, "rho0", None),
            )
            out["impulse_terms"] = terms
            out["impulse_summary"] = self.momentum_impulse_summary(terms=terms)
        return out

    def momentum_impulse_terms(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        *,
        ocean_rhs: Any,
        atmosphere_rhs: State,
        stress_on_ocean_edge: Any | None = None,
        ocean_density: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return tensor impulse terms for a coupled surface-momentum exchange.

        The ocean term is column-integrated over all ocean layers, so internal
        vertical stress redistribution cancels in the audit. The atmosphere
        term is the pressure-coordinate bottom-layer impulse per unit area.
        """

        level = self._level_slice(int(atmosphere_state.u_e.shape[-1]))
        tau = (
            self.surface_stress_edge(atmosphere_state, ocean_state)
            if stress_on_ocean_edge is None
            else self._edge_column_field(stress_on_ocean_edge, ocean_state.u_e[..., :1], name="stress_on_ocean_edge")
        )
        h_edge = ocean_state.mesh.cell_to_edge(ocean_state.h).to(dtype=ocean_rhs.u_e.dtype, device=ocean_rhs.u_e.device)
        rho0_value = ocean_density
        if rho0_value is None:
            rho0_value = getattr(getattr(ocean_state, "column", None), "rho0", 1025.0)
        rho0 = self._tensor_like(rho0_value, ocean_rhs.u_e)
        ocean_impulse = rho0 * (h_edge * ocean_rhs.u_e).sum(dim=-1, keepdim=True)

        dp_edge = atmosphere_state.mesh.cell_to_edge(atmosphere_state.layer_dp())[..., level].to(
            dtype=atmosphere_rhs.u_e.dtype,
            device=atmosphere_rhs.u_e.device,
        )
        gravity = float((atmosphere_state.attrs or {}).get("gravity", DEFAULT_GRAVITY))
        atmosphere_impulse = dp_edge * atmosphere_rhs.u_e[..., level] / gravity
        ocean_impulse = ocean_impulse.to(dtype=tau.dtype, device=tau.device)
        atmosphere_impulse = atmosphere_impulse.to(dtype=tau.dtype, device=tau.device)
        residual = ocean_impulse + atmosphere_impulse
        return {
            "stress_on_ocean_edge": tau,
            "ocean_column_impulse": ocean_impulse,
            "atmosphere_bottom_impulse": atmosphere_impulse,
            "coupled_impulse_residual": residual,
        }

    def momentum_impulse_summary(
        self,
        atmosphere_state: State | None = None,
        ocean_state: Any | None = None,
        *,
        terms: dict[str, torch.Tensor] | None = None,
        ocean_rhs: Any | None = None,
        atmosphere_rhs: State | None = None,
        stress_on_ocean_edge: Any | None = None,
        ocean_density: float | torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Return detached scalar summary metrics for the impulse audit."""

        if terms is None:
            if atmosphere_state is None or ocean_state is None or ocean_rhs is None or atmosphere_rhs is None:
                raise ValueError(
                    "pass terms=... or pass atmosphere_state, ocean_state, ocean_rhs, and atmosphere_rhs"
                )
            terms = self.momentum_impulse_terms(
                atmosphere_state,
                ocean_state,
                ocean_rhs=ocean_rhs,
                atmosphere_rhs=atmosphere_rhs,
                stress_on_ocean_edge=stress_on_ocean_edge,
                ocean_density=ocean_density,
            )

        def rms(value: torch.Tensor) -> float:
            return float(value.detach().square().mean().sqrt().cpu().item())

        def max_abs(value: torch.Tensor) -> float:
            return float(value.detach().abs().max().cpu().item())

        stress = terms["stress_on_ocean_edge"]
        residual = terms["coupled_impulse_residual"]
        stress_rms = rms(stress)
        return {
            "stress_rms": stress_rms,
            "stress_max_abs": max_abs(stress),
            "ocean_column_impulse_rms": rms(terms["ocean_column_impulse"]),
            "atmosphere_bottom_impulse_rms": rms(terms["atmosphere_bottom_impulse"]),
            "coupled_impulse_residual_rms": rms(residual),
            "coupled_impulse_residual_max_abs": max_abs(residual),
            "coupled_impulse_relative_residual": max_abs(residual) / max(stress_rms, 1.0e-300),
        }

    def atmospheric_momentum_tendency(
        self,
        atmosphere_state: State,
        ocean_state: Any | None = None,
        *,
        stress_on_ocean_edge: Any | None = None,
        atmospheric_wind_edge: Any | None = None,
        atmospheric_wind_xyz_edge: Any | None = None,
    ) -> State:
        """Return a State-shaped bottom-level velocity tendency for feedback.

        ``stress_on_ocean_edge`` is positive in the direction of ocean
        acceleration. The atmosphere receives the equal-and-opposite stress,
        divided by the pressure-coordinate bottom-layer mass per unit area.
        """

        level = self._level_slice(int(atmosphere_state.u_e.shape[-1]))
        like = atmosphere_state.u_e[..., level]
        if atmosphere_state.mesh is None or atmosphere_state.column is None:
            raise ValueError("atmospheric momentum feedback requires a mesh-bound and column-bound State")
        if stress_on_ocean_edge is None:
            if ocean_state is None:
                raise ValueError("ocean_state is required when stress_on_ocean_edge is not supplied")
            tau = self.surface_stress_edge(
                atmosphere_state,
                ocean_state,
                atmospheric_wind_edge=atmospheric_wind_edge,
                atmospheric_wind_xyz_edge=atmospheric_wind_xyz_edge,
            )
        else:
            tau = self._edge_column_field(stress_on_ocean_edge, like, name="stress_on_ocean_edge")
        dp_edge = atmosphere_state.mesh.cell_to_edge(atmosphere_state.layer_dp())[..., level].to(
            dtype=like.dtype,
            device=like.device,
        )
        gravity = float((atmosphere_state.attrs or {}).get("gravity", DEFAULT_GRAVITY))
        du = -gravity * tau / dp_edge.clamp_min(torch.finfo(dp_edge.dtype).tiny)
        u_t = torch.zeros_like(atmosphere_state.u_e)
        u_t[..., level] = du
        tracer_t = None if atmosphere_state.tracer_m is None else torch.zeros_like(atmosphere_state.tracer_m)
        return State(
            u_t,
            torch.zeros_like(atmosphere_state.mtheta_i),
            torch.zeros_like(atmosphere_state.ps),
            tracer_t,
            mesh=atmosphere_state.mesh,
            column=atmosphere_state.column,
            attrs=dict(atmosphere_state.attrs or {}),
        )


register_component("air_sea_momentum_coupler", AirSeaMomentumCoupler)


__all__ = ["AirSeaMomentumCoupler"]
