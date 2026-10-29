"""Named recipe models and grid-ingest glue for VFS-O."""

from __future__ import annotations

from typing import Any

import torch

from .model import VFSModel
from .ocean import (
    HydrostaticBoussinesqOceanDynamics,
    MultilayerOceanDynamics,
    OceanColumn,
    OceanDomain,
    OceanEkmanSurfacePhysics,
    OceanState,
    OceanSurfaceFluxPhysics,
    OceanSurfaceForcing,
)
from .ocean.helpers import _normalize_tracer_tensor
from .ocean_diagnostics import OceanDiagnostics
from .core.registry import register_model


def _external_ocean_data_payload(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("values", "data", "value"):
            if key in value:
                return value[key]
    if hasattr(value, "shape") and hasattr(value, "to"):
        return value
    for attr in ("values", "data"):
        payload = getattr(value, attr, None)
        if payload is not None and not callable(payload):
            return payload
    return value


def _coerce_ocean_cell_layer(value: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    value = value.to(dtype=like.dtype, device=like.device)
    if value.ndim == 1:
        value = value[:, None]
    expected = tuple(like.shape[-2:])
    if tuple(value.shape[-2:]) == expected:
        return value
    if tuple(value.shape[-2:]) == (expected[0], 1):
        return value.expand(value.shape[:-1] + (expected[1],))
    raise ValueError(f"ocean ingestion field {name!r} must map to trailing shape {expected}; got {tuple(value.shape)}")


def _coerce_ocean_cell_column(value: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    value = value.to(dtype=like.dtype, device=like.device)
    if value.ndim == 1:
        value = value[:, None]
    expected = like.shape[:-1] + (1,)
    if tuple(value.shape[-2:]) != tuple(expected[-2:]):
        raise ValueError(f"ocean ingestion field {name!r} must map to trailing shape {tuple(expected[-2:])}; got {tuple(value.shape)}")
    return value


def _coerce_ocean_edge_layer(value: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    value = value.to(dtype=like.dtype, device=like.device)
    if value.ndim == 1:
        value = value[:, None]
    expected = tuple(like.shape[-2:])
    if tuple(value.shape[-2:]) == expected:
        return value
    if tuple(value.shape[-2:]) == (expected[0], 1):
        return value.expand(value.shape[:-1] + (expected[1],))
    raise ValueError(f"ocean ingestion field {name!r} must map to trailing shape {expected}; got {tuple(value.shape)}")


def _coerce_ocean_tracer(value: torch.Tensor, h: torch.Tensor, *, name: str) -> torch.Tensor:
    value = value.to(dtype=h.dtype, device=h.device)
    n_cells = int(h.shape[-2])
    n_layers = int(h.shape[-1])
    if value.ndim == 1:
        value = value[:, None, None].expand(n_cells, n_layers, 1)
    elif value.ndim >= 2 and tuple(value.shape[-2:]) == (n_cells, 1):
        value = value.expand(value.shape[:-1] + (n_layers,)).unsqueeze(-1)
    return _normalize_tracer_tensor(value, int(h.shape[-2]), int(h.shape[-1]))


_OCEAN_INGEST_TARGET_ALIASES = {
    "ssh": "eta",
    "sea_surface_height": "eta",
    "surface_height": "eta",
    "layer_thickness": "h",
    "temp": "temperature",
    "theta": "temperature",
    "salt": "salinity",
    "tracer": "tracers",
    "q_tracer": "tracers",
    "edge_velocity": "u_e",
    "velocity_edge": "u_e",
}


class OceanModelBase(VFSModel):
    """Shared model contract for VFS-O named recipes."""

    def bind_state(self, state: OceanState) -> OceanState:
        mesh = self.mesh if self.mesh is not None else state.mesh
        column = self.column if self.column is not None else state.column
        if mesh is state.mesh and column is state.column:
            return state
        return OceanState(
            state.u_e,
            state.h,
            state.temp_h,
            state.salt_h,
            state.tracer_h,
            mesh=mesh,
            column=column,
            attrs=state._attrs_copy(),
        )

    def make_state(
        self,
        *,
        u_e=None,
        h=None,
        temp_h=None,
        salt_h=None,
        tracer_h=None,
        temperature=None,
        salinity=None,
        tracer=None,
        attrs: dict[str, Any] | None = None,
    ) -> OceanState:
        if self.mesh is None or self.column is None:
            raise ValueError("make_state requires a model with mesh and column")
        if u_e is None and h is None and temp_h is None and salt_h is None and temperature is None and salinity is None:
            return OceanState(mesh=self.mesh, column=self.column, attrs=attrs)
        if temperature is not None or salinity is not None:
            if temp_h is not None or salt_h is not None:
                raise ValueError("pass either temperature/salinity or temp_h/salt_h, not both")
            if h is None or u_e is None or temperature is None or salinity is None:
                raise ValueError("temperature/salinity state construction requires u_e, h, temperature, and salinity")
            return OceanState.from_temperature(
                u_e,
                h,
                temperature,
                salinity,
                tracer=tracer,
                mesh=self.mesh,
                column=self.column,
                attrs=attrs,
            )
        return OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=self.mesh, column=self.column, attrs=attrs)

    @property
    def dynamics(self):
        return self.components.get("dynamics")

    @property
    def physics(self):
        return self.components.get("physics")

    @property
    def diagnostics(self) -> OceanDiagnostics:
        if not hasattr(self, "_ocean_diagnostics"):
            self._ocean_diagnostics = OceanDiagnostics(self.dynamics, physics=self.physics, provider=self)
        return self._ocean_diagnostics

    def ingest(
        self,
        data: dict[str, Any],
        *,
        background: OceanState,
        component: str | None = None,
        field_map: dict[str, str] | None = None,
        requirement=None,
        attrs: dict[str, Any] | None = None,
    ) -> OceanState:
        """Ingest gridded ocean fields into an ``OceanState``.

        This is the explicit regrid-based implementation of the semantic
        operation. Future learned operators can keep the same public API while
        replacing the grid-to-mesh component behind it.
        """

        base = self.bind_state(background)
        base.validate()
        self.validate_data(data, component=component, requirement=requirement)
        mapping = dict(field_map or {})
        u_e = base.u_e.clone()
        h = base.h.clone()
        temperature = base.temperature().clone()
        salinity = base.salinity().clone()
        tracers = None if base.tracer_h is None else base.tracers().clone()
        temp_h_direct = None
        salt_h_direct = None
        tracer_h_direct = None

        for source_name, value in data.items():
            target_name = mapping.get(str(source_name), str(source_name))
            target_name = _OCEAN_INGEST_TARGET_ALIASES.get(str(target_name), str(target_name))
            native = self.grid_to_mesh(_external_ocean_data_payload(value), component=component)
            if target_name == "u_e":
                u_e = _coerce_ocean_edge_layer(native, u_e, name=target_name)
            elif target_name == "eta":
                eta = _coerce_ocean_cell_column(native, h, name=target_name)
                total_depth = (base.column.total_depth_like(h) + eta).clamp_min(torch.finfo(h.dtype).tiny)
                fractions = h / h.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(h.dtype).tiny)
                h = fractions * total_depth
            elif target_name == "h":
                h = _coerce_ocean_cell_layer(native, h, name=target_name)
            elif target_name == "temperature":
                temperature = _coerce_ocean_cell_layer(native, h, name=target_name)
                temp_h_direct = None
            elif target_name == "salinity":
                salinity = _coerce_ocean_cell_layer(native, h, name=target_name)
                salt_h_direct = None
            elif target_name == "temp_h":
                temp_h_direct = _coerce_ocean_cell_layer(native, h, name=target_name)
            elif target_name == "salt_h":
                salt_h_direct = _coerce_ocean_cell_layer(native, h, name=target_name)
            elif target_name == "tracers":
                tracers = _coerce_ocean_tracer(native, h, name=target_name)
                tracer_h_direct = None
            elif target_name == "tracer_h":
                tracer_h_direct = _coerce_ocean_tracer(native, h, name=target_name)
            else:
                raise ValueError(f"ocean grid ingestion cannot populate OceanState field {target_name!r}")

        temp_h = temp_h_direct if temp_h_direct is not None else h * temperature
        salt_h = salt_h_direct if salt_h_direct is not None else h * salinity
        if tracer_h_direct is not None:
            tracer_h = tracer_h_direct
        elif tracers is not None:
            tracer_h = h[..., None] * tracers
        else:
            tracer_h = None
        out_attrs = dict(base.attrs or {})
        out_attrs.update(attrs or {})
        out = OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=base.mesh, column=base.column, attrs=out_attrs)
        domain = getattr(self.dynamics, "domain", None)
        if domain is not None:
            out = domain.project_state(out)
        out.validate()
        return out


@register_model(
    "multilayer_ocean",
    status="provisional",
    component="ocean",
    description="Experimental VFS-O multilayer ocean-planet core recipe.",
    default_safe=False,
)
class MultilayerOceanModel(OceanModelBase):
    """Named VFS-O multilayer dynamics recipe."""

    recipe_name = "multilayer_ocean"

    def __init__(
        self,
        *,
        mesh,
        column: OceanColumn,
        domain: OceanDomain | None = None,
        dynamics: MultilayerOceanDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
        **dynamics_kwargs,
    ) -> None:
        all_components = {
            "dynamics": dynamics or MultilayerOceanDynamics(domain=domain, **dynamics_kwargs),
        }
        all_components.update(components or {})
        super().__init__(
            mesh=mesh,
            column=column,
            components=all_components,
            plan=[{"type": "rk", "integrator": integrator, "tendencies": ["dynamics"]}],
            observation=observation,
            ingestion=ingestion,
        )

    def tendency(self, state: OceanState) -> OceanState:
        state = self.bind_state(state)
        return self.dynamics.tendency(state)


@register_model(
    "hydrostatic_boussinesq_ocean",
    status="provisional",
    component="ocean",
    description="Hydrostatic-Boussinesq VFS-O recipe with optional diffusion/remap terms.",
    default_safe=False,
)
class HydrostaticBoussinesqOceanModel(MultilayerOceanModel):
    """Named hydrostatic-Boussinesq VFS-O dynamics recipe."""

    recipe_name = "hydrostatic_boussinesq_ocean"

    def __init__(
        self,
        *,
        mesh,
        column: OceanColumn,
        domain: OceanDomain | None = None,
        dynamics: HydrostaticBoussinesqOceanDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
        **dynamics_kwargs,
    ) -> None:
        OceanModelBase.__init__(
            self,
            mesh=mesh,
            column=column,
            components={
                "dynamics": dynamics or HydrostaticBoussinesqOceanDynamics(domain=domain, **dynamics_kwargs),
                **dict(components or {}),
            },
            plan=[{"type": "rk", "integrator": integrator, "tendencies": ["dynamics"]}],
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "forced_ocean",
    status="provisional",
    component="ocean",
    description="VFS-O dynamics plus additive ocean forcing/physics recipe.",
    default_safe=False,
)
class ForcedOceanModel(OceanModelBase):
    """Named VFS-O dynamics plus additive ocean physics recipe."""

    recipe_name = "forced_ocean"

    def __init__(
        self,
        *,
        mesh,
        column: OceanColumn,
        domain: OceanDomain | None = None,
        dynamics: HydrostaticBoussinesqOceanDynamics | None = None,
        physics: torch.nn.Module | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
        **dynamics_kwargs,
    ) -> None:
        all_components = {
            "dynamics": dynamics or HydrostaticBoussinesqOceanDynamics(domain=domain, **dynamics_kwargs),
            "physics": physics or OceanSurfaceForcing(domain=domain),
        }
        all_components.update(components or {})
        super().__init__(
            mesh=mesh,
            column=column,
            components=all_components,
            plan=[{"type": "rk", "integrator": integrator, "tendencies": ["dynamics", "physics"]}],
            observation=observation,
            ingestion=ingestion,
        )

    def tendency(self, state: OceanState) -> OceanState:
        state = self.bind_state(state)
        return self.dynamics.tendency(state) + self.physics.tendency(state)


@register_model(
    "ocean_planet_flux",
    status="experimental",
    component="ocean",
    description="Flux-forced ocean-planet research recipe using JMD95/sigma-depth defaults.",
    default_safe=False,
)
class OceanPlanetFluxModel(ForcedOceanModel):
    """Named forced VFS-O ocean-planet recipe with flux-form surface physics."""

    recipe_name = "ocean_planet_flux"

    def __init__(
        self,
        *,
        mesh,
        column: OceanColumn,
        domain: OceanDomain | None = None,
        dynamics: HydrostaticBoussinesqOceanDynamics | None = None,
        physics: torch.nn.Module | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
        eos: str = "jmd95",
        pressure_gradient_mode: str = "sigma_depth",
        **dynamics_kwargs,
    ) -> None:
        if dynamics is None:
            dynamics_kwargs = {
                "eos": eos,
                "pressure_gradient_mode": pressure_gradient_mode,
                **dynamics_kwargs,
            }
        super().__init__(
            mesh=mesh,
            column=column,
            domain=domain,
            dynamics=dynamics,
            physics=physics or OceanSurfaceFluxPhysics(domain=domain),
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
            **dynamics_kwargs,
        )


@register_model(
    "ocean_ekman_surface",
    status="experimental",
    component="ocean",
    description="Top-refined Ekman/surface-stress research recipe.",
    default_safe=False,
)
class OceanEkmanSurfaceModel(ForcedOceanModel):
    """Named VFS-O recipe for top-refined Ekman/surface-stress experiments."""

    recipe_name = "ocean_ekman_surface"

    def __init__(
        self,
        *,
        mesh,
        column: OceanColumn | None = None,
        domain: OceanDomain | None = None,
        dynamics: HydrostaticBoussinesqOceanDynamics | None = None,
        physics: OceanEkmanSurfacePhysics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
        n_layers: int = 20,
        total_depth: float = 4000.0,
        top_refinement_stretch: float = 1.35,
        eos: str = "jmd95",
        pressure_gradient_mode: str = "sigma_depth",
        stress_mode: str = "analytic",
        wind_stress_amplitude: float = 0.05,
        wind_stress_pattern: str = "zonal_sin2lat",
        prescribed_stress_edge: Any | None = None,
        air_density: float = 1.225,
        drag_coefficient: float = 1.3e-3,
        vertical_momentum_viscosity: float = 1.0e-2,
        horizontal_momentum_viscosity: float = 0.0,
        linear_drag_timescale: float = 0.0,
        bottom_drag_timescale: float = 20.0 * 86400.0,
        **dynamics_kwargs,
    ) -> None:
        if column is None:
            column = OceanColumn.top_refined(
                int(n_layers),
                total_depth=float(total_depth),
                stretch=float(top_refinement_stretch),
            )
        if dynamics is None:
            dynamics_kwargs = {
                "eos": eos,
                "pressure_gradient_mode": pressure_gradient_mode,
                **dynamics_kwargs,
            }
        super().__init__(
            mesh=mesh,
            column=column,
            domain=domain,
            dynamics=dynamics,
            physics=physics
            or OceanEkmanSurfacePhysics(
                domain=domain,
                stress_mode=stress_mode,
                wind_stress_amplitude=wind_stress_amplitude,
                wind_stress_pattern=wind_stress_pattern,
                prescribed_stress_edge=prescribed_stress_edge,
                air_density=air_density,
                drag_coefficient=drag_coefficient,
                vertical_momentum_viscosity=vertical_momentum_viscosity,
                horizontal_momentum_viscosity=horizontal_momentum_viscosity,
                linear_drag_timescale=linear_drag_timescale,
                bottom_drag_timescale=bottom_drag_timescale,
            ),
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
            **dynamics_kwargs,
        )


__all__ = [
    "ForcedOceanModel",
    "HydrostaticBoussinesqOceanModel",
    "MultilayerOceanModel",
    "OceanEkmanSurfaceModel",
    "OceanModelBase",
    "OceanPlanetFluxModel",
]
