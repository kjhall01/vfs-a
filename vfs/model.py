"""Explicit composition of dynamics, physics, and integrators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .core.config import DataRequirement
from .atmosphere import (
    DryHydrostaticDynamics,
    DryHydrostaticLogPressureSigmaDynamics,
    DryHydrostaticUnscaledCoordWorkDynamics,
    DryHydrostaticVirtualTemperatureDynamics,
)
from .core.integrators import forward_euler, rk4, ssprk3
from .physics import HeldSuarezConstants, HeldSuarezPhysics, SCMPhysics, VFSNativeSCM
from .process import PhysicsState, PhysicsStepResult, initial_physics_state_for_components, step_process_components
from .geometry.regrid import LatLonGrid, LatLonToMeshBilinear, MeshToLatLonKNN
from .core.registry import get_component, get_integrator, get_model, get_regrid_operator, register_component, register_integrator, register_model
from .core.state import PrognosticFields, State


_DEFAULT_COMPONENTS = {
    "held_suarez": HeldSuarezPhysics,
    "scm_physics": SCMPhysics,
    "vfs_native_scm": VFSNativeSCM,
}

_DEFAULT_INTEGRATORS = {
    "forward_euler": forward_euler,
    "rk4": rk4,
    "ssprk3": ssprk3,
}

def _spec_kwargs(spec: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    kwargs = dict(spec.get("kwargs", {}))
    for key in keys:
        if key in spec:
            kwargs[key] = spec[key]
    return kwargs


def _typed_to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def _spec_dict(spec: Any) -> dict[str, Any]:
    spec = _typed_to_dict(spec)
    if isinstance(spec, str):
        return {"type": spec}
    return dict(spec or {})


def _attach_component_requirements(component: Any, spec: dict[str, Any]) -> Any:
    if "data_requirement" in spec and spec["data_requirement"] is not None:
        component.data_requirement = DataRequirement.merge(spec["data_requirement"])
    return component


def _latlon_grid_from_spec(spec: dict[str, Any]) -> LatLonGrid:
    grid_spec = dict(spec.get("grid", {}))
    nlat = int(spec.get("nlat", grid_spec.get("nlat", 181)))
    nlon = int(spec.get("nlon", grid_spec.get("nlon", 360)))
    kwargs = {
        "lat_min": float(spec.get("lat_min", grid_spec.get("lat_min", -90.0))),
        "lat_max": float(spec.get("lat_max", grid_spec.get("lat_max", 90.0))),
        "lon_min": float(spec.get("lon_min", grid_spec.get("lon_min", 0.0))),
        "lon_max": float(spec.get("lon_max", grid_spec.get("lon_max", 360.0))),
    }
    return LatLonGrid.regular(nlat, nlon, **kwargs)


def _build_regrid_component(type_name: str, spec: dict[str, Any], *, mesh):
    params = _spec_kwargs(
        spec,
        (
            "nlat",
            "nlon",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "k",
            "power",
            "eps",
            "method",
        ),
    )
    merged = {**spec, **params}
    if type_name == "mesh_to_latlon_knn":
        if mesh is None:
            raise ValueError("mesh_to_latlon_knn component requires model mesh")
        grid = _latlon_grid_from_spec(merged)
        return _attach_component_requirements(MeshToLatLonKNN.from_mesh(
            mesh,
            grid,
            k=int(merged.get("k", 8)),
            power=float(merged.get("power", 2.0)),
            eps=float(merged.get("eps", 1.0e-12)),
        ), spec)
    if type_name == "latlon_to_mesh_bilinear":
        if mesh is None:
            raise ValueError("latlon_to_mesh_bilinear component requires model mesh")
        grid = _latlon_grid_from_spec(merged)
        return _attach_component_requirements(LatLonToMeshBilinear.from_mesh(grid, mesh), spec)
    constructor = get_regrid_operator(type_name)
    if constructor is None:
        raise ValueError(f"unknown regrid component type {type_name!r}")
    kwargs = dict(spec.get("kwargs", {}))
    return _attach_component_requirements(constructor(**kwargs), spec)


def _build_component(spec: Any, *, name: str, mesh, column):
    del column
    if isinstance(spec, nn.Module) or callable(spec) or any(
        hasattr(spec, attr) for attr in ("tendency", "step", "state_to_latlon")
    ):
        return spec
    spec = _spec_dict(spec)
    type_name = str(spec.get("type", name))
    if type_name in ("mesh_to_latlon_knn", "latlon_to_mesh_bilinear") or get_regrid_operator(type_name) is not None:
        return _build_regrid_component(type_name, spec, mesh=mesh)
    constructor = get_component(type_name) or _DEFAULT_COMPONENTS.get(type_name)
    if constructor is None:
        raise ValueError(f"unknown component type {type_name!r}")
    kwargs = dict(spec.get("kwargs", {}))
    return _attach_component_requirements(constructor(**kwargs), spec)


def _has_tendency(component: Any) -> bool:
    return hasattr(component, "tendency")


def _is_observation_component(component: Any) -> bool:
    return hasattr(component, "state_to_latlon")


def _is_grid_to_mesh_component(component: Any) -> bool:
    return isinstance(component, LatLonToMeshBilinear) or (
        callable(component) and not _has_tendency(component) and not _is_observation_component(component)
    )


def _state_requirement_for_component(component: Any):
    method = getattr(component, "required_state_requirement", None)
    if callable(method):
        return method()
    return getattr(component, "state_requirement", None)


def _data_requirement_for_component(component: Any):
    method = getattr(component, "required_data_requirement", None)
    if callable(method):
        return method()
    return getattr(component, "data_requirement", None)


def _data_context_for_component(component: Any, *, mesh=None, column=None) -> dict[str, int]:
    context: dict[str, int] = {}
    for name in ("nlat", "nlon", "nCells", "nEdges", "nVertices"):
        if hasattr(component, name):
            context[name] = int(getattr(component, name))
    grid = getattr(component, "grid", None)
    if grid is not None:
        if hasattr(grid, "nlat"):
            context["nlat"] = int(grid.nlat)
        if hasattr(grid, "nlon"):
            context["nlon"] = int(grid.nlon)
    if mesh is not None:
        context.setdefault("nCells", int(mesh.nCells))
        context.setdefault("nEdges", int(mesh.nEdges))
        context.setdefault("nVertices", int(mesh.nVertices))
    if column is not None:
        if hasattr(column, "n_levels"):
            context["nLevels"] = int(column.n_levels)
            context["nInterfaces"] = int(column.n_levels) + 1
        if hasattr(column, "n_layers"):
            context["nLayers"] = int(column.n_layers)
            context["nInterfaces"] = int(column.n_layers) + 1
    return context


def _external_data_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("values", "data", "value"):
            if key in value:
                return value[key]
    if hasattr(value, "shape") and hasattr(value, "to") and hasattr(value, "detach"):
        return value
    for attr in ("values", "data"):
        payload = getattr(value, attr, None)
        if payload is not None and not callable(payload):
            return payload
    return value


def _plan_component_names(item: Mapping[str, Any], *, keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        if key in item:
            value = item[key]
            if isinstance(value, str):
                return [value]
            return [str(name) for name in value]
    return []


class VFSModel(nn.Module):
    """Small static execution plan for additive tendencies and split processes."""

    def __init__(
        self,
        *,
        components: dict[str, Any],
        plan: list[dict[str, Any]],
        mesh: Any | None = None,
        column: Any | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.column = column
        self.components: dict[str, Any] = dict(components)
        self.observation_component = None if observation is None else str(observation)
        self.ingestion_component = None if ingestion is None else str(ingestion)
        self.module_components = nn.ModuleDict(
            {name: comp for name, comp in self.components.items() if isinstance(comp, nn.Module)}
        )
        self.plan = []
        for item in plan:
            item = _spec_dict(item)
            plan_type = str(item.get("type", "rk"))
            if plan_type == "rk":
                integrator_name = str(item.get("integrator", "ssprk3"))
                integrator = get_integrator(integrator_name) or _DEFAULT_INTEGRATORS.get(integrator_name)
                if integrator is None:
                    raise ValueError(f"unknown integrator {integrator_name!r}")
                provider_names = _plan_component_names(item, keys=("tendencies", "providers"))
                if not provider_names:
                    raise ValueError("rk plan item requires at least one tendency provider")
                providers = [self.components[name] for name in provider_names]
                self.plan.append(
                    {"type": "rk", "integrator": integrator, "providers": providers, "names": provider_names}
                )
            elif plan_type == "process":
                process_names = _plan_component_names(item, keys=("processes", "components", "component"))
                if not process_names:
                    raise ValueError("process plan item requires at least one process component")
                processes = {name: self.components[name] for name in process_names}
                substeps = max(1, int(item.get("substeps", item.get("process_substeps", 1))))
                self.plan.append(
                    {"type": "process", "processes": processes, "substeps": substeps, "names": process_names}
                )
            else:
                raise ValueError("VFSModel plan items must have type 'rk' or 'process'")

    def step(self, state, dt: float):
        return self.step_with_physics_state(state, dt).state

    def initial_physics_state(self, state) -> PhysicsState:
        """Initialize process-owned restart state for process plan items."""

        bound = self.bind_state(state)
        processes: dict[str, Any] = {}
        for item in self.plan:
            if item["type"] == "process":
                processes.update(item["processes"])
        return initial_physics_state_for_components(processes, bound)

    def step_with_physics_state(
        self,
        state,
        dt: float,
        physics_state: PhysicsState | Mapping[str, Any] | None = None,
    ) -> PhysicsStepResult:
        """Advance one model step and return both prognostic and process state."""

        current = self.bind_state(state)
        if physics_state is None:
            root = self.initial_physics_state(current)
        elif isinstance(physics_state, PhysicsState):
            root = physics_state
        elif isinstance(physics_state, Mapping):
            root = PhysicsState(dict(physics_state))
        else:
            raise TypeError("physics_state must be PhysicsState, a mapping, or None")
        diagnostics: dict[str, Any] = {}
        for item in self.plan:
            if item["type"] == "rk":
                providers = item["providers"]
                current = item["integrator"](providers if len(providers) > 1 else providers[0], current, dt)
            elif item["type"] == "process":
                result = step_process_components(
                    item["processes"],
                    current,
                    dt,
                    physics_state=root,
                    substeps=item["substeps"],
                )
                current = result.state
                root = result.physics_state
                if result.diagnostics:
                    diagnostics.update(result.diagnostics)
        return PhysicsStepResult(current, root, diagnostics)

    def forward(self, state, dt: float):
        return self.step(state, dt)

    def bind_state(self, state: State) -> State:
        """Bind a State to model-owned mesh/column when available."""

        mesh = self.mesh if self.mesh is not None else state.mesh
        column = self.column if self.column is not None else state.column
        if mesh is state.mesh and column is state.column:
            return state
        return State(fields=state.fields, mesh=mesh, column=column, attrs=state._attrs_copy())

    def make_state(
        self,
        fields: PrognosticFields | None = None,
        *,
        u_e=None,
        mtheta_i=None,
        ps=None,
        tracer_m=None,
        attrs: dict[str, Any] | None = None,
    ) -> State:
        """Build a State bound to this model's mesh and column."""

        if self.mesh is None or self.column is None:
            raise ValueError("make_state requires a model with mesh and column")
        if fields is None:
            fields = PrognosticFields(u_e, mtheta_i, ps, tracer_m)
        return State(fields=fields, mesh=self.mesh, column=self.column, attrs=attrs)

    def move_state(self, state: State, *args, **kwargs) -> State:
        """Move State fields and bind them to the model-owned mesh/column."""

        return self.bind_state(state).to(*args, **kwargs)

    def validate_state(self, state: State, *, components: tuple[str, ...] | list[str] | None = None) -> State:
        """Validate a State against model geometry and component requirements."""

        bound = self.bind_state(state)
        names = [str(name) for name in components] if components is not None else list(self.components)
        for name in names:
            if name not in self.components:
                raise KeyError(f"unknown component {name!r}; available={sorted(self.components)}")
            component = self.components[name]
            requirement = _state_requirement_for_component(component)
            if requirement is not None:
                requirement.validate(bound, owner=f"component {name!r}")
            checker = getattr(component, "check_state_compatible", None)
            if callable(checker):
                checker(bound)
        return bound

    def _select_component(self, name: str | None, predicate, role: str):
        if name is not None:
            key = str(name)
            if key not in self.components:
                raise KeyError(f"unknown {role} component {key!r}; available={sorted(self.components)}")
            component = self.components[key]
            if not predicate(component):
                raise TypeError(f"component {key!r} cannot be used for {role}")
            return component
        candidates = [(key, component) for key, component in self.components.items() if predicate(component)]
        if len(candidates) == 1:
            return candidates[0][1]
        if not candidates:
            raise KeyError(f"model has no {role} component")
        names = [key for key, _component in candidates]
        raise KeyError(f"model has multiple {role} components {names}; set the model role or pass component=...")

    def observe(self, state: State, *, component: str | None = None, fields="default") -> dict[str, Any]:
        """Project a State into the model's observation/grid space."""

        name = self.observation_component if component is None else str(component)
        observer = self._select_component(name, _is_observation_component, "observation")
        return self.bind_state(state).to_latlon(observer, fields=fields)

    def grid_to_mesh(self, field, *, component: str | None = None):
        """Map a regular-grid field through the model's grid-to-mesh component."""

        name = self.ingestion_component if component is None else str(component)
        operator = self._select_component(name, _is_grid_to_mesh_component, "grid-to-mesh")
        return operator(field)

    def validate_data(
        self,
        data: Mapping[str, Any],
        *,
        component: str | None = None,
        requirement: DataRequirement | dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Validate external data against a component-owned data requirement."""

        name = self.ingestion_component if component is None else str(component)
        operator = self._select_component(name, _is_grid_to_mesh_component, "grid-to-mesh")
        component_requirement = _data_requirement_for_component(operator)
        merged = DataRequirement.merge(component_requirement, requirement)
        if merged.fields:
            merged.validate(
                data,
                owner=f"component {name!r}" if name is not None else "grid-to-mesh component",
                context=_data_context_for_component(operator, mesh=self.mesh, column=self.column),
            )
        return data

    def ingest(
        self,
        data: Mapping[str, Any],
        *,
        background: State,
        component: str | None = None,
        field_map: dict[str, str] | None = None,
        requirement: DataRequirement | dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> State:
        """Ingest gridded prognostic fields into a State using a model component.

        This is intentionally narrow: it updates State prognostic fields from a
        background. Future learned/assimilation components can replace the
        internal grid-to-mesh step while preserving the semantic API.
        """

        base = self.bind_state(background)
        fields = base._require_fields().clone()
        self.validate_data(data, component=component, requirement=requirement)
        mapping = dict(field_map or {})
        for source_name, value in data.items():
            target_name = mapping.get(str(source_name), str(source_name))
            native = self.grid_to_mesh(_external_data_payload(value), component=component)
            if target_name == "ps":
                if native.ndim == 1:
                    native = native[:, None]
                fields.ps = native.to(dtype=fields.ps.dtype, device=fields.ps.device)
            elif target_name == "mtheta_i":
                fields.mtheta_i = native.to(dtype=fields.mtheta_i.dtype, device=fields.mtheta_i.device)
            elif target_name == "tracer_m":
                if fields.tracer_m is None:
                    fields.tracer_m = native.to(dtype=fields.ps.dtype, device=fields.ps.device)
                else:
                    fields.tracer_m = native.to(dtype=fields.tracer_m.dtype, device=fields.tracer_m.device)
            else:
                raise ValueError(f"grid ingestion cannot populate State field {target_name!r}")
        out_attrs = dict(base.attrs or {})
        out_attrs.update(attrs or {})
        return State(fields=fields, mesh=base.mesh, column=base.column, attrs=out_attrs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VFSModel":
        config = _spec_dict(config)
        model_type = config.get("model") or config.get("type")
        if model_type and "plan" not in config:
            constructor = get_model(str(model_type))
            if constructor is None:
                raise ValueError(f"unknown model type {model_type!r}")
            kwargs = dict(config.get("kwargs", {}))
            mesh = config.get("mesh", kwargs.get("mesh"))
            column = config.get("column", kwargs.get("column"))
            if "mesh" in config:
                kwargs["mesh"] = mesh
            if "column" in config:
                kwargs["column"] = column
            extra_components = {}
            for name, spec in dict(config.get("components", {})).items():
                extra_components[str(name)] = _build_component(spec, name=str(name), mesh=mesh, column=column)
            if extra_components:
                existing = dict(kwargs.get("components", {}))
                existing.update(extra_components)
                kwargs["components"] = existing
            if "observation" in config:
                kwargs["observation"] = config.get("observation")
            if "ingestion" in config:
                kwargs["ingestion"] = config.get("ingestion")
            return constructor(**kwargs)
        components = {}
        mesh = config.get("mesh")
        column = config.get("column")
        for name, spec in dict(config.get("components", {})).items():
            components[str(name)] = _build_component(spec, name=str(name), mesh=mesh, column=column)
        return cls(
            mesh=mesh,
            column=column,
            components=components,
            plan=list(config.get("plan", [])),
            observation=config.get("observation"),
            ingestion=config.get("ingestion"),
        )

    @property
    def dry(self):
        return self.components.get("dry")

    @property
    def physics(self):
        return self.components.get("hs")

    @property
    def diagnostics(self):
        dry = self.dry
        return None if dry is None else getattr(dry, "diagnostics", None)


register_component("held_suarez", HeldSuarezPhysics)
register_component("scm_physics", SCMPhysics)
register_component("vfs_native_scm", VFSNativeSCM)
register_component("mesh_to_latlon_knn", lambda **kwargs: _build_regrid_component("mesh_to_latlon_knn", kwargs, mesh=kwargs.get("mesh")))
register_component("latlon_to_mesh_bilinear", lambda **kwargs: _build_regrid_component("latlon_to_mesh_bilinear", kwargs, mesh=kwargs.get("mesh")))
register_integrator("forward_euler", forward_euler)
register_integrator("rk4", rk4)
register_integrator("ssprk3", ssprk3)


@register_model(
    "dry_hydrostatic",
    status="stable",
    component="atmosphere",
    description="Canonical dry hydrostatic dycore recipe with explicit RK stepping.",
    default_safe=True,
)
class DryHydrostaticModel(VFSModel):
    """Named dry-core recipe.

    The model owns mesh/column modules and binds incoming State fields to those
    modules before stepping.
    """

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        all_components = {"dry": dynamics or DryHydrostaticDynamics()}
        all_components.update(components or {})
        super().__init__(
            mesh=mesh,
            column=column,
            components=all_components,
            plan=[{"type": "rk", "integrator": integrator, "tendencies": ["dry"]}],
            observation=observation,
            ingestion=ingestion,
        )

    def tendency(self, state: State) -> State:
        state = self.bind_state(state)
        return self.dry.tendency(state)


@register_model(
    "dry_hydrostatic_unscaled_coord_work",
    status="experimental",
    component="atmosphere",
    description="Comparison recipe retaining the original unscaled coordinate-work formula.",
    default_safe=False,
)
class DryHydrostaticUnscaledCoordWorkModel(DryHydrostaticModel):
    """Dry-core recipe retaining the original unscaled coordinate-work formula."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticUnscaledCoordWorkDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticUnscaledCoordWorkDynamics(),
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "dry_hydrostatic_log_pressure_sigma",
    status="provisional",
    component="atmosphere",
    description="Explicit GL2-compatible log-pressure sigma PGF recipe for terrain audits.",
    default_safe=False,
)
class DryHydrostaticLogPressureSigmaModel(DryHydrostaticModel):
    """Dry-core recipe that forces the log-pressure sigma PGF.

    The canonical dry dynamics now dispatches to this correction for GL2
    terrain-aware columns. This recipe remains an explicit audit/config path.
    """

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticLogPressureSigmaDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticLogPressureSigmaDynamics(),
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "dry_hydrostatic_virtual_temperature",
    status="experimental",
    component="atmosphere",
    description="Passive-vapor virtual-temperature dry-core research recipe.",
    default_safe=False,
)
class DryHydrostaticVirtualTemperatureModel(DryHydrostaticModel):
    """Dry-core recipe with experimental passive-vapor virtual temperature."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticVirtualTemperatureDynamics | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticVirtualTemperatureDynamics(),
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "held_suarez",
    status="stable",
    component="atmosphere",
    description="Canonical dry dycore plus Held-Suarez forcing recipe.",
    default_safe=True,
)
class HeldSuarezModel(VFSModel):
    """Named dry dynamics plus Held-Suarez physics recipe."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticDynamics | None = None,
        physics: HeldSuarezPhysics | None = None,
        constants: HeldSuarezConstants | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        all_components = {
            "dry": dynamics or DryHydrostaticDynamics(),
            "hs": physics or HeldSuarezPhysics(constants),
        }
        all_components.update(components or {})
        super().__init__(
            mesh=mesh,
            column=column,
            components=all_components,
            plan=[{"type": "rk", "integrator": integrator, "tendencies": ["dry", "hs"]}],
            observation=observation,
            ingestion=ingestion,
        )

    @property
    def constants(self) -> HeldSuarezConstants:
        return self.physics.constants

    @property
    def gravity(self) -> float:
        return self.dry.gravity

    def tendency(self, state: State) -> State:
        state = self.bind_state(state)
        return self.dry.tendency(state) + self.physics.tendency(state)

    def equilibrium_temperature(self, state: State, pressure):
        return self.physics.equilibrium_temperature(self.bind_state(state), pressure)

    def thermal_relaxation_rate(self, state: State, sigma):
        return self.physics.thermal_relaxation_rate(self.bind_state(state), sigma)

    def drag_rate(self, state: State, sigma):
        return self.physics.drag_rate(self.bind_state(state), sigma)


@register_model(
    "held_suarez_unscaled_coord_work",
    status="experimental",
    component="atmosphere",
    description="Held-Suarez comparison recipe using unscaled coordinate work.",
    default_safe=False,
)
class HeldSuarezUnscaledCoordWorkModel(HeldSuarezModel):
    """Held-Suarez recipe using the unscaled coordinate-work comparison dry core."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticUnscaledCoordWorkDynamics | None = None,
        physics: HeldSuarezPhysics | None = None,
        constants: HeldSuarezConstants | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticUnscaledCoordWorkDynamics(),
            physics=physics,
            constants=constants,
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "held_suarez_log_pressure_sigma",
    status="provisional",
    component="atmosphere",
    description="Held-Suarez recipe that forces the log-pressure sigma terrain PGF.",
    default_safe=False,
)
class HeldSuarezLogPressureSigmaModel(HeldSuarezModel):
    """Held-Suarez recipe for explicit log-pressure sigma PGF checks."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticLogPressureSigmaDynamics | None = None,
        physics: HeldSuarezPhysics | None = None,
        constants: HeldSuarezConstants | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticLogPressureSigmaDynamics(),
            physics=physics,
            constants=constants,
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )


@register_model(
    "held_suarez_virtual_temperature",
    status="experimental",
    component="atmosphere",
    description="Held-Suarez research recipe using passive-vapor virtual temperature.",
    default_safe=False,
)
class HeldSuarezVirtualTemperatureModel(HeldSuarezModel):
    """Held-Suarez recipe using the experimental virtual-temperature dry core."""

    def __init__(
        self,
        *,
        mesh,
        column,
        dynamics: DryHydrostaticVirtualTemperatureDynamics | None = None,
        physics: HeldSuarezPhysics | None = None,
        constants: HeldSuarezConstants | None = None,
        integrator: str = "ssprk3",
        components: dict[str, Any] | None = None,
        observation: str | None = None,
        ingestion: str | None = None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            column=column,
            dynamics=dynamics or DryHydrostaticVirtualTemperatureDynamics(),
            physics=physics,
            constants=constants,
            integrator=integrator,
            components=components,
            observation=observation,
            ingestion=ingestion,
        )
