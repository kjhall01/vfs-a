"""Explicit atmosphere/ocean coupled model driver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..core.integrators import ssprk3
from ..core.registry import register_component
from ..core.state import State
from ..process import PhysicsState, initial_physics_state_for_components, step_process_components
from .momentum import AirSeaMomentumCoupler
from .state import CoupledState, CoupledStepResult
from .surface_flux import AirSeaSurfaceFluxCoupler


def _named_components(components: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None) -> dict[str, Any]:
    if components is None:
        return {}
    if isinstance(components, Mapping):
        return {str(name): component for name, component in components.items()}
    return {f"process_{i}": component for i, component in enumerate(components)}


class _AtmosphereSurfaceState:
    """Atmosphere state plus compact time-integrated ocean surface fluxes."""

    def __init__(
        self,
        atmosphere: Any,
        stress: torch.Tensor,
        heat: torch.Tensor,
        freshwater: torch.Tensor,
    ) -> None:
        self.atmosphere = atmosphere
        self.stress = stress
        self.heat = heat
        self.freshwater = freshwater

    def __add__(self, other: "_AtmosphereSurfaceState") -> "_AtmosphereSurfaceState":
        return _AtmosphereSurfaceState(
            self.atmosphere + other.atmosphere,
            self.stress + other.stress,
            self.heat + other.heat,
            self.freshwater + other.freshwater,
        )

    def __sub__(self, other: "_AtmosphereSurfaceState") -> "_AtmosphereSurfaceState":
        return _AtmosphereSurfaceState(
            self.atmosphere - other.atmosphere,
            self.stress - other.stress,
            self.heat - other.heat,
            self.freshwater - other.freshwater,
        )

    def __mul__(self, scalar: float) -> "_AtmosphereSurfaceState":
        return _AtmosphereSurfaceState(
            self.atmosphere * scalar,
            self.stress * scalar,
            self.heat * scalar,
            self.freshwater * scalar,
        )

    def __rmul__(self, scalar: float) -> "_AtmosphereSurfaceState":
        return self.__mul__(scalar)


class CoupledAtmosphereOceanModel(nn.Module):
    """Explicit atmosphere/ocean stepper using VFS-native State objects.

    This first coupled model keeps the atmospheric and ocean models as separate
    owners. Its coupled ``tendency`` adds direct air-sea exchange to an
    optional atmosphere-model tendency and an optional ocean-dynamics tendency.
    Ocean model physics is used only as the exchange stress operator, so the
    same surface stress is not applied twice. Passing ``atmosphere_rk_substeps``
    switches the dynamics step to a multirate mode: the atmosphere and compact
    surface fluxes substep against a frozen ocean, then the ocean advances once
    with the time-averaged surface forcing. Stateful process components can be
    split before, after, or around the coupled RK exchange step through
    ``step_with_physics_state``.
    """

    recipe_name = "coupled_atmosphere_ocean"

    def __init__(
        self,
        *,
        atmosphere_model: Any | None = None,
        ocean_model: Any,
        coupler: AirSeaMomentumCoupler | None = None,
        thermodynamic_coupler: AirSeaSurfaceFluxCoupler | None = None,
        ocean_physics: Any | None = None,
        integrator: Any = ssprk3,
        include_atmosphere_tendency: bool = True,
        include_ocean_dynamics: bool = True,
        couple_momentum: bool = True,
        momentum_edge_mask: Any | None = None,
        atmosphere_processes: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        coupled_processes: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        process_order: str = "coupled_then_process",
        atmosphere_process_substeps: int = 1,
        coupled_process_substeps: int | Mapping[str, int] = 1,
        coupled_rk_substeps: int = 1,
        atmosphere_rk_substeps: int | None = None,
    ) -> None:
        super().__init__()
        self.atmosphere_model = atmosphere_model
        self.ocean_model = ocean_model
        self.coupler = coupler or AirSeaMomentumCoupler()
        self.thermodynamic_coupler = thermodynamic_coupler
        self.ocean_physics = ocean_physics
        self.integrator = integrator
        self.include_atmosphere_tendency = bool(include_atmosphere_tendency)
        self.include_ocean_dynamics = bool(include_ocean_dynamics)
        self.couple_momentum = bool(couple_momentum)
        self.momentum_edge_mask = momentum_edge_mask
        self.atmosphere_processes = _named_components(atmosphere_processes)
        self.coupled_processes = _named_components(coupled_processes)
        self.process_order = str(process_order)
        self.atmosphere_process_substeps = max(1, int(atmosphere_process_substeps))
        if isinstance(coupled_process_substeps, Mapping):
            self.coupled_process_substeps = {
                str(name): max(1, int(value)) for name, value in coupled_process_substeps.items()
            }
        else:
            self.coupled_process_substeps = max(1, int(coupled_process_substeps))
        self.coupled_rk_substeps = max(1, int(coupled_rk_substeps))
        self.atmosphere_rk_substeps = (
            None if atmosphere_rk_substeps is None else max(1, int(atmosphere_rk_substeps))
        )
        if self.atmosphere_rk_substeps is not None and self.coupled_rk_substeps != 1:
            raise ValueError(
                "atmosphere_rk_substeps cannot be combined with coupled_rk_substeps; "
                "the multirate path advances the ocean once per outer step"
            )
        if self.process_order not in ("coupled_then_process", "process_then_coupled", "strang_process"):
            raise ValueError(
                "process_order must be 'coupled_then_process', 'process_then_coupled', or 'strang_process'"
            )

    @staticmethod
    def _bind_component_state(model: Any | None, state: Any) -> Any:
        binder = getattr(model, "bind_state", None)
        return binder(state) if callable(binder) else state

    @staticmethod
    def _model_tendency(model: Any | None, state: Any, *, enabled: bool, name: str) -> Any:
        if not enabled or model is None:
            return state.zeros_like()
        tendency = getattr(model, "tendency", None)
        if callable(tendency):
            return tendency(state)
        if callable(model):
            return model(state)
        raise TypeError(f"{name} must provide tendency(state) or be callable")

    @staticmethod
    def _tensor_rms(value: torch.Tensor) -> float:
        return float(value.detach().to(dtype=torch.float32).square().mean().sqrt().cpu().item())

    @staticmethod
    def _finite_state(state: Any) -> bool:
        tensors = []
        for name in ("u_e", "mtheta_i", "ps", "tracer_m", "h", "temp_h", "salt_h", "tracer_h"):
            value = getattr(state, name, None)
            if torch.is_tensor(value):
                tensors.append(value)
        return all(bool(torch.isfinite(value).all().detach().cpu().item()) for value in tensors)

    @staticmethod
    def _stage_callback(callback: Any | None, stage: str, state: CoupledState) -> None:
        if callback is not None:
            callback(stage, state)

    def bind_state(self, state: CoupledState) -> CoupledState:
        """Bind child states to their model-owned geometry where available."""

        return CoupledState(
            self._bind_component_state(self.atmosphere_model, state.atmosphere),
            self._bind_component_state(self.ocean_model, state.ocean),
            attrs=state._attrs_copy(),
        )

    def _ocean_exchange_physics(self) -> Any:
        physics = self.ocean_physics
        if physics is None:
            physics = getattr(self.ocean_model, "physics", None)
        if physics is None:
            raise ValueError("momentum coupling requires ocean_physics or ocean_model.physics")
        return physics

    def _momentum_mask_like(self, value: torch.Tensor) -> torch.Tensor | None:
        """Normalize the optional fractional ocean-edge mask for stress use."""

        if self.momentum_edge_mask is None:
            return None
        return self.coupler._edge_column_field(
            self.momentum_edge_mask,
            value,
            name="momentum_edge_mask",
        ).clamp(0.0, 1.0)

    def _ocean_dynamics_tendency(self, ocean_state: Any) -> Any:
        dynamics = getattr(self.ocean_model, "dynamics", None)
        if dynamics is not None:
            return self._model_tendency(
                dynamics,
                ocean_state,
                enabled=self.include_ocean_dynamics,
                name="ocean dynamics",
            )
        if not self.include_ocean_dynamics:
            return ocean_state.zeros_like()
        raise TypeError(
            "CoupledAtmosphereOceanModel expects ocean_model.dynamics so ocean dynamics "
            "can be advanced without reapplying ocean surface physics"
        )

    def initial_physics_state(self, state: CoupledState) -> PhysicsState:
        """Initialize process-owned restart state for coupled split steps."""

        bound = self.bind_state(state)
        data: dict[str, Any] = {}
        if self.atmosphere_processes:
            data["atmosphere"] = initial_physics_state_for_components(
                self.atmosphere_processes,
                bound.atmosphere,
            ).data
        if self.coupled_processes:
            data["coupled"] = initial_physics_state_for_components(
                self.coupled_processes,
                bound,
            ).data
        return PhysicsState(data, attrs={"coupled_process_order": self.process_order})

    @staticmethod
    def _as_physics_state(value: PhysicsState | Mapping[str, Any] | None) -> PhysicsState:
        if value is None:
            return PhysicsState()
        if isinstance(value, PhysicsState):
            return value
        if isinstance(value, Mapping):
            return PhysicsState(dict(value))
        raise TypeError("physics_state must be PhysicsState, a mapping, or None")

    def _run_atmosphere_processes(
        self,
        state: CoupledState,
        dt: float,
        root: PhysicsState,
    ) -> tuple[CoupledState, PhysicsState, dict[str, Any]]:
        if not self.atmosphere_processes:
            return state, root, {}
        sub_state = PhysicsState(dict(root.component("atmosphere", {}) or {}))
        result = step_process_components(
            self.atmosphere_processes,
            state.atmosphere,
            dt,
            physics_state=sub_state,
            substeps=self.atmosphere_process_substeps,
        )
        next_root = root.with_component(
            "atmosphere",
            {} if result.physics_state is None else dict(result.physics_state.data),
        )
        next_state = CoupledState(result.state, state.ocean, attrs=state._attrs_copy())
        return next_state, next_root, dict(result.diagnostics or {})

    def _run_coupled_processes(
        self,
        state: CoupledState,
        dt: float,
        root: PhysicsState,
    ) -> tuple[CoupledState, PhysicsState, dict[str, Any]]:
        if not self.coupled_processes:
            return state, root, {}
        sub_state = PhysicsState(dict(root.component("coupled", {}) or {}))
        result = step_process_components(
            self.coupled_processes,
            state,
            dt,
            physics_state=sub_state,
            substeps=self.coupled_process_substeps,
        )
        next_root = root.with_component(
            "coupled",
            {} if result.physics_state is None else dict(result.physics_state.data),
        )
        return result.state, next_root, dict(result.diagnostics or {})

    def _run_processes(
        self,
        state: CoupledState,
        dt: float,
        root: PhysicsState,
    ) -> tuple[CoupledState, PhysicsState, dict[str, Any]]:
        diagnostics: dict[str, Any] = {}
        current, root, atmosphere_diag = self._run_atmosphere_processes(state, dt, root)
        if atmosphere_diag:
            diagnostics["atmosphere"] = atmosphere_diag
        current, root, coupled_diag = self._run_coupled_processes(current, dt, root)
        if coupled_diag:
            diagnostics["coupled"] = coupled_diag
        return current, root, diagnostics

    def exchange(self, state: CoupledState, *, include_audit: bool = True) -> dict[str, Any]:
        """Return current air-sea exchange tendencies and optional audit terms."""

        if not self.couple_momentum:
            return {}
        bound = self.bind_state(state)
        return self.coupler.coupled_momentum_tendencies(
            bound.atmosphere,
            bound.ocean,
            self._ocean_exchange_physics(),
            edge_mask=self.momentum_edge_mask,
            include_audit=include_audit,
        )

    def thermodynamic_exchange(self, state: CoupledState, *, include_audit: bool = True) -> dict[str, Any]:
        """Return current air-sea thermodynamic exchange tendencies."""

        if self.thermodynamic_coupler is None:
            return {}
        bound = self.bind_state(state)
        return self.thermodynamic_coupler.coupled_thermodynamic_tendencies(
            bound.atmosphere,
            bound.ocean,
            include_audit=include_audit,
        )

    def surface_exchange(self, state: CoupledState, *, include_audit: bool = True) -> dict[str, Any]:
        """Return current momentum and thermodynamic exchange payloads."""

        out: dict[str, Any] = {}
        if self.couple_momentum:
            momentum = self.exchange(state, include_audit=include_audit)
            out.update(momentum)
            out["momentum"] = momentum
        if self.thermodynamic_coupler is not None:
            out["thermodynamic"] = self.thermodynamic_exchange(state, include_audit=include_audit)
        return out

    def tendency(
        self,
        state: CoupledState,
        *,
        atmosphere_tendency: Any | None = None,
    ) -> CoupledState:
        """Return a coupled atmosphere/ocean RHS in ``CoupledState`` form."""

        bound = self.bind_state(state)
        if atmosphere_tendency is None:
            atmosphere_rhs = self._model_tendency(
                self.atmosphere_model,
                bound.atmosphere,
                enabled=self.include_atmosphere_tendency,
                name="atmosphere model",
            )
        elif not self.include_atmosphere_tendency or self.atmosphere_model is None:
            atmosphere_rhs = bound.atmosphere.zeros_like()
        else:
            atmosphere_rhs = atmosphere_tendency(bound.atmosphere)
            if not isinstance(atmosphere_rhs, State):
                raise TypeError("atmosphere_tendency override must return a VFS State")
        ocean_rhs = self._ocean_dynamics_tendency(bound.ocean)
        if self.couple_momentum:
            exchange = self.coupler.coupled_momentum_tendencies(
                bound.atmosphere,
                bound.ocean,
                self._ocean_exchange_physics(),
                edge_mask=self.momentum_edge_mask,
                include_audit=False,
            )
            atmosphere_rhs = atmosphere_rhs + exchange["atmosphere_rhs"]
            ocean_rhs = ocean_rhs + exchange["ocean_rhs"]
        if self.thermodynamic_coupler is not None:
            thermodynamic = self.thermodynamic_coupler.coupled_thermodynamic_tendencies(
                bound.atmosphere,
                bound.ocean,
                include_audit=False,
            )
            atmosphere_rhs = atmosphere_rhs + thermodynamic["atmosphere_rhs"]
            ocean_rhs = ocean_rhs + thermodynamic["ocean_rhs"]
        return CoupledState(atmosphere_rhs, ocean_rhs, attrs=bound._attrs_copy())

    def _coupled_rk_step(self, state: CoupledState, dt: float) -> CoupledState:
        """Advance one dynamics/surface-exchange step without process physics."""

        if self.atmosphere_rk_substeps is not None:
            return self._multirate_rk_step(state, dt)
        current = self.bind_state(state)
        substeps = max(1, int(self.coupled_rk_substeps))
        sub_dt = float(dt) / substeps
        for _ in range(substeps):
            current = self.integrator(self, current, sub_dt)
        return current

    def _multirate_rk_step(self, state: CoupledState, dt: float) -> CoupledState:
        """Substep the atmosphere while holding the ocean state fixed."""

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("multirate atmosphere/ocean stepping requires dt > 0")
        current = self.bind_state(state)
        frozen_ocean = current.ocean
        ocean_physics = self._ocean_exchange_physics() if self.couple_momentum else None
        if ocean_physics is not None and not callable(
            getattr(ocean_physics, "tendency_with_surface_stress", None)
        ):
            raise TypeError(
                "multirate momentum coupling requires ocean physics with "
                "tendency_with_surface_stress(state, stress)"
            )

        stress0 = torch.zeros_like(frozen_ocean.u_e[..., :1])
        heat0 = torch.zeros_like(frozen_ocean.temp_h[..., :1])
        freshwater0 = torch.zeros_like(frozen_ocean.h[..., :1])
        fast = _AtmosphereSurfaceState(current.atmosphere, stress0, heat0, freshwater0)

        def atmosphere_tendency(fast_state: _AtmosphereSurfaceState) -> _AtmosphereSurfaceState:
            atmosphere = self._bind_component_state(self.atmosphere_model, fast_state.atmosphere)
            atmosphere_rhs = self._model_tendency(
                self.atmosphere_model,
                atmosphere,
                enabled=self.include_atmosphere_tendency,
                name="atmosphere model",
            )
            stress = torch.zeros_like(stress0)
            heat = torch.zeros_like(heat0)
            freshwater = torch.zeros_like(freshwater0)
            if self.couple_momentum:
                prepared = self.coupler.prepare_ocean_state(frozen_ocean, atmosphere)
                stress_provider = getattr(ocean_physics, "wind_stress_edge", None)
                stress = (
                    stress_provider(prepared)
                    if callable(stress_provider)
                    else self.coupler.surface_stress_edge(atmosphere, frozen_ocean)
                )
                mask = self._momentum_mask_like(stress)
                if mask is not None:
                    stress = stress * mask
                atmosphere_rhs = atmosphere_rhs + self.coupler.atmospheric_momentum_tendency(
                    atmosphere,
                    stress_on_ocean_edge=stress,
                )
            if self.thermodynamic_coupler is not None:
                fluxes = self.thermodynamic_coupler.fluxes(atmosphere, frozen_ocean)
                atmosphere_rhs = atmosphere_rhs + self.thermodynamic_coupler.atmospheric_tendency(
                    atmosphere,
                    fluxes=fluxes,
                )
                heat = fluxes["ocean_surface_heat_flux_down"]
                freshwater = fluxes["freshwater_flux_down"]
            return _AtmosphereSurfaceState(atmosphere_rhs, stress, heat, freshwater)

        substeps = int(self.atmosphere_rk_substeps or 1)
        sub_dt = dt / substeps
        for _ in range(substeps):
            fast = self.integrator(atmosphere_tendency, fast, sub_dt)

        average_stress = fast.stress / dt
        average_heat = fast.heat / dt
        average_freshwater = fast.freshwater / dt

        def ocean_tendency(ocean_state: Any) -> Any:
            ocean = self._bind_component_state(self.ocean_model, ocean_state)
            ocean_rhs = self._ocean_dynamics_tendency(ocean)
            if ocean_physics is not None:
                ocean_rhs = ocean_rhs + ocean_physics.tendency_with_surface_stress(
                    ocean,
                    average_stress,
                )
            if self.thermodynamic_coupler is not None:
                ocean_rhs = ocean_rhs + self.thermodynamic_coupler.ocean_tendency(
                    ocean,
                    fluxes={
                        "ocean_surface_heat_flux_down": average_heat,
                        "freshwater_flux_down": average_freshwater,
                    },
                )
            return ocean_rhs

        ocean = self.integrator(ocean_tendency, frozen_ocean, dt)
        return self.bind_state(
            CoupledState(fast.atmosphere, ocean, attrs=current._attrs_copy())
        )

    def _diagnostics_from_exchange(
        self,
        exchange: dict[str, Any],
        next_state: CoupledState,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "atmosphere_finite": self._finite_state(next_state.atmosphere),
            "ocean_finite": self._finite_state(next_state.ocean),
        }
        momentum = exchange.get("momentum", {})
        if momentum:
            diagnostics.update(momentum.get("impulse_summary", {}))
            stress0 = momentum["stress_on_ocean_edge"]
            stress1 = self.coupler.surface_stress_edge(next_state.atmosphere, next_state.ocean)
            diagnostics["stress_after_to_before_ratio"] = self._tensor_rms(stress1) / max(
                self._tensor_rms(stress0),
                1.0e-300,
            )
        thermodynamic = exchange.get("thermodynamic", {})
        if thermodynamic:
            diagnostics.update(thermodynamic.get("flux_summary", {}))
        return diagnostics

    def step(self, state: CoupledState, dt: float) -> CoupledState:
        """Advance one coupled step, discarding any process restart state."""

        return self.step_with_physics_state(state, dt).state

    def step_with_physics_state(
        self,
        state: CoupledState,
        dt: float,
        physics_state: PhysicsState | Mapping[str, Any] | None = None,
        stage_callback: Any | None = None,
    ) -> CoupledStepResult:
        """Advance one coupled step with explicit process-order semantics.

        ``process_order`` controls where split process components run relative
        to the coupled dynamics/surface-exchange RK step:

        - ``coupled_then_process``: dynamics/surface exchange, then processes.
        - ``process_then_coupled``: processes, then dynamics/surface exchange.
        - ``strang_process``: half processes, dynamics/surface exchange,
          half processes.
        """

        current = self.bind_state(state)
        root = (
            self.initial_physics_state(current)
            if physics_state is None
            else self._as_physics_state(physics_state)
        )
        self._stage_callback(stage_callback, "start", current)
        process_diagnostics: dict[str, Any] = {}
        exchange_state = current
        if self.process_order == "process_then_coupled":
            self._stage_callback(stage_callback, "before_processes", current)
            current, root, process_diagnostics = self._run_processes(current, float(dt), root)
            self._stage_callback(stage_callback, "after_processes", current)
            exchange_state = current
            self._stage_callback(stage_callback, "before_coupled_rk", current)
            current = self._coupled_rk_step(current, dt)
            self._stage_callback(stage_callback, "after_coupled_rk", current)
        elif self.process_order == "strang_process":
            self._stage_callback(stage_callback, "before_pre_processes", current)
            current, root, pre_diag = self._run_processes(current, 0.5 * float(dt), root)
            self._stage_callback(stage_callback, "after_pre_processes", current)
            if pre_diag:
                process_diagnostics["pre"] = pre_diag
            exchange_state = current
            self._stage_callback(stage_callback, "before_coupled_rk", current)
            current = self._coupled_rk_step(current, dt)
            self._stage_callback(stage_callback, "after_coupled_rk", current)
            self._stage_callback(stage_callback, "before_post_processes", current)
            current, root, post_diag = self._run_processes(current, 0.5 * float(dt), root)
            self._stage_callback(stage_callback, "after_post_processes", current)
            if post_diag:
                process_diagnostics["post"] = post_diag
        else:
            exchange_state = current
            self._stage_callback(stage_callback, "before_coupled_rk", current)
            current = self._coupled_rk_step(current, dt)
            self._stage_callback(stage_callback, "after_coupled_rk", current)
            self._stage_callback(stage_callback, "before_processes", current)
            current, root, process_diagnostics = self._run_processes(current, float(dt), root)
            self._stage_callback(stage_callback, "after_processes", current)
        self._stage_callback(stage_callback, "final", current)

        exchange = self.surface_exchange(exchange_state, include_audit=True)
        diagnostics = self._diagnostics_from_exchange(exchange, current)
        diagnostics["process_order"] = self.process_order
        diagnostics["coupled_rk_substeps"] = int(self.coupled_rk_substeps)
        diagnostics["coupled_rk_substep_dt"] = float(dt) / max(1, int(self.coupled_rk_substeps))
        diagnostics["time_integration_mode"] = (
            "atmosphere_substeps" if self.atmosphere_rk_substeps is not None else "common_clock"
        )
        diagnostics["atmosphere_rk_substeps"] = (
            1 if self.atmosphere_rk_substeps is None else int(self.atmosphere_rk_substeps)
        )
        diagnostics["atmosphere_rk_substep_dt"] = (
            float(dt)
            if self.atmosphere_rk_substeps is None
            else float(dt) / int(self.atmosphere_rk_substeps)
        )
        diagnostics["ocean_rk_substeps"] = (
            int(self.coupled_rk_substeps) if self.atmosphere_rk_substeps is None else 1
        )
        if process_diagnostics:
            diagnostics["process"] = process_diagnostics
        return CoupledStepResult(current, exchange=exchange, diagnostics=diagnostics, physics_state=root)

    def step_with_diagnostics(self, state: CoupledState, dt: float) -> CoupledStepResult:
        """Advance one step and return interface diagnostics."""

        return self.step_with_physics_state(state, dt)

    def forward(self, state: CoupledState, dt: float) -> CoupledState:
        return self.step(state, dt)


register_component("coupled_atmosphere_ocean", CoupledAtmosphereOceanModel)


__all__ = ["CoupledAtmosphereOceanModel"]
