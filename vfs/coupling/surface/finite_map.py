"""Extended-state surface physics source for coupled ARK stages.

``SurfaceSystem`` owns rich, restartable land/ice/runoff state but its public
kernel is a finite, positivity-limited map.  This adapter makes that state an
explicit ARK memory tree and defines a controlled rate by a small, separately
recorded surface microstep.  It is deliberately guarded: the model must not
also apply its legacy direct air--sea thermodynamic tendency, because this
source owns the fraction-aware ocean heat and freshwater packet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...field import Field
from ...surface import AtmosSurfaceFluxPacket, SurfaceFractions, SurfaceState
from ...surface.system import SurfaceStepResult
from ...surface.glacier_land_ice import GlacierState
from ...surface.icebergs import IcebergState
from ...surface.runoff_router import RunoffState
from ...surface.sea_ice import SeaIceState
from ...surface.conservation import TRANSFER_QUANTITIES, TransferLedger
from ...surface.utils import field_values
from ...land import LandSurfaceState
from ...ice.evp import SeaIceEVP, SeaIceEVPStepResult
from ..ark_stage_sources import CoupledARKStageSourceResult
from ..momentum import AirSeaMomentumCoupler
from ..state import CoupledState


def _values(value: Field | torch.Tensor) -> torch.Tensor:
    result = field_values(value)
    if not torch.is_tensor(result):
        raise TypeError(f"surface ARK memory values must be tensors, got {type(result).__name__}")
    return result


@dataclass(frozen=True)
class SurfaceARKMemoryCodec:
    """Flatten dynamic ``SurfaceState`` reservoirs into strict tensor memory."""

    template: SurfaceState

    def encode(self, state: SurfaceState) -> dict[str, Any]:
        state.validate()
        template = self.template
        if int(state.n_cells) != int(template.n_cells):
            raise ValueError("surface ARK state n_cells differs from the source template")
        out: dict[str, Any] = {
            "fractions": {
                "sea_ice_fraction": _values(state.fractions.sea_ice_fraction),
                "glacier_fraction": _values(state.fractions.glacier_fraction),
                "snow_fraction": _values(state.fractions.snow_fraction),
            }
        }
        if state.land is not None:
            out["land"] = {
                "soil_liquid_water": _values(state.land.soil_liquid_water),
                "soil_temperature": _values(state.land.soil_temperature),
                "snow_water_equivalent": _values(state.land.snow_water_equivalent),
                "skin_temperature": _values(state.land.skin_temperature),
                "soil_ice": _values(state.land.soil_ice),
                "canopy_water": _values(state.land.canopy_water),
            }
        if state.sea_ice is not None:
            out["sea_ice"] = {
                "sea_ice_thickness": _values(state.sea_ice.sea_ice_thickness),
                "snow_on_sea_ice": _values(state.sea_ice.snow_on_sea_ice),
                "sea_ice_surface_temperature": _values(state.sea_ice.sea_ice_surface_temperature),
            }
            if state.sea_ice.sea_ice_u_e is not None:
                out["sea_ice"].update(
                    {
                        "sea_ice_u_e": _values(state.sea_ice.sea_ice_u_e),
                        "sea_ice_stress_xx": _values(state.sea_ice.sea_ice_stress_xx),
                        "sea_ice_stress_xy": _values(state.sea_ice.sea_ice_stress_xy),
                        "sea_ice_stress_yy": _values(state.sea_ice.sea_ice_stress_yy),
                    }
                )
        if state.glacier is not None:
            out["glacier"] = {
                "land_ice_thickness": _values(state.glacier.land_ice_thickness),
                "snow_on_glacier": _values(state.glacier.snow_on_glacier),
                "glacier_surface_temperature": _values(state.glacier.glacier_surface_temperature),
            }
        if state.icebergs is not None:
            out["icebergs"] = {"iceberg_mass": _values(state.icebergs.iceberg_mass)}
        if state.runoff is not None:
            out["runoff"] = {"river_storage": _values(state.runoff.river_storage)}
        return out

    def decode(self, memory: Mapping[str, Any]) -> SurfaceState:
        if not isinstance(memory, Mapping):
            raise TypeError("surface ARK memory must be a mapping")
        template = self.template
        fractions_memory = self._component(memory, "fractions")
        fractions = SurfaceFractions(
            template.fractions.land_fraction,
            template.fractions.ocean_fraction,
            fractions_memory["sea_ice_fraction"],
            fractions_memory["glacier_fraction"],
            fractions_memory["snow_fraction"],
        )

        land = None
        if template.land is not None:
            values = self._component(memory, "land")
            land = LandSurfaceState(
                values["soil_liquid_water"],
                values["soil_temperature"],
                values["snow_water_equivalent"],
                values["skin_temperature"],
                values["soil_ice"],
                values["canopy_water"],
                template.land.land_use_type,
            )
        sea_ice = None
        if template.sea_ice is not None:
            values = self._component(memory, "sea_ice")
            sea_ice = SeaIceState(
                fractions.sea_ice_fraction,
                values["sea_ice_thickness"],
                values["snow_on_sea_ice"],
                values["sea_ice_surface_temperature"],
                values.get("sea_ice_u_e"),
                values.get("sea_ice_stress_xx"),
                values.get("sea_ice_stress_xy"),
                values.get("sea_ice_stress_yy"),
            )
        glacier = None
        if template.glacier is not None:
            values = self._component(memory, "glacier")
            glacier = GlacierState(
                fractions.glacier_fraction,
                values["land_ice_thickness"],
                values["snow_on_glacier"],
                values["glacier_surface_temperature"],
            )
        icebergs = None
        if template.icebergs is not None:
            icebergs = IcebergState(self._component(memory, "icebergs")["iceberg_mass"])
        runoff = None
        if template.runoff is not None:
            runoff = RunoffState(self._component(memory, "runoff")["river_storage"])
        state = SurfaceState(
            fractions,
            land,
            sea_ice,
            glacier,
            icebergs,
            runoff,
            template._attrs_copy(),
        )
        state.validate()
        return state

    @staticmethod
    def _component(memory: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
        value = memory.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"surface ARK memory is missing mapping component {name!r}")
        if not all(torch.is_tensor(item) for item in value.values()):
            raise TypeError(f"surface ARK memory component {name!r} must contain tensors only")
        return value

    def rate(self, before: SurfaceState, after: SurfaceState, dt: float) -> dict[str, Any]:
        if not torch.isfinite(torch.as_tensor(float(dt))) or float(dt) <= 0.0:
            raise ValueError("surface ARK rate dt must be finite and positive")
        left = self.encode(before)
        right = self.encode(after)
        return self._tree_rate(left, right, float(dt))

    @classmethod
    def _tree_rate(cls, before: Mapping[str, Any], after: Mapping[str, Any], dt: float) -> dict[str, Any]:
        if set(before) != set(after):
            raise ValueError("surface ARK state components changed during a microstep")
        out: dict[str, Any] = {}
        for name, value in before.items():
            next_value = after[name]
            if isinstance(value, Mapping) and isinstance(next_value, Mapping):
                out[name] = cls._tree_rate(value, next_value, dt)
            elif torch.is_tensor(value) and torch.is_tensor(next_value):
                if value.shape != next_value.shape:
                    raise ValueError(f"surface ARK memory shape changed for {name!r}")
                out[name] = (next_value - value) / float(dt)
            else:
                raise TypeError(f"surface ARK memory leaf {name!r} is not a matching tensor")
        return out


@dataclass(frozen=True)
class CoupledSurfaceSubcycleResult:
    """One physical, source-only surface subcycle.

    ``memory`` is the exact endpoint of ``SurfaceSystem.step`` rather than an
    RK linear combination of an estimated memory rate.  This small result is
    used by the Strang/subcycling route below the ARK gravity-wave core; it is
    intentionally separate from :class:`CoupledARKStageSourceResult` so the
    two integration contracts cannot be confused.
    """

    state: CoupledState
    memory: dict[str, Any]
    diagnostics: dict[str, Any]


class CoupledSurfaceSystemStageSource:
    """Stage-coupled SCM plus prognostic land/ice/runoff surface source.

    The SCM supplies atmosphere tendencies and an :class:`AtmosSurfaceFluxPacket`
    at each tableau stage. ``SurfaceSystem`` advances one short positive,
    conservative microstep from the stage surface memory; its finite difference
    defines the surface-memory RHS. Users must demonstrate convergence when
    reducing ``surface_rate_substep_seconds`` before promoting a timestep.
    """

    recipe_name = "coupled_surface_system_stage_source"

    def __init__(
        self,
        scm_process: Any,
        surface_system: Any,
        surface_state: SurfaceState,
        *,
        surface_rate_substep_seconds: float = 1.0,
        ocean_density: float = 1025.0,
        ocean_heat_capacity: float = 3990.0,
        water_density: float = 1000.0,
        reference_salinity: float = 35.0,
        snow_temperature_k: float = 273.15,
        ocean_domain: Any | None = None,
        sea_ice_momentum: SeaIceEVP | None = None,
    ) -> None:
        if not callable(getattr(scm_process, "evaluate_ark_stage", None)):
            raise TypeError("surface stage source requires CoupledSCMProcess.evaluate_ark_stage(...)")
        if not callable(getattr(surface_system, "step", None)):
            raise TypeError("surface stage source requires SurfaceSystem.step(...)")
        if float(surface_rate_substep_seconds) <= 0.0:
            raise ValueError("surface_rate_substep_seconds must be positive")
        if min(float(ocean_density), float(ocean_heat_capacity), float(water_density)) <= 0.0:
            raise ValueError("surface/ocean densities and heat capacity must be positive")
        self.scm_process = scm_process
        self.surface_system = surface_system
        self.codec = SurfaceARKMemoryCodec(surface_state)
        self.surface_rate_substep_seconds = float(surface_rate_substep_seconds)
        self.ocean_density = float(ocean_density)
        self.ocean_heat_capacity = float(ocean_heat_capacity)
        self.water_density = float(water_density)
        self.reference_salinity = float(reference_salinity)
        self.snow_temperature_k = float(snow_temperature_k)
        self.ocean_domain = ocean_domain
        if sea_ice_momentum is not None and not isinstance(sea_ice_momentum, SeaIceEVP):
            raise TypeError("sea_ice_momentum must be SeaIceEVP or None")
        if sea_ice_momentum is not None and surface_state.sea_ice is None:
            raise ValueError("sea_ice_momentum requires SurfaceState.sea_ice")
        self.sea_ice_momentum = sea_ice_momentum
        self._ice_momentum_coupler = AirSeaMomentumCoupler()
        self._ark_outer_step_seconds: float | None = None

    def set_ark_step_context(self, dt: float) -> None:
        """Receive the outer explicit step used to bound reservoir rates."""

        if not torch.isfinite(torch.as_tensor(float(dt))) or float(dt) <= 0.0:
            raise ValueError("ARK surface step context must be finite and positive")
        self._ark_outer_step_seconds = float(dt)

    @staticmethod
    def _nonnegative_memory_leaf(path: tuple[str, ...]) -> bool:
        # Temperatures are signed anomalies/absolute temperatures and must not
        # be positivity-projected. EVP velocity and stress are signed
        # prognostics too. Every remaining encoded surface quantity is a
        # fraction, mass, depth, or water store with a physical lower bound.
        return path[-1] not in {
            "soil_temperature",
            "skin_temperature",
            "sea_ice_surface_temperature",
            "glacier_surface_temperature",
            "sea_ice_u_e",
            "sea_ice_stress_xx",
            "sea_ice_stress_xy",
            "sea_ice_stress_yy",
        }

    @classmethod
    def _limit_memory_rate_tree(
        cls,
        memory: Mapping[str, Any],
        tendency: Mapping[str, Any],
        dt: float,
        path: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in memory.items():
            rate = tendency.get(name)
            current_path = (*path, str(name))
            if isinstance(value, Mapping):
                if not isinstance(rate, Mapping):
                    raise TypeError(f"surface ARK memory tendency is missing mapping at {'.'.join(current_path)}")
                out[str(name)] = cls._limit_memory_rate_tree(value, rate, dt, current_path)
            elif torch.is_tensor(value) and torch.is_tensor(rate):
                if value.shape != rate.shape:
                    raise ValueError(f"surface ARK memory rate shape differs at {'.'.join(current_path)}")
                out[str(name)] = (
                    torch.maximum(rate, -value / float(dt)) if cls._nonnegative_memory_leaf(current_path) else rate
                )
            else:
                raise TypeError(f"surface ARK memory leaf is not a matching tensor at {'.'.join(current_path)}")
        return out

    @classmethod
    def _project_memory_tree(cls, memory: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in memory.items():
            current_path = (*path, str(name))
            if isinstance(value, Mapping):
                out[str(name)] = cls._project_memory_tree(value, current_path)
            elif torch.is_tensor(value):
                out[str(name)] = value.clamp_min(0.0) if cls._nonnegative_memory_leaf(current_path) else value
            else:
                raise TypeError(f"surface ARK memory leaf is not a tensor at {'.'.join(current_path)}")
        return out

    def project_ark_memory(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        """Project bounded reservoirs after an explicit tableau combination."""

        projected = self._project_memory_tree(memory)
        canopy = self._canopy_capacity_like(projected)
        if canopy is not None:
            projected["land"]["canopy_water"] = projected["land"]["canopy_water"].minimum(canopy)
        return projected

    def _canopy_capacity_like(self, memory: Mapping[str, Any]) -> torch.Tensor | None:
        if "land" not in memory or not isinstance(memory["land"], Mapping):
            return None
        canopy = memory["land"].get("canopy_water")
        land_model = getattr(self.surface_system, "land", None)
        template_land = self.codec.template.land
        if not torch.is_tensor(canopy) or land_model is None or template_land is None:
            return None
        fields = land_model.land_use_fields(template_land)
        return (
            float(land_model.parameters.canopy_water_capacity)
            * fields.canopy_capacity_factor.to(dtype=canopy.dtype, device=canopy.device)
        )

    def _limit_canopy_rate(
        self,
        memory: Mapping[str, Any],
        tendency: dict[str, Any],
        dt: float,
    ) -> dict[str, Any]:
        capacity = self._canopy_capacity_like(memory)
        if capacity is None:
            return tendency
        canopy = memory["land"]["canopy_water"]
        out = dict(tendency)
        land = dict(out["land"])
        land["canopy_water"] = torch.minimum(land["canopy_water"], (capacity - canopy) / float(dt))
        out["land"] = land
        return out

    def validate_ark_model(self, model: Any) -> None:
        if getattr(model, "thermodynamic_coupler", None) is not None:
            raise ValueError(
                "CoupledSurfaceSystemStageSource owns fraction-aware surface heat/freshwater routing; "
                "construct the coupled model with thermodynamic_coupler=None to avoid double application"
            )
        fractions = self.codec.template.fractions
        has_land = bool((_values(fractions.land_fraction) > 0.0).any().detach().cpu())
        if has_land and bool(getattr(model, "couple_momentum", False)) and getattr(model, "momentum_edge_mask", None) is None:
            raise ValueError(
                "CoupledSurfaceSystemStageSource with land requires a surface-masked momentum route; "
                "pass the source-derived momentum_edge_mask to the coupled model"
            )
        if self.sea_ice_momentum is not None and not bool(getattr(model, "couple_momentum", False)):
            raise ValueError("sea_ice_momentum requires coupled open-water atmosphere/ocean momentum")

    def momentum_edge_mask(self, mesh: Any) -> torch.Tensor:
        """Return the native-edge ocean ownership factor for drag exchange.

        A mixed cell contains an idealized ocean share even when it also owns
        land/ice reservoirs.  Projecting that ocean share to each edge keeps
        the atmosphere/ocean stress pair equal and opposite while eliminating
        fictitious land-side air--sea drag. Sea ice remains part of the ocean
        momentum domain until a dedicated ice-rheology stress route exists.
        """

        if int(mesh.nCells) != int(self.codec.template.n_cells):
            raise ValueError("surface state and momentum-mask mesh have different nCells")
        ocean = _values(self.codec.template.fractions.ocean_fraction)
        if self.sea_ice_momentum is not None:
            # Air--ice and ice--ocean drag are owned by the physical surface
            # map.  The ARK core receives only the exposed open-water share.
            ocean = (ocean - _values(self.codec.template.fractions.sea_ice_fraction)).clamp_min(0.0)
        mask = mesh.cell_to_edge(ocean).to(dtype=ocean.dtype, device=ocean.device).clamp(0.0, 1.0)
        if self.ocean_domain is not None:
            mask = self.ocean_domain.apply_edge(mask)
        return mask

    def initial_ark_physics_state(self, state: CoupledState) -> dict[str, Any]:
        if int(state.atmosphere.mesh.nCells) != int(self.codec.template.n_cells):
            raise ValueError("surface state and coupled atmosphere mesh have different nCells")
        self._ensure_ice_momentum_memory(state)
        return self.codec.encode(self.codec.template)

    def _ensure_ice_momentum_memory(self, state: CoupledState) -> None:
        """Attach edge velocity/stress leaves once the native mesh is known."""

        if self.sea_ice_momentum is None or self.codec.template.sea_ice is None:
            return
        template_ice = self.codec.template.sea_ice
        if template_ice.sea_ice_u_e is not None:
            return
        initialized = self.sea_ice_momentum.initialize(template_ice, state.atmosphere.mesh)
        self.codec = SurfaceARKMemoryCodec(self.codec.template.with_components(sea_ice=initialized))

    def ark_stage_tendency(
        self,
        state: CoupledState,
        physics_state: Mapping[str, Any],
        *,
        time_seconds: float,
    ) -> CoupledARKStageSourceResult:
        """Return the legacy explicit-tableau surface-memory rate.

        The ARK stage contract remains useful for short, smooth surface
        intervals.  Components containing capacity or complete-melt events
        should instead use :meth:`advance_explicit_substep` through the
        separate physical-subcycling integrator.  In that path the finite map
        itself, not an averaged derivative, advances restart memory.
        """

        surface_state = self.codec.decode(physics_state)
        evaluation, step, tendency = self._evaluate_physical_surface_substep(
            state,
            surface_state,
            self.surface_rate_substep_seconds,
            time_seconds=float(time_seconds),
        )
        memory_tendency = self.codec.rate(surface_state, step.state, self.surface_rate_substep_seconds)
        outer_dt = self.surface_rate_substep_seconds if self._ark_outer_step_seconds is None else self._ark_outer_step_seconds
        memory_tendency = self._limit_memory_rate_tree(physics_state, memory_tendency, outer_dt)
        memory_tendency = self._limit_canopy_rate(physics_state, memory_tendency, outer_dt)
        diagnostics = self._surface_diagnostics(
            evaluation,
            step,
            source_mode="microstep_rate",
            substep_seconds=self.surface_rate_substep_seconds,
        )
        diagnostics["surface_ark_outer_step_seconds"] = outer_dt
        return CoupledARKStageSourceResult(tendency, memory_tendency, diagnostics)

    def advance_explicit_substep(
        self,
        state: CoupledState,
        physics_state: Mapping[str, Any],
        dt: float,
        *,
        time_seconds: float,
    ) -> CoupledSurfaceSubcycleResult:
        """Advance one bounded physical surface substep exactly once.

        This is a source-only first-order substep: the SCM/ocean tendency is
        applied to the coupled state over ``dt`` and the restart memory is the
        endpoint returned by the physical ``SurfaceSystem`` map over exactly
        the same interval.  Repeating it under a symmetric split around the
        ARK2 gravity-wave core preserves the land canopy and sea-ice limiter
        semantics without passing a clipped finite-map rate through an ARK
        tableau.
        """

        dt = float(dt)
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0.0:
            raise ValueError("physical surface substep dt must be finite and positive")
        surface_state = self.codec.decode(physics_state)
        evaluation, step, tendency = self._evaluate_physical_surface_substep(
            state,
            surface_state,
            dt,
            time_seconds=float(time_seconds),
        )
        next_state = state + dt * tendency
        next_memory = self.codec.encode(step.state)
        diagnostics = self._surface_diagnostics(
            evaluation,
            step,
            source_mode="physical_subcycle_map",
            substep_seconds=dt,
        )
        return CoupledSurfaceSubcycleResult(next_state, next_memory, diagnostics)

    def _evaluate_physical_surface_substep(
        self,
        state: CoupledState,
        surface_state: SurfaceState,
        dt: float,
        *,
        time_seconds: float,
    ) -> tuple[Any, Any, CoupledState]:
        """Evaluate the shared SCM packet and one conservative surface map."""

        evaluation = self.scm_process.evaluate_ark_stage(
            state,
            {},
            time_seconds=float(time_seconds),
            surface_state=surface_state,
            include_ocean_outputs=False,
        )
        fluxes = self._atmos_surface_flux_packet(evaluation.outputs, evaluation.surface_params)
        step = self.surface_system.step(
            surface_state,
            float(dt),
            fluxes=fluxes,
            ocean_state=state.ocean,
            mesh=state.atmosphere.mesh,
        )
        ocean_rhs = self._ocean_tendency_from_packet(state.ocean, step.ocean_fluxes)
        atmosphere_rhs = evaluation.state_tendency.atmosphere
        if self.sea_ice_momentum is not None:
            step, ice_result = self._apply_sea_ice_momentum(step, state, float(dt))
            atmosphere_rhs = atmosphere_rhs + self._ice_momentum_coupler.atmospheric_momentum_tendency(
                state.atmosphere,
                stress_on_ocean_edge=ice_result.stress_on_ice_from_air_edge,
            )
            ocean_rhs = ocean_rhs + self._ocean_tendency_from_ice_stress(
                state.ocean,
                ice_result.stress_on_ice_from_ocean_edge,
            )
        tendency = CoupledState(atmosphere_rhs, ocean_rhs, attrs=state._attrs_copy())
        return evaluation, step, tendency

    def _apply_sea_ice_momentum(
        self,
        step: SurfaceStepResult,
        state: CoupledState,
        dt: float,
    ) -> tuple[SurfaceStepResult, SeaIceEVPStepResult]:
        """Apply native EVP and append explicit external momentum transfers."""

        assert self.sea_ice_momentum is not None and step.state.sea_ice is not None
        ice_result = self.sea_ice_momentum.step(
            step.state.sea_ice,
            mesh=state.atmosphere.mesh,
            atmosphere_u_e=state.atmosphere.u_e[..., -1:],
            ocean_u_e=state.ocean.u_e[..., :1],
            dt=float(dt),
        )
        transfers = (*step.transfers, *self._ice_momentum_transfers(ice_result, state.atmosphere.mesh, float(dt)))
        diagnostics = dict(step.diagnostics)
        diagnostics.update(ice_result.diagnostics)
        return (
            SurfaceStepResult(
                step.state.with_components(sea_ice=ice_result.state),
                step.ocean_fluxes,
                transfers,
                diagnostics,
            ),
            ice_result,
        )

    @staticmethod
    def _ice_momentum_transfers(
        result: SeaIceEVPStepResult,
        mesh: Any,
        dt: float,
    ) -> tuple[Any, Any]:
        """Return exact air/ocean-to-ice impulse transfers for the ledger."""

        from ...surface.conservation import ConservativeTransfer

        dtype, device = result.stress_on_ice_from_air_edge.dtype, result.stress_on_ice_from_air_edge.device
        area = mesh.edge_side_area.sum(dim=-1, keepdim=True).to(dtype=dtype, device=device)
        normal = mesh.edge_normal.to(dtype=dtype, device=device)

        def impulse(stress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            vector = float(dt) * area * stress * normal
            return vector[:, 0], vector[:, 1]

        air_x, air_y = impulse(result.stress_on_ice_from_air_edge)
        ocean_x, ocean_y = impulse(result.stress_on_ice_from_ocean_edge)
        return (
            ConservativeTransfer(
                "sea_ice_air_drag",
                "atmosphere",
                "sea_ice",
                momentum_x=air_x,
                momentum_y=air_y,
            ),
            ConservativeTransfer(
                "sea_ice_ocean_drag",
                "ocean",
                "sea_ice",
                momentum_x=ocean_x,
                momentum_y=ocean_y,
            ),
        )

    def _ocean_tendency_from_ice_stress(self, ocean_state: Any, stress_on_ice: torch.Tensor) -> Any:
        """Return the equal-and-opposite top-layer ocean momentum tendency."""

        u_t = torch.zeros_like(ocean_state.u_e)
        h_edge = ocean_state.mesh.cell_to_edge(ocean_state.h)[..., :1].to(dtype=u_t.dtype, device=u_t.device)
        stress = stress_on_ice.to(dtype=u_t.dtype, device=u_t.device)
        u_t[..., :1] = -stress / (float(self.ocean_density) * h_edge.clamp_min(torch.finfo(u_t.dtype).tiny))
        if self.ocean_domain is not None:
            u_t = self.ocean_domain.apply_edge(u_t)
        return ocean_state.with_tensors(
            u_t,
            torch.zeros_like(ocean_state.h),
            torch.zeros_like(ocean_state.temp_h),
            torch.zeros_like(ocean_state.salt_h),
            None if ocean_state.tracer_h is None else torch.zeros_like(ocean_state.tracer_h),
        )

    def _surface_diagnostics(
        self,
        evaluation: Any,
        step: Any,
        *,
        source_mode: str,
        substep_seconds: float,
    ) -> dict[str, Any]:
        """Collect the common SCM, packet, and internal-transfer diagnostics."""

        diagnostics = dict(evaluation.diagnostics)
        diagnostics.update(
            {
                "surface_source_mode": str(source_mode),
                "surface_rate_substep_seconds": float(substep_seconds),
                "surface_ocean_heat_flux_max_abs": float(step.ocean_fluxes.heat_flux_to_ocean.abs().max().detach().cpu()),
                "surface_ocean_freshwater_flux_max_abs": float(
                    step.ocean_fluxes.freshwater_flux_to_ocean.abs().max().detach().cpu()
                ),
            }
        )
        transfer_ledger = TransferLedger(list(step.transfers))
        for quantity in TRANSFER_QUANTITIES:
            residual = transfer_ledger.global_residual(quantity)
            scale = torch.zeros_like(residual)
            for transfer in step.transfers:
                value = transfer.quantity(quantity)
                if value is not None:
                    scale = scale + value.abs().sum().to(dtype=scale.dtype, device=scale.device)
            diagnostics[f"surface_transfer_{quantity}_relative_residual"] = float(
                (residual.abs() / scale.clamp_min(1.0)).detach().cpu()
            )
        for key, value in step.diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                diagnostics[f"surface_{key}"] = float(value.detach().cpu())
        return diagnostics

    def _atmos_surface_flux_packet(
        self,
        outputs: Mapping[str, Any],
        surface_params: Mapping[str, torch.Tensor],
    ) -> AtmosSurfaceFluxPacket:
        reference = surface_params["surface_temperature"]

        def output(name: str) -> torch.Tensor:
            value = outputs.get(name)
            if value is None:
                raise ValueError(f"surface ARK source requires SCM output {name!r}")
            tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.shape != reference.shape:
                raise ValueError(f"SCM output {name!r} must have shape {tuple(reference.shape)}")
            return tensor

        precipitation = output("surface_precipitation_rate").clamp_min(0.0)
        temperature = torch.as_tensor(
            outputs.get("surface_temperature", surface_params["surface_temperature"]),
            dtype=reference.dtype,
            device=reference.device,
        )
        if temperature.ndim == 1:
            temperature = temperature[:, None]
        if temperature.shape != reference.shape:
            raise ValueError("SCM surface_temperature must match the surface parameter shape")
        snow = precipitation * (temperature < self.snow_temperature_k).to(dtype=reference.dtype)
        rain = precipitation - snow
        zero = torch.zeros_like(reference)
        return AtmosSurfaceFluxPacket(
            output("surface_shortwave_down"),
            output("surface_longwave_down") - output("surface_longwave_up"),
            output("sensible_heat_flux_up"),
            output("latent_heat_flux_up"),
            rain,
            snow,
            zero,
            zero,
        )

    def _ocean_tendency_from_packet(self, ocean_state: Any, packet: Any) -> Any:
        u_t = torch.zeros_like(ocean_state.u_e)
        h_t = torch.zeros_like(ocean_state.h)
        temp_h_t = torch.zeros_like(ocean_state.temp_h)
        salt_h_t = torch.zeros_like(ocean_state.salt_h)
        heat = _values(packet.heat_flux_to_ocean).to(dtype=temp_h_t.dtype, device=temp_h_t.device)
        freshwater = _values(packet.freshwater_flux_to_ocean).to(dtype=salt_h_t.dtype, device=salt_h_t.device)
        salt = _values(packet.salt_flux_to_ocean).to(dtype=salt_h_t.dtype, device=salt_h_t.device)
        temp_h_t[..., :1] = heat / (self.ocean_density * self.ocean_heat_capacity)
        salt_h_t[..., :1] = -self.reference_salinity * freshwater / self.water_density + salt / self.ocean_density
        raw = ocean_state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, None if ocean_state.tracer_h is None else torch.zeros_like(ocean_state.tracer_h))
        routed = self.scm_process.surface_exchange.route_ocean_tendency(raw)
        if self.ocean_domain is None:
            return routed
        return routed.with_tensors(
            self.ocean_domain.apply_edge(routed.u_e),
            self.ocean_domain.apply_cell(routed.h),
            self.ocean_domain.apply_cell(routed.temp_h),
            self.ocean_domain.apply_cell(routed.salt_h),
            None if routed.tracer_h is None else self.ocean_domain.apply_cell(routed.tracer_h),
        )


__all__ = [
    "CoupledSurfaceSubcycleResult",
    "CoupledSurfaceSystemStageSource",
    "SurfaceARKMemoryCodec",
]
