"""Roquet ocean for standalone forcing and coupled ocean–ice integration."""
import torch
from .prognostics import OceanState
from .layers import LayerGrid
from .conservative import ConstantDensityDynamics
from .transport import HorizontalTransport
from .temperature.model import LayerDynamics, layer_fields
from .temperature.shelf_transport import PartialCellDynamics
from .temperature.ale import ConstantHeatCapacityRemapper, TemperatureEnergy
from .temperature.roquet import (RoquetEnergy, RoquetCaloric, RoquetEntropyEnergy,
                                RoquetFreezingEOS, RoquetMixing, RoquetHeating, RoquetExchange)
from .temperature.eddy_energy import EddyState, RoquetHamiltonian
from .stratified.tke_model import TKEStepReport


class LayerMechanics:
    """Rotation and physical-state checks using the selected Roquet energy."""
    def __init__(self, domain, energy):
        self.domain = domain
        self.gravity = energy.gravity
        self.rho0 = energy.rho0
        self.energy = energy
        self.mechanical = ConstantDensityDynamics(domain=domain, gravity=self.gravity, rho0=self.rho0)
        self.min_thickness = self.mechanical.min_thickness

    def check_state(self, state):
        self.mechanical._check_state(state)
        for name in ('u_e', 'h', 'temp_h', 'salt_h', 'tracer_h'):
            value = getattr(state, name)
            if value is not None and not torch.isfinite(value).all().item():
                raise FloatingPointError(f'Roquet ocean {name} must be finite')
        if (state.salt_h < 0).any().item():
            raise FloatingPointError('Roquet ocean requires nonnegative salinity; no clipping')
        rho = self.energy.density(state)
        if not torch.isfinite(rho).all().item() or (rho <= 0).any().item():
            raise FloatingPointError('Roquet ocean density must be finite and positive')


class RoquetLayers:
    """Geometry, transport and thermodynamics used by layer-local operators."""
    def __init__(self, mesh, column, domain, config):
        self.mesh, self.column, self.domain = mesh, column, domain
        energy = RoquetEnergy(config, gravity=column.gravity, rho0=column.rho0, domain=domain)
        mechanics = LayerMechanics(domain, energy)
        self.base = LayerGrid(mesh, column, domain, mechanics)
        self.base.sources = ()
        transport = HorizontalTransport(mesh, domain=domain, scheme='donor',
                                        min_thickness=mechanics.min_thickness)
        transport.prepare(column.n_layers)
        self.base._stratified_transport = transport
        self.reference = RoquetCaloric(config)
        self.energy = RoquetEntropyEnergy(energy, self.reference)

    def check_state(self, state, *, _physical=None):
        layer_fields(self, state)
        return state


class Ocean:
    """Active Conservative Temperature and Absolute Salinity on material layers."""
    def __init__(self, mesh, column, *, config=None, domain=None, closure=None, mixing=None,
                 boundary=None, heating=None, exchange=None, backscatter=None, shelf_transport=False):
        from .temperature.backscatter import EddyBackscatter

        config = RoquetFreezingEOS() if config is None else config
        if mixing is not None and closure is not None:
            raise ValueError("supply mixing or closure once")
        self.closure = mixing if mixing is not None else closure
        if not isinstance(config, RoquetFreezingEOS):
            raise TypeError("RoquetFreezingEOS required")
        for value, kind in ((self.closure, RoquetMixing), (heating, RoquetHeating),
                            (exchange, RoquetExchange), (backscatter, EddyBackscatter)):
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"expected {kind.__name__}")
        if shelf_transport and domain is None:
            raise ValueError("shelf transport requires a wet ocean domain")
        self.mesh, self.column, self.domain = mesh, column, domain
        self.config = config
        self.layers = RoquetLayers(mesh, column, domain, config)
        self.dynamics = PartialCellDynamics(self.layers) if shelf_transport else LayerDynamics(self.layers)
        self.energy = TemperatureEnergy(self.layers)
        self.hamiltonian = RoquetHamiltonian(self.energy)
        self.boundary, self.heating, self.exchange = boundary, heating, exchange
        self.backscatter = backscatter
        self.remapper = ConstantHeatCapacityRemapper()

    @property
    def rotation_rate(self):
        return self.layers.base.dynamics.mechanical.rotation_rate

    @property
    def ocean(self):
        """Layer operators used by older standalone experiments."""
        return self.layers

    @property
    def mixing(self):
        return self.closure

    def compile_thermodynamics(self, *, backend='inductor'):
        """Compile selected pure kernels; validation and timestep control stay eager."""
        previous = getattr(self, "_compile_backend", None)
        if previous is not None:
            if previous != backend:
                raise ValueError("rebuild the model to change compilation backend")
            return self
        options = dict(backend=backend, fullgraph=True, dynamic=True)
        energy = self.hamiltonian.base
        energy.stratification = torch.compile(energy.stratification, **options)
        energy.parcel_potential = torch.compile(energy.parcel_potential, **options)
        energy.coefficients = torch.compile(energy.coefficients, **options)
        caloric = self.layers.reference
        caloric._values = torch.compile(caloric._values, **options)
        self._compile_backend = backend
        return self

    def state(self, *, u_e, h, conservative_temperature, absolute_salinity,
              tracer=None, tke=0., eddy_energy=None):
        """Construct layer inventories from CT (°C), SA (g/kg) and energies (J/kg)."""
        layers = self.layers.base.state(u_e=u_e, h=h, temperature=conservative_temperature,
                                       salinity=absolute_salinity, tracer=tracer)
        turbulent = h * torch.as_tensor(tke, dtype=h.dtype, device=h.device)
        state = OceanState.from_layers(layers, turbulent)
        self.check(state)
        if eddy_energy is not None:
            state = self.with_eddy_reservoir(state, eddy_energy)
        return state

    def total_energy(self, state):
        return self.hamiltonian.terms(state)["total"]

    def check(self, state):
        self.dynamics.fields(state.ocean)
        k, h = state.tke_h, state.ocean.h
        if (k.shape != h.shape or k.dtype != h.dtype or k.device != h.device
                or not torch.isfinite(k).all() or (k < 0).any()):
            raise ValueError("TKE must match h and be finite/nonnegative")
        wet = (self.energy.mass_area(state.ocean) > 0).expand_as(h) & (h > 0)
        if (k[~wet] != 0).any():
            raise ValueError("dry TKE must be zero")
        self._check_eddy(state)

    def _check_eddy(self, state):
        """Validate the reservoir when all other fields were checked unchanged."""
        if isinstance(state, EddyState):
            e, h = state.eddy_h, state.ocean.h
            if (
                e.shape != h.shape
                or e.dtype != h.dtype
                or e.device != h.device
                or not torch.isfinite(e).all()
                or (e < 0).any()
            ):
                raise ValueError(
                    "eddy reservoir must match h and be finite/nonnegative"
                )
            wet = (self.energy.mass_area(state.ocean) > 0) & (h > 0)
            if (e[~wet] != 0).any():
                raise ValueError("dry eddy reservoir must be zero")

    def with_eddy_reservoir(self, state, specific_energy):
        """Explicitly initialize e [J/kg]; never invent or reset a hidden seed."""
        if isinstance(state, EddyState):
            raise TypeError("state already has an eddy reservoir")
        self.check(state)
        h = state.ocean.h
        e = h * torch.as_tensor(specific_energy, dtype=h.dtype, device=h.device)
        wet = (self.energy.mass_area(state.ocean) > 0).expand_as(h)
        out = EddyState.from_layers(
            state.ocean, state.tke_h, torch.where(wet, e, torch.zeros_like(e))
        )
        self.check(out)
        return out

    @staticmethod
    def pack_transport(state):
        """One bounded carrier for passive tracers, optional EKE, then TKE."""
        layers = state.ocean
        contents = [] if layers.tracer_h is None else [layers.tracer_h]
        if isinstance(state, EddyState):
            contents.append(state.eddy_h[..., None])
        contents.append(state.tke_h[..., None])
        packed = torch.cat(contents, dim=-1) if len(contents) > 1 else contents[0]
        return layers.with_tensors(layers.u_e, layers.h, layers.temp_h, layers.salt_h, packed)

    @staticmethod
    def unpack_transport(packed, original):
        has_eddy = isinstance(original, EddyState)
        count = 2 if has_eddy else 1
        passive = None if original.ocean.tracer_h is None else packed.tracer_h[..., :-count]
        layers = packed.with_tensors(packed.u_e, packed.h, packed.temp_h, packed.salt_h, passive)
        tke = packed.tracer_h[..., -1]
        if has_eddy:
            return EddyState.from_layers(layers, tke, packed.tracer_h[..., -2])
        return OceanState.from_layers(layers, tke)

    def step_with_report(self, state, dt, *, surface_fluxes=None, surface_stress=None,
                         shortwave=None, surface_heat=None):
        """Standalone split step. Coupled ice calls the stage dynamics directly."""
        self.check(state)
        packed = self.pack_transport(state)
        reports = []
        if self.backscatter is None:
            transported = self.dynamics.step(packed, dt)
        else:
            if not isinstance(state, EddyState):
                raise TypeError("enabled backscatter requires EddyState")
            if dt * self.backscatter.fraction > self.backscatter.timescale:
                raise ValueError("backscatter source positivity requires dt*fraction <= timescale")

            def backscatter_stage(stage):
                fluid = self.unpack_transport(stage, state)
                du, de, work = self.backscatter.tendency(self, fluid)
                reports.append(work)
                contents = stage.tracer_h
                dq = torch.cat((torch.zeros_like(contents[..., :-2]), de[..., None],
                                torch.zeros_like(contents[..., -1:])), dim=-1)
                return du, dq

            transported = self.dynamics.step(packed, dt, extra_tendency=backscatter_stage)

        # The surface and mixing operators carry EKE with the passive contents.
        # TKE participates independently in the mixing budget.
        has_transport_tracers = isinstance(state, EddyState) or state.ocean.tracer_h is not None
        tracers = transported.tracer_h[..., :-1] if has_transport_tracers else None
        fluid = transported.with_tensors(transported.u_e, transported.h, transported.temp_h,
                                        transported.salt_h, tracers)
        transported_tke = transported.tracer_h[..., -1]
        tke = transported_tke
        exchange = boundary = heating = mixing = None

        if surface_fluxes is not None:
            if self.exchange is None:
                raise ValueError("surface fluxes require exchange")
            fluid, exchange = self.exchange.apply(self.layers, fluid, dt, surface_fluxes,
                                                  _physical=fluid)
        if self.boundary is not None:
            fluid, tke, boundary = self.boundary.apply(self.layers, fluid, tke, dt,
                                                       surface_stress=surface_stress, _physical=fluid)
        elif surface_stress is not None:
            raise ValueError("stress requires boundary")
        if self.heating is not None:
            fluid, heating = self.heating.apply(self.layers, fluid, dt, shortwave=shortwave,
                                               surface_heat=surface_heat, _physical=fluid)
        elif shortwave is not None or surface_heat is not None:
            raise ValueError("heat flux requires heating")
        if self.closure is not None:
            fluid, tke, mixing = self.closure.apply(self.layers, fluid, tke, dt)

        if isinstance(state, EddyState):
            eddy = fluid.tracer_h[..., -1]
            passive = None if state.ocean.tracer_h is None else fluid.tracer_h[..., :-1]
            fluid = fluid.with_tensors(fluid.u_e, fluid.h, fluid.temp_h, fluid.salt_h, passive)
            end = EddyState.from_layers(fluid, tke, eddy)
        else:
            end = OceanState.from_layers(fluid, tke)
        self.check(end)
        residual = (self.energy.mass_area(state.ocean) * (transported_tke - state.tke_h)).sum()
        report = TKEStepReport(None, mixing, residual, boundary, heating, exchange)
        if self.backscatter is not None:
            from .temperature.backscatter import BackscatterStepReport
            report = BackscatterStepReport(**vars(report), backscatter=tuple(reports))
        return end, report

    def remap_with_report(self, state, target_h):
        self.check(state)
        mapped, report = self.remapper.remap_with_report(
            self.layers.base, self.pack_transport(state), target_h, self.energy
        )
        end = self.unpack_transport(mapped, state)
        self.check(end)
        return end, report

    def step(self, state, dt, **forcing):
        return self.step_with_report(state, dt, **forcing)[0]
