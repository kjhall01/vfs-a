"""Opt-in PyAM21 integration. No package monkeypatching or forward autograd.

This is an audited finite physics update, not a claim that AM2 conserves VFS H.
Water sources change actual atmospheric mass; dry-air inventory is retained.
Energy transfer preserves AM2's own imbalance and reports it separately.
"""

from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path
import torch
import pyam21.constants as am2_constants
from pyam21.physics.am2 import AM2Physics
from pyam21.physics.moist import MoistProcesses
from pyam21.time_manager import set_date, set_time
from vfs.atmosphere.moist_conservative import MoistState
from vfs.atmosphere.moist_cp import interface_pressure_mass
from .mapping import ColumnMap, average, distribute, species_increment, species_endpoint
from .state import PhysicsState, PhysicsReport, copy_tree
from .energy import exchanges, transfer_energy
from .surface import ColumnTiles, fund_column_evaporation, fund_ice_sublimation, surface_layer, tile_fluxes
from .vapor import fund_interface_vapor, restore_positive_vapor

PYAM21_REVISION = "e1bdfcc3b23c05adea80b35af880702bb85718eb"
RAD_FIELDS = ("lw_alarm", "sw_alarm", "saved", "coszen")


class _CloudStageMoistProcesses(MoistProcesses):
    """Retain the cloud-area operands from the actual upstream calls."""

    def __call__(self, *args, **kwargs):
        stages = {}

        def ras(*args, **kwargs):
            result = self.ras(*args, **kwargs)
            stages["ras_end"] = result["qa"]
            stages["ras_rate"] = result["qatnd"]
            return result

        def strat(*args, **kwargs):
            result = self.strat(*args, **kwargs)
            stages["strat_change"] = result["SA"]
            return result

        # The public driver owns the ordering and all arithmetic. The two
        # callbacks retain tensor references only for this call.
        driver = MoistProcesses(self.c, ras, strat)
        result = driver(*args, **kwargs)
        result["cloud_stages"] = stages
        return result


def _consistent_cloud_stages(initial, rate, endpoint, dt, stages):
    """Check each way PyAM21 accumulates cloud area, including a RAS round trip."""
    ras_rate, ras_end = stages["ras_rate"], stages["ras_end"]
    strat_change, turbulent_rate = stages["strat_change"], stages["turbulent_rate"]
    start = initial + dt * turbulent_rate
    ras_change = dt * ras_rate
    # RAS forms start + change and change * (1/dt) separately. Rebuilding
    # the former uses a reciprocal, two products and an addition. Bound
    # roundoff by the local operands of those operations.
    scale = torch.maximum(start.abs(), ras_change.abs())
    ras_tol = (8 * torch.finfo(initial.dtype).eps) * scale
    return (
        torch.isfinite(ras_change) & torch.isfinite(start) & torch.isfinite(ras_end)
        & ((ras_end - (start + ras_change)).abs() <= ras_tol)
        # Both published results must reproduce their respective upstream
        # expressions exactly, even where a large RAS term cancels in strat.
        & (endpoint == ras_end + strat_change)
        & (rate == (turbulent_rate + ras_rate) + strat_change * (1.0 / dt))
    )


def _resolve_cloud_roundoff(initial, tracers, out, dt, bad_cloud, cutoff):
    """Return a copy only for verified RAS/stratiform cancellation at a cloud bound."""
    cells, levels = bad_cloud.nonzero(as_tuple=True)
    raw = tracers[cells, levels, 3]
    stages = {k: v[cells, levels] for k, v in out["cloud_stages"].items()}
    ras = stages["ras_end"]
    strat = stages["strat_change"]
    magnitude = torch.maximum(ras.abs(), strat.abs())
    spacing = torch.nextafter(magnitude, torch.full_like(magnitude, torch.inf)) - magnitude
    bounded = raw.clamp(0, 1)
    # Allow one unit in the last place of the larger operand, provided that
    # this uncertainty is below StratCloud's smallest resolved cloud fraction.
    accepted = (
        ((ras > 0) & (strat < 0) | (ras < 0) & (strat > 0))
        & ((raw - bounded).abs() <= spacing) & (spacing <= cutoff)
        & _consistent_cloud_stages(initial[cells, levels], out["rdt"][cells, levels, 3],
                                   raw, dt, stages)
    )
    if not accepted.any():
        return tracers, None
    cells, levels = cells[accepted], levels[accepted]
    adjusted = tracers.clone()
    adjusted[cells, levels, 3] = bounded[accepted]
    return adjusted, dict(indices=torch.stack((cells, levels), dim=-1),
                          raw=raw[accepted], applied=bounded[accepted])


def _inadmissible_column_detail(temperature, tracers, out, eps, condensate_tol):
    """Report a rejecting predicate. Finite negative vapor is handled later."""
    for name in ("tdt", "rdt", "udt", "vdt"):
        rate = out[name]
        bad = ~torch.isfinite(rate)
        if bad.any():
            index = tuple(int(i) for i in bad.nonzero()[0])
            return (f"field={name} tendency index={index}"
                    f" value={float(rate[index]):.17g}")
    if not torch.isfinite(tracers).all():
        cell, layer, species = (~torch.isfinite(tracers)).nonzero()[0].tolist()
        field = ("vapor_mass_fraction", "liquid_mass_fraction", "ice_mass_fraction", "cloud_fraction")[species]
        return f"field={field} cell={cell} level={layer} value={float(tracers[cell, layer, species])} required=finite"
    checks = (
        ("temperature_K", temperature, ~torch.isfinite(temperature) | (temperature <= 0), 0.0),
        ("liquid_mass_fraction", tracers[..., 1],
         tracers[..., 1] < -condensate_tol[0], -float(condensate_tol[0])),
        ("ice_mass_fraction", tracers[..., 2],
         tracers[..., 2] < -condensate_tol[1], -float(condensate_tol[1])),
        ("total_water_fraction", tracers[..., :3].sum(-1),
         tracers[..., :3].sum(-1) >= 1, 1.0),
        ("cloud_fraction", tracers[..., 3],
         (tracers[..., 3] < -128 * eps) | (tracers[..., 3] > 1 + 128 * eps),
         None),
    )
    for name, values, bad, bound in checks:
        if bad.any():
            cell, layer = (int(v) for v in bad.nonzero()[0])
            value = float(values[cell, layer])
            threshold = (
                f" required_bound={bound:.17g}" if bound is not None
                else f" allowed_range=[{-128 * eps:.17g},{1 + 128 * eps:.17g}]"
            )
            return (
                f"field={name} cell={cell} level={layer}"
                f" value={value:.17g}{threshold}"
            )
    return None


class AM2Adapter:
    def __init__(
        self,
        model,
        namelist,
        *,
        start=(2000, 1, 1, 0, 0, 0),
        radiation=None,
        radiation_id=None,
        hprime=0.0,
        energy_mode="conservative",
    ):
        self.model = model
        self.logger = logging.getLogger(__name__)
        if energy_mode != "conservative":
            raise ValueError("energy_mode must be conservative")
        self.energy_mode = energy_mode
        ref = model.mesh.cell_area
        from workflows.coupled.device import validate_precision
        validate_precision(ref.device, ref.dtype)
        if model.dynamics.kernel.gravity != am2_constants.GRAV:
            raise ValueError(
                "configure the host kernel gravity to PyAM21 GRAV=9.8 before initializing state"
            )
        self.mapping = ColumnMap(model)
        self.start = set_date(*start)
        self.radiation = radiation
        if radiation is not None and (
            not isinstance(radiation_id, str) or not radiation_id
        ):
            raise ValueError(
                "radiation_id must identify the package and input-data/configuration manifest for restart validation"
            )
        xyz = model.mesh.cell_xyz
        lat = torch.asin(xyz[:, 2])
        lon = torch.atan2(xyz[:, 1], xyz[:, 0])
        self.physics = AM2Physics(
            namelist,
            lat,
            lon,
            hprime=hprime,
            radiation=None if radiation is None else radiation.for_physics(),
        ).to(device=ref.device, dtype=ref.dtype)
        moist = self.physics.moist
        if (
            moist.ras is not None
            and moist.strat is not None and not moist.c.use_tau
        ):
            self.physics.moist = _CloudStageMoistProcesses(moist.c, moist.ras, moist.strat)
        # Pin the restart to actual installed source, configuration and geometry.
        root = Path(am2_constants.__file__).parent
        source = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
        config = dict(
            namelist=namelist,
            start=list(start),
            source=source,
            hprime=float(hprime),
            thermo=asdict(model.dynamics.thermodynamics),
            gravity=model.dynamics.kernel.gravity,
            radiation=None if radiation is None else asdict(radiation.c),
            radiation_id=radiation_id,
            energy_mode=energy_mode,
            schema=4,
        )
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for field in (
            model.mesh.cell_xyz,
            model.mesh.cellsOnEdge,
            model.mesh.cell_area,
            model.column.A_interface,
            model.column.B_interface,
            model.column.phi_s,
        ):
            digest.update(field.detach().cpu().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        self.configuration = config

    def check(self, state):
        self.model._check_state(state.atmosphere)
        self.model.dynamics.transport_state(state.atmosphere)
        cloud = state.cloud_fraction
        expected = state.atmosphere.entropy_content
        cloud_tolerance = 128 * torch.finfo(expected.dtype).eps
        if (
            cloud.shape != expected.shape
            or cloud.dtype != expected.dtype
            or cloud.device != expected.device
            or not torch.isfinite(cloud).all()
            or (cloud < -cloud_tolerance).any()
            or (cloud > 1 + cloud_tolerance).any()
        ):
            raise ValueError(
                "cloud fraction must match CP shape and lie in [0,1] apart from roundoff"
            )
        if type(state.time_seconds) is not int or state.time_seconds < 0:
            raise ValueError("time_seconds must be a nonnegative integer")

    def initialize(self, atmosphere, *, sst, cloud_fraction=None):
        c, i = atmosphere.entropy_content.shape
        ref = atmosphere.entropy_content
        if (
            not isinstance(sst, torch.Tensor)
            or sst.shape != (c,)
            or sst.dtype != ref.dtype
            or sst.device != ref.device
        ):
            raise ValueError(
                "prescribed SST must be a matching [cell] tensor in kelvin"
            )
        if not torch.isfinite(sst).all() or (sst <= 0).any():
            raise ValueError("positive finite SST required")
        surface = self.physics.init_ocean_surface(c, dtype=ref.dtype, device=ref.device)
        surface["t_surf"] = sst.clone()
        radiation = (
            {}
            if self.radiation is None
            else {k: copy_tree(getattr(self.radiation, k)) for k in RAD_FIELDS}
        )
        state = PhysicsState(
            atmosphere,
            torch.zeros_like(ref) if cloud_fraction is None else cloud_fraction.clone(),
            self.physics.init_memory(c, i - 1, dtype=ref.dtype, device=ref.device),
            surface,
            radiation,
            0,
        )
        self.check(state)
        return state

    def pack(self, state):
        """Explicit tensor/primitive payload, compatible with weights_only=True."""
        self.check(state)
        return dict(
            fingerprint=self.fingerprint,
            atmosphere={
                k: getattr(state.atmosphere, k).clone()
                for k in ("u_e", "ps", "entropy_content", "water_content")
            },
            cloud_fraction=state.cloud_fraction.clone(),
            memory=copy_tree(state.memory),
            surface=copy_tree(state.surface),
            radiation=copy_tree(state.radiation),
            time_seconds=state.time_seconds,
        )

    def unpack(self, payload, *, verified_grid=False):
        """The coupled NetCDF reader can independently validate the checkpoint grid."""
        if not verified_grid and payload["fingerprint"] != self.fingerprint:
            raise ValueError(
                "restart geometry/configuration/PyAM21 source does not match"
            )
        p = copy_tree(payload)
        state = PhysicsState(
            MoistState(
                **p["atmosphere"], mesh=self.model.mesh, column=self.model.column
            ),
            p["cloud_fraction"],
            p["memory"],
            p["surface"],
            p["radiation"],
            p["time_seconds"],
        )
        self.check(state)
        return state

    def _fund_interface_vapor(self, water, mass):
        transport = self.model.transport
        if transport is None:
            return water, None
        return fund_interface_vapor(water, mass, transport.vapor_floor, self.model.dynamics.kernel.gravity)

    def _warn_interface_vapor(self, correction, seconds, context):
        cell, interface = correction["indices"][0].tolist()
        self.logger.warning(
            "Day %.6f: native vapor floor funded during %s; interfaces=%d, "
            "first cell=%d interface=%d, minimum incoming vapor %.6g, floor %.6g kg/kg",
            seconds / 86400, context, len(correction["indices"]), cell, interface,
            correction["minimum_before"], correction["floor"],
        )

    def fund_restart_vapor(self, state):
        """Bring an older restart across the same CP floor at fixed native layer energy."""
        s = state.atmosphere
        dp = s.column.interface_pressure(s.ps).diff()
        water, correction = self._fund_interface_vapor(s.water_content, interface_pressure_mass(dp))
        if correction is None:
            return state, None
        end, energy = transfer_energy(self, s, replace(s, water_content=water), torch.zeros_like(dp))
        result = replace(state, atmosphere=end)
        self.check(result)
        correction["layer_energy_residual_J_m2"] = energy["layer_residual"]
        self._warn_interface_vapor(correction, state.time_seconds, "restart")
        return result, correction

    def column_call(self, state, dt, columns, geometry, *, surface_tiles=None):
        """Call all four upstream stages, retaining their actual exchange outputs.

        surf/mem are copied. Radiation alarms/cache are installed temporarily,
        copied into the returned state and restored even after an exception.
        An adapter instance is intentionally not reentrant/thread-safe.
        """
        surf, mem = copy_tree(state.surface), copy_tree(state.memory)
        original_rad = None
        if self.radiation is not None:
            original_rad = {k: getattr(self.radiation, k) for k in RAD_FIELDS}
            for k in RAD_FIELDS:
                setattr(self.radiation, k, copy_tree(state.radiation[k]))
        try:
            p = self.physics
            T, u, v, R, ps = (columns[k] for k in ("T", "u", "v", "tracers", "ps"))
            pf, ph, zf, zh, omega = (
                geometry[k] for k in ("pfull", "phalf", "zfull", "zhalf", "omega")
            )
            zero = torch.zeros_like(T)
            tile_raw = None
            if surface_tiles is None:
                sfc = p.surface_layer(
                    T,
                    u,
                    v,
                    R,
                    ps,
                    zf[:, -1] - zh[:, -1],
                    pf[:, -1],
                    surf["t_surf"],
                    surf,
                    mem,
                )
            else:
                sfc, tile_raw = surface_layer(
                    self, columns, geometry, surf, mem, surface_tiles
                )
            bad_surface = [k for k, v in sfc.items() if not torch.isfinite(v).all()]
            if bad_surface:
                raise FloatingPointError(
                    "PyAM21 surface layer returned nonfinite "
                    + ", ".join(bad_surface)
                    + f"; minimum gust={float(mem['gust'].min()):g} m/s; no state committed"
                )
            dn = p.down(
                dt,
                self.start + set_time(state.time_seconds),
                T,
                u,
                v,
                R,
                pf,
                ph,
                zf,
                zh,
                zero,
                zero,
                zero,
                torch.zeros_like(R),
                sfc,
                surf,
                mem,
                surf["t_surf"],
            )
            cpl = p.couple(dn["vd"], sfc, surf, update_ocean=surface_tiles is None)
            if isinstance(surface_tiles, ColumnTiles):
                cpl, tile_raw = fund_column_evaporation(p, dn["vd"], surf, sfc, cpl, surface_tiles, dt, tile_raw)
            ice_limits = {}
            if surface_tiles is not None and surface_tiles.frozen_mass is not None:
                cpl, tile_raw, ice_limits = fund_ice_sublimation(
                    p, dn["vd"], surf, sfc, cpl, surface_tiles, dt, tile_raw
                )
            up = p.up(dt, T, u, v, R, pf, ph, zf, zh, omega, dn, cpl, surf, mem)
            out = {k: up[k] for k in ("udt", "vdt", "tdt", "rdt")}
            out.update(ice_limits)
            out.update(
                lprec=up["moist"]["lprec"],
                fprec=up["moist"]["fprec"],
                flux_t=cpl["flux_t"],
                flux_q=cpl["flux_q"],
                flux_u=cpl["flux_u"],
                flux_v=cpl["flux_v"],
                radiation=dn["diag"].get("radiation"),
                endpoint_temperature=up["moist"]["tin"],
                endpoint_tracers=torch.stack(
                    [
                        up["moist"][k]
                        for k in ("qin", "lsc_liquid", "lsc_ice", "lsc_cloud_area")
                    ],
                    -1,
                ),
            )
            if "cloud_stages" in up["moist"]:
                out["cloud_stages"] = dict(
                    up["moist"]["cloud_stages"], turbulent_rate=up["up_rdt"][..., 3]
                )
            bad_column = [
                k for k, v in out.items()
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
            ]
            if bad_column:
                raise FloatingPointError(
                    "PyAM21 column physics returned nonfinite "
                    + ", ".join(bad_column) + "; no state committed"
                )
            if tile_raw is not None:
                out["tile_fluxes"] = tile_fluxes(tile_raw, surface_tiles, dn["vd"], cpl)
                for key, value in out["tile_fluxes"].items():
                    error = (value.sum(-1) - out[key]).abs()
                    if (
                        error
                        > 256
                        * torch.finfo(value.dtype).eps
                        * value.abs().sum(-1).clamp_min(1)
                    ).any():
                        raise FloatingPointError(
                            "tile and aggregate implicit fluxes disagree"
                        )
            rad = (
                {}
                if self.radiation is None
                else {k: copy_tree(getattr(self.radiation, k)) for k in RAD_FIELDS}
            )
            return out, surf, mem, rad
        finally:
            if original_rad is not None:
                for k, value in original_rad.items():
                    setattr(self.radiation, k, value)

    def apply(self, state, dt, *, omega=None, water_boundary=None):
        """Advance physics once; dt is a positive whole number of seconds.

        Dynamics have already been advanced by the caller, so their tendencies
        are zero in PyAM21. Surface flux signs are positive into the atmosphere;
        precipitation is positive downward. The first pass supports stationary,
        prescribed-SST open ocean only.
        """
        self.check(state)
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not float(dt).is_integer()
            or dt <= 0
        ):
            raise ValueError("PyAM21 physics dt must be positive integer seconds")
        dt = int(dt)
        if state.surface["land"].any():
            raise ValueError("this adapter supports prescribed-SST open ocean only")
        sst = state.surface["t_surf"]
        if (
            sst.shape != state.atmosphere.ps[:, 0].shape
            or not torch.isfinite(sst).all()
            or (sst <= 0).any()
        ):
            raise ValueError(
                "prescribed SST must remain finite, positive and cell-shaped"
            )
        s = state.atmosphere
        column, geom = self.mapping.columns(s, state.cloud_fraction, omega)
        out, surf, mem, rad = self.column_call(state, dt, column, geom)
        return self.apply_result(
            state, dt, column, geom, out, surf, mem, rad, water_boundary=water_boundary
        )

    def apply_result(
        self, state, dt, column, geom, out, surf, mem, rad, *, water_boundary=None
    ):
        """Apply a checked upstream result; also permits isolated mapping tests."""
        if water_boundary is not None and out.get("exchange") is not None:
            raise ValueError("shared exchange owns water boundary")
        s = state.atmosphere
        model, mapping = self.model, self.mapping
        thermo = model.dynamics.thermodynamics
        g = model.dynamics.kernel.gravity
        p = geom["phalf"]
        dp, mass = p.diff(), interface_pressure_mass(p.diff())
        T1, R1 = column["T"] + dt * out["tdt"], column["tracers"] + dt * out["rdt"]
        eps = torch.finfo(T1.dtype).eps
        if "endpoint_tracers" in out:
            Tend, Rend = out["endpoint_temperature"], out["endpoint_tracers"]
            for actual, expected in ((Tend, T1), (Rend, R1)):
                if (
                    actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or actual.device != expected.device
                    or not torch.isfinite(actual).all()
                    or not torch.isfinite(expected).all()
                ):
                    raise ValueError("invalid explicit PyAM21 endpoint or tendency reconstruction")
            # The port exposes its actual endpoint as well as tendencies. Use
            # it to avoid reconstructing an exhausted store by cancellation.
            # Available water bounds intermediate liquid/vapor/ice transfers.
            water_scale = torch.maximum(
                column["tracers"][..., :3].abs().sum(-1).amax(),
                Rend[..., :3].abs().sum(-1).amax(),
            )
            tolerance = torch.cat((water_scale.expand(3), water_scale.new_ones(1))) * (
                128 * eps
            )
            temp_error = (Tend - T1).abs()
            temp_tol = 128 * eps * column["T"].abs().amax()
            tracer_error = (Rend - R1).abs()
            bad_temp = temp_error > temp_tol
            bad_tracer = tracer_error > tolerance
            if bad_temp.any() or bad_tracer.any():
                if "cloud_stages" in out and isinstance(self.physics.moist, _CloudStageMoistProcesses):
                    # StratCloud itself clears cloud fractions smaller than qmin.
                    # Even genuine numerical cancellation must not erase a
                    # physically resolved difference between the two results.
                    below_cloud_cutoff = tracer_error[..., 3] <= self.physics.moist.strat.c.qmin
                    consistent = _consistent_cloud_stages(
                        column["tracers"][..., 3], out["rdt"][..., 3],
                        Rend[..., 3], dt, out["cloud_stages"],
                    )
                    bad_tracer[..., 3] &= ~(below_cloud_cutoff & consistent)
                if bad_temp.any() or bad_tracer.any():
                    if bad_temp.any():
                        cell, level = bad_temp.nonzero()[0].tolist()
                        field, error, allowed = "temperature", temp_error[cell, level], temp_tol
                    else:
                        cell, level, tracer = bad_tracer.nonzero()[0].tolist()
                        field = ("vapor", "liquid", "ice", "cloud_fraction")[tracer]
                        error, allowed = tracer_error[cell, level, tracer], tolerance[tracer]
                    raise ValueError(
                        "PyAM21 endpoint and tendencies disagree beyond roundoff"
                        f"; field={field} cell={cell} level={level} "
                        f"absolute_error={float(error):.8g} tolerance={float(allowed):.8g}"
                    )
            T1, R1 = Tend, Rend
        if not all(
            torch.isfinite(out[k]).all()
            for k in ("flux_t", "flux_q", "flux_u", "flux_v", "lprec", "fprec")
        ):
            raise ValueError("nonfinite PyAM21 boundary flux")
        # Upstream finite tendency addition can yield 1+~1e-14 cloud fraction
        # or signed condensate roundoff at an exhausted store. Admit and report
        # these unchanged, consistently with VFS's existing condensate policy.
        condensate_tolerance = 32 * eps * R1[..., 1:3].abs().amax((0, 1))
        cloud_roundoff = None
        cloud = R1[..., 3]
        bad_cloud = (cloud < -128 * eps) | (cloud > 1 + 128 * eps)
        if (
            not all(torch.isfinite(out[k]).all() for k in ("tdt", "rdt", "udt", "vdt"))
            or (~torch.isfinite(T1) | (T1 <= 0)).any()
            or not torch.isfinite(R1).all()
            or (R1[..., 1:3] < -condensate_tolerance).any()
            or (R1[..., :3].sum(-1) >= 1).any()
            or bad_cloud.any()
        ):
            if (bad_cloud.any() and "endpoint_tracers" in out and "cloud_stages" in out
                    and isinstance(self.physics.moist, _CloudStageMoistProcesses)):
                R1, cloud_roundoff = _resolve_cloud_roundoff(
                    column["tracers"][..., 3], R1, out, dt, bad_cloud, self.physics.moist.strat.c.qmin
                )
            detail = _inadmissible_column_detail(T1, R1, out, eps, condensate_tolerance)
            if detail is not None:
                raise ValueError(
                    "PyAM21 returned an inadmissible finite column update; no clipping"
                    f"; dt={dt} seconds; {detail}"
                )
        upstream_out = out
        T1, R1, out, vapor_correction = restore_positive_vapor(T1, R1, out, dt, dp=dp)
        if vapor_correction is not None:
            cell, level = vapor_correction["indices"].unbind(-1)
            layer_mass = dp[cell, level] / g
            vapor_correction.update(
                vapor_added_kg_m2=layer_mass * vapor_correction["vapor_added"],
                am2_energy_residual_J_m2=layer_mass * vapor_correction["am2_energy_residual"],
            )
            if "layer_water_transfer" in vapor_correction:
                vapor_correction["layer_water_transfer_kg_m2"] = layer_mass * vapor_correction["layer_water_transfer"]
                vapor_correction["water_residual_kg_m2"] = vapor_correction["column_water_residual_Pa"] / g
            else:
                vapor_correction["water_residual_kg_m2"] = layer_mass * vapor_correction["water_residual"]
        amount = dp[..., None] * (R1[..., :3] - column["tracers"][..., :3])
        water = species_endpoint(s.water_content, dp, amount, R1[..., :3])
        # Physics tendencies are interpreted as species amount sources on the
        # old carrier. Their sum changes ps, preserving column dry-air mass.
        ps = s.ps + amount.sum((1, 2))[:, None]
        p1 = s.column.interface_pressure(ps)
        mass1 = interface_pressure_mass(p1.diff())
        if (p1.diff() <= 0).any():
            raise ValueError("physics update exhausts CP carrier; reduce dt")
        water, interface_correction = self._fund_interface_vapor(water, mass1)
        dw = water - s.water_content
        q1 = water / mass1[..., None]
        mix1 = torch.cat((1 - q1.sum(-1, keepdim=True), q1), -1)
        # Map only the temperature increment. Never overwrite the original CP
        # profile with a lossy full-level round trip.
        Told = model.physical_fields(s)["temperature"]
        heat_increment = dp * (T1 - column["T"])
        Tnew = Told + distribute(heat_increment) / mass
        if (
            not torch.isfinite(Tnew).all()
            or (Tnew <= 0).any()
            or (mix1[..., 0] <= 0).any()
            or (q1[..., 0] <= 0).any()
        ):
            raise ValueError("physics update exhausts positive vapor required by VFS")
        entropy = mass1 * thermo.entropy(Tnew, p1, mix1)
        unchanged = (dw == 0).all(-1) & (Tnew == Told) & (p1 == p)
        entropy = torch.where(unchanged, s.entropy_content, entropy)
        # Cloud and clear fractions use complementary available stores for
        # increments, preserving [0,1] without clipping sharp boundaries.
        dc = dp * (R1[..., 3] - column["tracers"][..., 3])
        cloud = state.cloud_fraction
        positive = species_increment(1 - cloud, -dc.clamp_min(0))
        negative = species_increment(cloud, dc.clamp_max(0))
        cloud1 = cloud + (negative - positive) / mass
        trial = MoistState(s.u_e, ps, entropy, water, s.mesh, s.column)
        velocity_increment = dt * (
            out["udt"][..., None] * mapping.east[:, None]
            + out["vdt"][..., None] * mapping.north[:, None]
        )
        impulse = mapping.transpose(
            s.mesh.cell_area[:, None, None] * dp[..., None] / g * velocity_increment
        )
        metric0, metric1 = mapping.metric(s), mapping.metric(trial)
        # The epsilon transfer is internal. Keep the unmodified upstream
        # energy target (including its defects); the native solve pays for
        # the altered phase partition using the actual VFS mixture enthalpy.
        budget = exchanges(self, state, dt, column, geom, upstream_out, water_boundary)
        carrier_impulse = budget["mass_impulse"]
        u1 = (
            s.u_e
            + (impulse + carrier_impulse - (metric1 - metric0) * s.u_e) / metric1
        )
        end = replace(trial, u_e=u1)
        end, transfer = transfer_energy(self, s, end, budget["target"])
        transfer["target_layer_increment"] = budget["target"]
        result = PhysicsState(end, cloud1, mem, surf, rad, state.time_seconds + dt)
        self.check(result)
        if cloud_roundoff is not None:
            out = dict(out, cloud_area_roundoff=cloud_roundoff, endpoint_tracers=R1)
            indices = cloud_roundoff["indices"]
            cell, level = indices[0].tolist()
            self.logger.warning(
                "Day %.6f: PyAM21 cloud-area roundoff at a bound; layers=%d, "
                "first cell=%d level=%d raw=%.17g applied=%.1f",
                result.time_seconds / 86400, len(indices), cell, level,
                float(cloud_roundoff["raw"][0]), float(cloud_roundoff["applied"][0]),
            )
        actual_water = (water - s.water_content).sum(1) / g
        expected_water = amount.sum(1) / g
        if interface_correction is not None:
            changed_cell = interface_correction["changed_indices"][:, 0]
            expected_water = expected_water.index_add(
                0, changed_cell, interface_correction["species_change_Pa"] / g
            )
            self._warn_interface_vapor(interface_correction, result.time_seconds, "physics exchange")
        if vapor_correction is not None:
            self.logger.warning(
                "Day %.6f: PyAM21 vapor correction funded physics layers=%d",
                result.time_seconds / 86400,
                vapor_correction.get("recipient_count", len(vapor_correction["indices"])),
            )
        dry0 = mass.sum(-1) - s.water_content.sum((1, 2))
        dry1 = mass1.sum(-1) - water.sum((1, 2))
        fluxes = {
            k: out[k]
            for k in ("flux_t", "flux_q", "flux_u", "flux_v", "lprec", "fprec")
        }
        zero = torch.zeros_like(s.ps[:, 0])
        radiation_heat = zero
        for name in (
            "sw_net_surface",
            "lw_down_surface",
            "lw_net_surface",
            "toa_net_down",
        ):
            fluxes[name] = zero
        if out["radiation"] is not None:
            r = out["radiation"]
            fluxes.update(
                sw_net_surface=r["flux_sw_surf"],
                lw_down_surface=r["flux_lw_surf"],
                lw_net_surface=-r["flxnet"][:, -1],
                toa_net_down=r["dfsw"][:, 0] - r["ufsw"][:, 0] - r["flxnet"][:, 0],
            )
            radiation_heat = (dp * am2_constants.CP_AIR / g * r["tdt_rad"]).sum(-1)
        q0 = column["tracers"][..., :3]
        qfixed1 = R1[..., :3]
        mix0 = torch.cat((1 - q0.sum(-1, keepdim=True), q0), -1)
        mix_fixed1 = torch.cat((1 - qfixed1.sum(-1, keepdim=True), qfixed1), -1)
        vfs_column_change = (
            dp
            / g
            * (thermo.enthalpy(T1, mix_fixed1) - thermo.enthalpy(column["T"], mix0))
        ).sum(-1)
        declared_heat = dt * (
            out["flux_t"]
            + am2_constants.HLV * out["flux_q"]
            + am2_constants.HLF * out["fprec"]
            + radiation_heat
        )
        am2_proxy_change = budget["am2_thermal"]
        delta_H = mapping.energy(end) - mapping.energy(s)
        native_ke = model.dynamics.kernel.cell_kinetic_energy
        ke0 = (dp / g * native_ke(model.dynamics.transport_state(s))).sum(-1)
        ke1 = (p1.diff() / g * native_ke(model.dynamics.transport_state(end))).sum(-1)
        return result, PhysicsReport(
            fluxes,
            actual_water,
            (dry1 - dry0) / g,
            actual_water - expected_water,
            actual_water.sum(-1) - dt * (out["flux_q"] - out["lprec"] - out["fprec"]),
            dict(
                vfs_full_h=delta_H,
                native_kinetic=ke1 - ke0,
                vfs_fixed_mass_column_enthalpy=vfs_column_change,
                am2_thermal_latent_proxy=am2_proxy_change,
                declared_thermal_latent_forcing=declared_heat,
                am2_proxy_minus_forcing=am2_proxy_change - declared_heat,
                thermodynamic_convention_difference=vfs_column_change
                - am2_proxy_change,
                mapping_and_carrier_difference=delta_H
                - (ke1 - ke0)
                - vfs_column_change,
                am2_kinetic=budget["am2_kinetic"],
                am2_boundary_energy=budget["am2_boundary"],
                am2_energy_residual=budget["am2_residual"],
                boundary_datum_change=budget["datum"],
                external_energy=budget["external"],
                radiation_flux_residual=budget["radiation_residual"],
                transfer_residual=delta_H - budget["target"].sum(-1),
                full_h_minus_external=delta_H - budget["external"],
                **{"boundary_" + k: v for k, v in budget["terms"].items()},
            ),
            impulse,
            carrier_impulse,
            metric1 * u1 - metric0 * s.u_e - impulse - carrier_impulse,
            out,
            transfer,
            vapor_correction,
            interface_correction,
        )
