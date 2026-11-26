"""Coupled Roquet ocean/category-ice dynamics with one energy functional.

Research first coupling: donor ice transport, energy-adjoint pressure forces,
reciprocal vector drag to ocean TKE, prescribed wind, and compressive ridging.
The existing ocean transport/rotation is retained. No forward AD or energy fix.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.ice.ridging import CategoryIce, RidgingState, CompressionRidging
from vfs.ocean.temperature.roquet import RoquetWorkspace
from vfs.ocean.temperature.eddy_energy import EddyState
from vfs.ocean.stratified import TKEState
from vfs.ocean.energy import ConstantDensityEnergy


def safe(x):
    return torch.where(x > 0, x, torch.ones_like(x))


def vector_transpose(mesh, force):
    """Euclidean transpose of cell_wind_vector; input [cell,3] is force in N."""
    c1 = (force * mesh.cell_rec_basis1).sum(-1)
    c2 = (force * mesh.cell_rec_basis2).sum(-1)
    parts = (
        c1[:, None] * mesh.cell_rec_w1 + c2[:, None] * mesh.cell_rec_w2
    ) * mesh.edgeSignOnCell
    parts = torch.where(mesh.valid_edges_on_cell, parts, 0.0)
    return force.new_zeros((mesh.nEdges, 1)).index_add(
        0, mesh.safe_edges_on_cell.flatten(), parts.reshape(-1, 1)
    )


@dataclass(frozen=True)
class CoupledReport:
    wind_power: torch.Tensor  # W, prescribed atmospheric mechanical input
    drag_tke_power: torch.Tensor  # W, resolved loss -> ocean turbulent energy
    ridge_heat_power: torch.Tensor  # W, crushing/friction -> ocean thermal energy
    ice_transport_power: torch.Tensor  # W, diagnostic total ice transport pairing
    loading_power: torch.Tensor  # W, ocean KE work (opposite ice height work)


class OceanIceMechanics:
    """Own an ocean plus optional category-ice mechanics on the same mesh.

    State is RidgingState: existing ocean state plus CategoryIce. Ocean surface sources are owned by this wrapper. Existing vertical mixing
    and mesoscale backscatter are supported; no configured closure is silently
    dropped. Other externally scheduled operators need explicit composition.
    """

    def __init__(
        self,
        ocean,
        *,
        drag_coefficient=5e-3,
        air_drag_coefficient=1.3e-3,
        air_density=1.2,
        packing_friction=1.0,
        ridge=None,
    ):
        if any(
            getattr(ocean, k, None) is not None
            for k in ("boundary", "heating", "exchange")
        ):
            raise ValueError(
                "coupling owns surface sources; use an unforced RoquetOcean"
            )
        if any(
            not math.isfinite(x) or x < 0
            for x in (
                drag_coefficient,
                air_drag_coefficient,
                air_density,
                packing_friction,
            )
        ):
            raise ValueError("finite nonnegative exchange parameters required")
        self.ocean = ocean
        self.ridge = CompressionRidging(ocean) if ridge is None else ridge
        if self.ridge.hamiltonian.model is not ocean:
            raise ValueError("ridge must share ocean")
        self.hamiltonian = self.ridge.hamiltonian
        self.drag_coefficient = drag_coefficient
        self.air_drag_coefficient = air_drag_coefficient
        self.air_density = air_density
        self.packing_friction = packing_friction

    def check(self, state):
        try:
            self.ridge.check(state)
        except ValueError as error:
            if "nonnegative stores" in str(error):
                area, volume = state.ice.area, state.ice.volume
                coverage = area.sum(-1)
                invalid = (area < 0).any(-1) | (volume < 0).any(-1) | (coverage > 1)
                cell = invalid.nonzero()[0, 0].item()
                raise ValueError(
                    f"inadmissible coupled stage: cell={cell}, "
                    f"category areas={area[cell].tolist()}, volumes={volume[cell].tolist()}, "
                    f"coverage={coverage[cell].item():.17g}, "
                    f"coverage minus one={coverage[cell].item() - 1:.17g}"
                ) from error
            raise

    def conjugates(self, state, birth_thickness=None):
        """Cell energy derivatives per area at fixed ice velocity; omit constant latent datum."""
        H = self.hamiltonian
        s = state.ocean.ocean
        mesh = s.mesh
        rho = H.slab.parameters.density
        V = state.ice.volume.sum(-1, keepdim=True)
        h = state.ice.thickness()
        if birth_thickness is not None:
            h = torch.where(state.ice.area > 0, h, birth_thickness)
        C = H.coefficient(state)
        kv = (
            0.5
            * rho
            * ConstantDensityEnergy.interpolation_transpose(
                mesh,
                (mesh.dc_edge * mesh.dv_edge)[:, None] * state.ice.velocity.square(),
            )
            / mesh.cell_area[:, None]
        )
        gv = rho * s.column.gravity * (s.column.eta(s.h) + V) + 2 * C * (h - V) + kv
        ga = -C * h.square()
        return ga, gv

    def ice_transport(self, state):
        H = self.hamiltonian
        s = state.ocean.ocean
        mesh = s.mesh
        ice = state.ice
        ui = ice.velocity
        active = self.ridge.active_edges(state)
        c0, c1 = mesh.cellsOnEdge.T
        donor = lambda q: torch.where(ui >= 0, q[c0], q[c1])
        ta, tv = donor(ice.area), donor(ice.volume)
        da = -mesh.div(ta * ui)
        dv = -mesh.div(tv * ui)
        # H contains V^2/a. At an empty category its derivative is directional:
        # use the thickness of actual incoming material, not a fictitious zero.
        # With zero incoming flux, extend neighbor thickness for a resting edge.
        neighbor_a = mesh.edge_to_cell(mesh.cell_to_edge(ice.area))
        neighbor_v = mesh.edge_to_cell(mesh.cell_to_edge(ice.volume))
        birth = torch.where(da > 0, dv / safe(da), neighbor_v / safe(neighbor_a))
        ga, gv = self.conjugates(state, birth)
        force = -(ta * mesh.grad(ga) + tv * mesh.grad(gv)).sum(-1, keepdim=True)
        me = mesh.cell_to_edge(H.mass(state))
        du = torch.where(active, force / safe(me), 0.0)
        # Pairwise rotation exchanges no kinetic energy. Harmonic pair masses
        # make the acceleration bounded as an edge loses its ice: each pair
        # mass divided by either participating mass is at most two. A raw
        # dual PV (vorticity/mass) stencil is singular at a moving ice margin.
        # For uniform mass this reduces to the W-skew mesh rotation. This
        # margin extension claims energy closure, not a discrete PV theorem.
        metric = H.metric(state)
        absolute = (
            mesh.curl(ui)
            + self.ocean.rotation_rate
            * mesh.coriolis_unit_vertex[:, None]
        )
        v0, v1 = mesh.verticesOnEdge.T
        q = 0.5 * (absolute[v0] + absolute[v1])
        ids = mesh.safe_rotation_stencil
        pair_mass = 2 * me[:, None, :] * me[ids] / safe(me[:, None, :] + me[ids])
        coefficients = (
            -0.5
            * (q[:, None, :] + q[ids])
            * mesh.rotation_weights_masked[..., None]
            * pair_mass
        )
        W = (mesh.dc_edge * mesh.dv_edge)[:, None]
        forward = (coefficients * ui[ids]).sum(1)
        transpose = torch.zeros_like(ui).index_add(
            0, ids.flatten(), (coefficients * (W * ui)[:, None, :]).reshape(-1, 1)
        )
        du = du + torch.where(
            active, 0.5 * (W * forward - transpose) / safe(metric), 0.0
        )
        power = (mesh.cell_area[:, None] * (ga * da + gv * dv)).sum() + (
            metric * ui * du
        ).sum()
        return da, dv, du, power

    def tendency(
        self,
        state,
        dt,
        *,
        wind_velocity=None,
        ocean_dynamics=True,
        transport=True,
        drag=True,
        ridging=True,
    ):
        self.check(state)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        H = self.hamiltonian
        s = state.ocean.ocean
        mesh = s.mesh
        ice = state.ice
        area = mesh.cell_area[:, None]
        rho = s.column.rho0
        wet = self.ocean.energy.mass_area(s) > 0
        active = self.ridge.active_edges(state)
        ocean_active = (
            torch.ones_like(active)
            if self.ocean.domain is None
            else self.ocean.domain.active_edge[:, None]
        )
        packed = self.ocean.pack_transport(state.ocean)
        if ocean_dynamics:
            rhs, transport_report = self.ocean.dynamics.tendency_with_report(packed, dt)
        else:
            rhs = packed.with_tensors(
                torch.zeros_like(packed.u_e),
                torch.zeros_like(packed.h),
                torch.zeros_like(packed.temp_h),
                torch.zeros_like(packed.salt_h),
                torch.zeros_like(packed.tracer_h),
            )
        base = self.ocean.unpack_transport(rhs, state.ocean)
        if self.ocean.backscatter is not None:
            closure = self.ocean.backscatter
            if dt * closure.fraction > closure.timescale:
                raise ValueError("backscatter requires dt*fraction <= timescale")
            extra_u, extra_e, _ = closure.tendency(self.ocean, state.ocean)
            b = base.ocean
            base = base.with_layers(b.with_tensors(
                    b.u_e + extra_u, b.h, b.temp_h, b.salt_h, b.tracer_h
                ), eddy_h=base.eddy_h + extra_e)
        du = base.ocean.u_e
        dtemp = base.ocean.temp_h
        dk = base.tke_h
        zero = ice.area.new_zeros(())
        da = torch.zeros_like(ice.area)
        dv = torch.zeros_like(ice.volume)
        di = torch.zeros_like(ice.velocity)
        transport_power = zero
        loading_power = zero
        if transport:
            da, dv, di, transport_power = self.ice_transport(state)
        if ocean_dynamics:
            load = self.ocean.dynamics.loading_acceleration(
                s, s.column.gravity * H.mass(state), transport_report
            )
            du = du + load
            loading_power = (
                self.ocean.energy.base.mechanical.metric(s) * s.u_e * load
            ).sum()
        ridge_heat = zero
        if ridging:
            # The full OceanIce stage was validated at tendency entry.
            # Do not repeat the same ocean/temperature validation in ridging.
            r, report = self.ridge.tendency(state, _validated=True)
            # A newly populated category can receive both transported ice and
            # ridge material of different thickness. Their unresolved thickness
            # variance releases PE; route that nonnegative mixing power to heat.
            birth_heat = H.coefficient(state) * torch.where(
                ice.area == 0,
                da
                * r.area
                / safe(da + r.area)
                * (dv / safe(da) - r.volume / safe(r.area)).square(),
                0.0,
            ).sum(-1, keepdim=True)
            da = da + r.area
            dv = dv + r.volume
            di = di + r.velocity
            dtemp = dtemp + r.temp_h
            ridge_heat = report.heating_power.sum()
            # Additional compaction of EVERY category prevents exhausting the
            # only donor from turning off resistance. At a=1 this offsets the
            # uniform convergent area increase. Stage bounds still enforce dt.
            a = ice.area.sum(-1, keepdim=True)
            c = (-mesh.div(ice.velocity)).clamp_min(0)
            rate = a.pow(self.ridge.crowding_power) * c
            pack_da = -rate * ice.area
            potential = (
                H.coefficient(state)
                * (ice.volume * ice.thickness()).sum(-1, keepdim=True)
                * rate
            )
            pressure = torch.where(
                c > 0,
                (1 + self.packing_friction)
                * H.coefficient(state)
                * (ice.volume * ice.thickness()).sum(-1, keepdim=True)
                * a.pow(self.ridge.crowding_power),
                0.0,
            )
            di = di + torch.where(
                active,
                -mesh.grad(pressure) / safe(mesh.cell_to_edge(H.mass(state))),
                0.0,
            )
            da = da + pack_da
            heat = self.packing_friction * potential + birth_heat
            workspace = RoquetWorkspace(self.ocean.layers, s)
            capacity = (
                workspace.reference.cp_theta
                + 2 * workspace.a * s.temperature()
                + workspace.b
            )
            dtemp = dtemp + torch.cat(
                (heat / (rho * capacity[:, :1]), torch.zeros_like(s.h[:, 1:])), -1
            )
            ridge_heat = ridge_heat + (area * heat).sum()
        Mi = H.metric(state)
        Mo = self.ocean.energy.base.mechanical.metric(s)[:, :1]
        uic = mesh.cell_wind_vector(ice.velocity)[:, 0, :]
        uoc = mesh.cell_wind_vector(s.u_e[:, :1])[:, 0, :]
        coverage = ice.area.sum(-1, keepdim=True)
        drag_power = zero
        if drag:
            # The same active-edge projection belongs on both sides of R.
            # Otherwise ice-free edges contribute fictitious TKE work.
            relative = mesh.cell_wind_vector(
                torch.where(active, s.u_e[:, :1] - ice.velocity, 0.0)
            )[:, 0, :]
            tau = (
                coverage
                * rho
                * self.drag_coefficient
                * torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                * relative
            )
            force = vector_transpose(mesh, area * tau)
            force = torch.where(active, force, 0.0)
            di = di + force / safe(Mi)
            du = du + torch.cat((-force / safe(Mo), torch.zeros_like(s.u_e[:, 1:])), -1)
            gain = (tau * relative).sum(-1, keepdim=True)
            dk = dk + torch.cat((gain / rho, torch.zeros_like(s.h[:, 1:])), -1)
            drag_power = (area * gain).sum()
        wind_power = zero
        if wind_velocity is not None:
            if (
                wind_velocity.shape != (mesh.nCells, 3)
                or wind_velocity.dtype != s.h.dtype
                or wind_velocity.device != s.h.device
                or not torch.isfinite(wind_velocity).all()
            ):
                raise ValueError(
                    "wind_velocity must be finite [cell,3], matching state"
                )
            radial = (wind_velocity * mesh.cell_xyz).sum(-1).abs()
            tolerance = (
                64
                * torch.finfo(s.h.dtype).eps
                * torch.linalg.vector_norm(wind_velocity, dim=-1)
            )
            if (radial > tolerance).any():
                raise ValueError("wind_velocity must be tangent to the sphere")
            for fraction, velocity, target in [
                (coverage, uic, "ice"),
                (wet.to(s.h.dtype) - coverage, uoc, "ocean"),
            ]:
                relative = wind_velocity - velocity
                tau = (
                    fraction
                    * self.air_density
                    * self.air_drag_coefficient
                    * torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                    * relative
                )
                force = vector_transpose(mesh, area * tau)
                mask = active if target == "ice" else ocean_active
                force = torch.where(mask, force, 0.0)
                if target == "ice":
                    di = di + force / safe(Mi)
                else:
                    du = du + torch.cat(
                        (force / safe(Mo), torch.zeros_like(s.u_e[:, 1:])), -1
                    )
                edge_velocity = ice.velocity if target == "ice" else s.u_e[:, :1]
                wind_power = wind_power + (force * edge_velocity).sum()
        physical = base.ocean.with_tensors(
            du, base.ocean.h, dtemp, base.ocean.salt_h, base.ocean.tracer_h
        )
        ocean_rhs = base.with_layers(physical, tke_h=dk)
        return RidgingState(ocean_rhs, CategoryIce(da, dv, di)), CoupledReport(
            wind_power, drag_power, ridge_heat, transport_power, loading_power
        )

    @staticmethod
    def add(state, rhs, dt):
        a, b = state.ocean, rhs.ocean
        ocean = a.with_layers(a.ocean + dt * b.ocean, tke_h=a.tke_h + dt * b.tke_h)
        if isinstance(a, EddyState):
            ocean = replace(ocean, eddy_h=a.eddy_h + dt * b.eddy_h)
        return RidgingState(
            ocean,
            CategoryIce(
                state.ice.area + dt * rhs.ice.area,
                state.ice.volume + dt * rhs.ice.volume,
                state.ice.velocity + dt * rhs.ice.velocity,
            ),
        )

    @staticmethod
    def blend(first, second, weight):
        a, b = first.ocean, second.ocean
        ocean = a.with_layers((1 - weight) * a.ocean + weight * b.ocean, tke_h=(1 - weight) * a.tke_h + weight * b.tke_h)
        if isinstance(a, EddyState):
            ocean = replace(ocean, eddy_h=(1 - weight) * a.eddy_h + weight * b.eddy_h)
        ice = CategoryIce(
            *(
                (1 - weight) * getattr(first.ice, k) + weight * getattr(second.ice, k)
                for k in ("area", "volume", "velocity")
            )
        )
        return RidgingState(ocean, ice)

    def dynamics_step(self, state, dt, *, integrator="ssprk3", **options):
        """Choose the explicit ocean–ice tableau without changing stage physics."""
        if integrator == "ssprk43":
            from vfs.ice.explicit_rk import ssprk43_dynamics
            return ssprk43_dynamics(self, state, dt, **options)
        if integrator != "ssprk3":
            raise ValueError("ocean–ice integrator must be ssprk3 or ssprk43")
        # Preserve the historical three-stage path, including all arithmetic.
        r0, p0 = self.tendency(state, dt, **options)
        first = self.add(state, r0, dt)
        r1, p1 = self.tendency(first, dt, **options)
        middle = self.blend(state, self.add(first, r1, dt), 0.25)
        r2, p2 = self.tendency(middle, dt, **options)
        end = self.blend(state, self.add(middle, r2, dt), 2 / 3)
        self.check(end)
        wind = dt * (p0.wind_power / 6 + p1.wind_power / 6 + 2 * p2.wind_power / 3)
        return end, wind

    def drag_step(self, state, dt):
        from vfs.ice.ice_drag import midpoint_drag

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        return midpoint_drag(self, state, dt)

    def step(
        self, state, dt, *, heat_flux=None, wind_velocity=None, air_temperature=None,
        dynamics_integrator="ssprk3",
    ):
        """Phase/drag half steps around selected SSP dynamics; external energy [J]."""
        from vfs.ice.category_phase import CategoryThermodynamics

        if heat_flux is not None and air_temperature is not None:
            raise ValueError("choose prescribed heat flux or air temperature")
        phase = CategoryThermodynamics(self)
        area = state.ocean.ocean.mesh.cell_area[:, None]
        prescribed = (
            torch.zeros_like(state.ice.area[:, :1]) if heat_flux is None else heat_flux
        )
        q0 = (
            prescribed
            if air_temperature is None
            else phase.air_flux(state, air_temperature)
        )
        state, _ = phase.apply(state, 0.5 * dt, heat_flux=q0)
        state, _ = self.drag_step(state, 0.5 * dt)
        state, work = self.dynamics_step(
            state, dt, wind_velocity=wind_velocity, drag=False,
            integrator=dynamics_integrator,
        )
        if self.ocean.closure is not None:
            previous = state.ocean.ocean
            mixed, k, _ = self.ocean.closure.apply(
                self.ocean.layers, previous, state.ocean.tke_h, dt
            )
            if not torch.equal(mixed.h, previous.h):
                raise ValueError("ice coupling requires fixed-geometry mixing")
            state = replace(state, ocean=state.ocean.with_layers(mixed, tke_h=k))
        state, _ = self.drag_step(state, 0.5 * dt)
        q1 = (
            prescribed
            if air_temperature is None
            else phase.air_flux(state, air_temperature)
        )
        state, _ = phase.apply(state, 0.5 * dt, heat_flux=q1)
        return state, work + 0.5 * dt * (area * (q0 + q1)).sum()


# Historical standalone category-mechanics entry point.
OceanIce = OceanIceMechanics
