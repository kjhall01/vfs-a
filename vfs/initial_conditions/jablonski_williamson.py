"""Jablonowski-Williamson analytic formulas and initial-condition machinery."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from vfs.atmosphere.column import CharneyPhillipsColumn
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.atmosphere.state import State
from vfs.geometry.mesh import VoronoiMesh


# Physical constants from the JW 2006 test case.
_a = 6.371229e6
_Omega = 7.292115e-5
_g = 9.80616
_Rd = 287.0
_cp = 1004.0
_kappa = _Rd / _cp

_u0 = 35.0
_T0 = 288.0
_Gamma = 0.005
_eta_t = 0.2
_DeltaT = 480000.0
_eta_0 = 0.252
_p0 = 1.0e5

_u_pert = 1.0
_phi_c = 2.0 * math.pi / 9.0
_lam_c = math.pi / 9.0
_R_pert = _a / 10.0


def _A_lat_jw(lat):
    """JW latitudinal shape A(phi)."""
    return -2.0 * np.sin(lat) ** 6 * (np.cos(lat) ** 2 + 1.0 / 3.0) + 10.0 / 63.0


def _B_lat_jw(lat):
    """JW planetary shape B(phi)."""
    return 8.0 / 5.0 * np.cos(lat) ** 3 * (np.sin(lat) ** 2 + 2.0 / 3.0) - math.pi / 4.0


def _eta_v(eta):
    """JW shifted vertical coordinate."""
    return (eta - _eta_0) * math.pi / 2.0


def _jw_bracket(lat, eta):
    """Temperature bracket in JW (2006), Eq. (6), not the Eq. (7) Phi bracket."""
    return (
        _A_lat_jw(lat) * (2.0 * _u0 * np.cos(_eta_v(eta)).clip(min=0.0) ** 1.5)
        + _B_lat_jw(lat) * (_a * _Omega)
    )


def _T_avg(eta):
    """Horizontally averaged JW temperature."""
    eta = np.asarray(eta, dtype=np.float64)
    T = _T0 * eta ** (_Rd * _Gamma / _g)
    return T + np.where(eta <= _eta_t, _DeltaT * (_eta_t - eta) ** 5, 0.0)


def _Phi_avg_jw(eta):
    """Horizontally averaged JW geopotential."""
    eta = np.asarray(eta, dtype=np.float64)
    eta_safe = np.clip(eta, 1.0e-12, None)
    exponent = _Rd * _Gamma / _g
    lower = _T0 * _g / _Gamma * (1.0 - eta_safe**exponent)

    polynomial = (
        (np.log(eta_safe / _eta_t) + 137.0 / 60.0) * _eta_t**5
        - 5.0 * _eta_t**4 * eta_safe
        + 5.0 * _eta_t**3 * eta_safe**2
        - (10.0 / 3.0) * _eta_t**2 * eta_safe**3
        + 1.25 * _eta_t * eta_safe**4
        - 0.2 * eta_safe**5
    )
    upper = lower - _Rd * _DeltaT * polynomial
    return np.where(eta_safe >= _eta_t, lower, upper)


def _Phi_jw(lat, eta):
    """Full analytic JW geopotential."""
    vertical_shape = np.cos(_eta_v(eta)).clip(min=0.0) ** 1.5
    # Eq. (7) has A*u0*C + B*a*Omega; the factor 2 belongs to
    # its vertical derivative (the temperature bracket), not Phi itself.
    return _Phi_avg_jw(eta) + _u0 * vertical_shape * (
        _A_lat_jw(lat) * _u0 * vertical_shape + _B_lat_jw(lat) * _a * _Omega
    )


def _phi_s_jw(lat):
    """Analytic JW surface geopotential for constant-ps ICs."""
    return _Phi_jw(lat, np.ones_like(lat, dtype=np.float64))


def _dPhi_jw_dlat(lat, eta, *, eps: float = 1.0e-5) -> np.ndarray:
    """Centered finite-difference latitude derivative of analytic JW Phi."""
    lat, eta = np.broadcast_arrays(np.asarray(lat, dtype=np.float64), np.asarray(eta, dtype=np.float64))
    pole_eps = 1.0e-10
    lat_p = np.minimum(lat + float(eps), 0.5 * math.pi - pole_eps)
    lat_m = np.maximum(lat - float(eps), -0.5 * math.pi + pole_eps)
    return (_Phi_jw(lat_p, eta) - _Phi_jw(lat_m, eta)) / np.maximum(lat_p - lat_m, 1.0e-15)


def _theta_from_phi_derivative_jw(lat, eta, p, *, eps: float = 1.0e-5) -> np.ndarray:
    """
    Potential temperature implied by analytic Phi.

    This is the clean thermodynamic JW diagnostic path for this branch. The
    `_T_jw` implements the equivalent published analytic temperature.
    This finite-difference route remains an independent diagnostic.
    """
    lat, eta, p = np.broadcast_arrays(
        np.asarray(lat, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
    )
    eta_safe = np.clip(eta, 1.0e-8, 1.0)
    eta_p = np.minimum(eta_safe + float(eps), 1.0)
    eta_m = np.maximum(eta_safe - float(eps), 1.0e-8)
    phi_anom_p = _Phi_jw(lat, eta_p) - _Phi_avg_jw(eta_p)
    phi_anom_m = _Phi_jw(lat, eta_m) - _Phi_avg_jw(eta_m)
    dphi_deta = (phi_anom_p - phi_anom_m) / np.maximum(eta_p - eta_m, 1.0e-15)

    pi = _cp * (np.clip(p, 1.0e-12, None) / _p0) ** _kappa
    dpi_deta = _kappa * pi / np.maximum(eta_safe, 1.0e-12)
    theta_mean = _T_avg(eta_safe) * (_p0 / np.clip(p, 1.0e-12, None)) ** _kappa
    theta = theta_mean - dphi_deta / np.maximum(dpi_deta, 1.0e-30)
    if not np.isfinite(theta).all():
        raise ValueError("non-finite theta from analytic Phi derivative")
    return theta.astype(np.float64)


def _u_jw(lat, eta):
    """JW zonal wind."""
    return _u0 * np.cos(_eta_v(eta)) ** 1.5 * np.sin(2.0 * lat) ** 2


def _u_from_phi_gradient_wind(lat, eta, *, eps: float = 1.0e-5) -> np.ndarray:
    """
    Zonal wind implied by continuous axisymmetric gradient-wind balance.

    This is a diagnostic negative-control IC helper, not the default wind.
    """
    lat, eta = np.broadcast_arrays(np.asarray(lat, dtype=np.float64), np.asarray(eta, dtype=np.float64))
    lat_safe = np.clip(lat, -0.5 * math.pi + 1.0e-10, 0.5 * math.pi - 1.0e-10)
    t = np.tan(lat_safe)
    f = 2.0 * _Omega * np.sin(lat_safe)
    b = _a * f
    c = _dPhi_jw_dlat(lat_safe, eta, eps=eps)

    disc = b * b - 4.0 * t * c
    disc_scale = np.maximum(b * b + np.abs(4.0 * t * c), 1.0)
    bad = disc < -1.0e-10 * disc_scale
    if np.any(bad):
        raise ValueError(
            "negative gradient-wind discriminant: "
            f"count={int(np.count_nonzero(bad))}, min={float(np.min(disc)):.6e}"
        )
    disc = np.maximum(disc, 0.0)

    u = np.zeros_like(disc, dtype=np.float64)
    linear = np.abs(t) < 1.0e-8
    use_geo = linear & (np.abs(b) > 1.0e-12)
    u[use_geo] = -c[use_geo] / b[use_geo]

    quad = ~linear
    sqrt_disc = np.sqrt(disc[quad])
    denom = 2.0 * t[quad]
    r1 = (-b[quad] + sqrt_disc) / denom
    r2 = (-b[quad] - sqrt_disc) / denom
    u_geo = np.where(np.abs(b[quad]) > 1.0e-12, -c[quad] / b[quad], _u_jw(lat_safe[quad], eta[quad]))
    u[quad] = np.where(np.abs(r1 - u_geo) <= np.abs(r2 - u_geo), r1, r2)

    if not np.isfinite(u).all():
        raise ValueError("non-finite wind from analytic Phi gradient-wind balance")
    return u.astype(np.float64)


def _T_jw(lat, eta):
    """Published analytic temperature, Jablonowski and Williamson (2006), Eq. (6)."""
    ev = _eta_v(eta)
    T = _T_avg(eta)
    return T + (3.0 / 4.0 * eta * math.pi * _u0 / _Rd) * np.sin(ev) * np.cos(ev) ** 0.5 * _jw_bracket(lat, eta)


def _ps_jw(lat):
    """JW surface pressure from gradient-wind balance."""
    ev_s = _eta_v(1.0)
    C = _u0 * math.cos(ev_s) ** 1.5
    return _p0 * np.exp(C / (_Rd * _T_avg(1.0)) * (C * _A_lat_jw(lat) + 2.0 * _Omega * _a * _B_lat_jw(lat)))


def _perturbation_u(lat, lon):
    """JW-B Gaussian zonal-wind perturbation."""
    r = _a * np.arccos(
        np.clip(
            np.sin(_phi_c) * np.sin(lat) + np.cos(_phi_c) * np.cos(lat) * np.cos(lon - _lam_c),
            -1.0,
            1.0,
        )
    )
    return _u_pert * np.exp(-(r / _R_pert) ** 2)


@dataclass
class ICBuildInfo:
    """Metadata describing how a JW initial condition was constructed."""

    kind: str
    balance_mode: str
    perturb: bool
    project_mass_flux: bool
    jw_ic_mode: str
    theta_source: str = "phi_derivative"
    wind_source: str = "U_jw"
    theta_ic_mode: str = "point"
    theta_lsq_lambda: float | None = None
    wind_scale: float = 1.0
    u_max: float | None = None
    theta_hydrostatic_rms_delta: float | None = None
    theta_hydrostatic_max_delta: float | None = None
    theta_hydrostatic_rel_rms_delta: float | None = None
    theta_hydrostatic_constraint_rms: float | None = None
    theta_hydrostatic_constraint_max: float | None = None
    theta_hydrostatic_theta_min: float | None = None
    theta_hydrostatic_theta_max: float | None = None
    theta_hydrostatic_level_start: int | None = None
    theta_hydrostatic_blend: float | None = None
    theta_hydrostatic_max_rel_delta: float | None = None
    theta_hydrostatic_p95_rel_delta: float | None = None


@dataclass
class ICDifferenceReport:
    """Normed difference between two JW-like initial states."""

    rms_du: float
    max_du: float
    rel_rms_du: float
    rms_dtheta: float
    max_dtheta: float
    rel_rms_dtheta: float
    max_rel_dtheta: float
    p95_rel_dtheta: float
    rms_dps: float
    max_dps: float
    rel_rms_dps: float
    analytic_u_max: float
    balanced_u_max: float
    analytic_energy: float
    balanced_energy: float
    rel_energy_difference: float


class JWInitialConditions:
    """Jablonowski-Williamson initial-condition builder."""

    VALID_MODES = ("standard_variable_ps", "constant_ps_surface_phi")

    @staticmethod
    def make_column_for_mode(
        mesh: VoronoiMesh,
        n_levels: int,
        A: np.ndarray,
        B: np.ndarray,
        *,
        mode: str = "standard_variable_ps",
        phi_s_scale: float = 1.0,
        phi_full_mode: str = "interface_mean",
    ) -> CharneyPhillipsColumn:
        """Build a column whose lower-boundary convention matches the JW IC mode."""
        mode = str(mode)
        if mode not in JWInitialConditions.VALID_MODES:
            raise ValueError(f"unknown JW IC mode {mode!r}; choose one of {JWInitialConditions.VALID_MODES}")

        if mode == "constant_ps_surface_phi":
            xyz = mesh.cell_xyz.detach().cpu().numpy()
            cell_lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
            phi_s = float(phi_s_scale) * _phi_s_jw(cell_lat)
        else:
            phi_s = 0.0

        return CharneyPhillipsColumn(n_levels, A, B, phi_s=phi_s, phi_full_mode=phi_full_mode)

    def __init__(self, mesh: VoronoiMesh, column: CharneyPhillipsColumn, *, mode: str = "standard_variable_ps"):
        mode = str(mode)
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown JW IC mode {mode!r}; choose one of {self.VALID_MODES}")

        self.mesh = mesh
        self.column = column
        self.N = column.n_levels
        self.mode = mode

        xyz = mesh.cell_xyz.detach().cpu().numpy()
        self.cell_lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
        self.cell_lon = np.arctan2(xyz[:, 1], xyz[:, 0])

        c0 = mesh.cellsOnEdge[:, 0].detach().cpu().numpy()
        c1 = mesh.cellsOnEdge[:, 1].detach().cpu().numpy()
        mid = xyz[c0] + xyz[c1]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True).clip(1.0e-15)

        self.edge_lat = np.arcsin(np.clip(mid[:, 2], -1.0, 1.0))
        self.edge_lon = np.arctan2(mid[:, 1], mid[:, 0])
        self.edge_normal = mesh.edge_normal.detach().cpu().numpy()

        self.A_i = column.A_interface.detach().cpu().numpy()
        self.B_i = column.B_interface.detach().cpu().numpy()
        self._mass_proj_laplacian: np.ndarray | None = None

    def _p_interface(self, ps: np.ndarray) -> np.ndarray:
        return self.A_i[None, :] + self.B_i[None, :] * ps[:, None]

    def _p_full(self, ps: np.ndarray) -> np.ndarray:
        p_i = self._p_interface(ps)
        return 0.5 * (p_i[:, :-1] + p_i[:, 1:])

    def _eta_from_p(self, p: np.ndarray, ps: np.ndarray | None = None) -> np.ndarray:
        if self.mode == "standard_variable_ps":
            if ps is None:
                raise ValueError("ps is required for standard_variable_ps eta")
            return p / ps[..., None]
        return p / _p0

    def compute_ps(self) -> np.ndarray:
        if self.mode == "constant_ps_surface_phi":
            return np.full(self.mesh.nCells, _p0, dtype=np.float64)
        return _ps_jw(self.cell_lat).astype(np.float64)

    def compute_theta_interfaces(self, ps: np.ndarray, *, theta_source: str = "phi_derivative") -> np.ndarray:
        theta_source = str(theta_source)
        if theta_source not in ("T_jw", "phi_derivative"):
            raise ValueError("theta_source must be 'T_jw' or 'phi_derivative'")
        p_i = self._p_interface(ps)
        eta = np.clip(self._eta_from_p(p_i, ps=ps), max(float(self.A_i[0]) / _p0, 1.0e-6), 1.0)
        if theta_source == "T_jw":
            theta = _T_jw(self.cell_lat[:, None], eta) * (_p0 / np.clip(p_i, 1.0, None)) ** self.column.kappa
        else:
            theta = _theta_from_phi_derivative_jw(self.cell_lat[:, None], eta, p_i)
        return theta.astype(np.float64)

    def compute_phi_interfaces(self, ps: np.ndarray) -> np.ndarray:
        p_i = self._p_interface(ps)
        eta = np.clip(self._eta_from_p(p_i, ps=ps), max(float(self.A_i[0]) / _p0, 1.0e-6), 1.0)
        phi_i = _Phi_jw(self.cell_lat[:, None], eta)
        if self.mode == "standard_variable_ps":
            phi_i = phi_i - _Phi_jw(self.cell_lat[:, None], np.ones_like(eta))
        return phi_i.astype(np.float64)

    def compute_theta_interfaces_hydrostatic_lsq(
        self,
        ps: np.ndarray,
        theta_ref: np.ndarray,
        *,
        theta_lsq_lambda: float = 0.1,
        anomaly_only: bool = True,
        return_diag: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """Regularized closest theta profile matching analytic Phi increments."""
        theta_ref = np.asarray(theta_ref, dtype=np.float64)
        lam_reg = max(float(theta_lsq_lambda), 0.0)
        phi_i = self.compute_phi_interfaces(ps)

        with torch.no_grad():
            ps_t = torch.as_tensor(ps[:, None], dtype=torch.float64)
            pi_i = self.column.exner_interface(ps_t).detach().cpu().numpy()

        delta_pi = pi_i[:, 1:] - pi_i[:, :-1]
        phi_increment = phi_i[:, :-1] - phi_i[:, 1:]
        N = self.N
        A = np.zeros((N, N + 1), dtype=np.float64)
        for k in range(N):
            A[k, k] = 0.5
            A[k, k + 1] = 0.5

        if anomaly_only:
            area = self.mesh.cell_area.detach().cpu().numpy()[:, None]
            area_sum = max(float(area.sum()), 1.0e-30)
            theta_bar_ref = theta_ref @ A.T
            current_increment = theta_bar_ref * delta_pi
            target_mean = (current_increment * area).sum(axis=0, keepdims=True) / area_sum
            phi_increment_anom = phi_increment - (phi_increment * area).sum(axis=0, keepdims=True) / area_sum
            phi_increment = target_mean + phi_increment_anom

        theta_bar_target = phi_increment / np.clip(delta_pi, 1.0e-30, None)
        residual = theta_ref @ A.T - theta_bar_target
        coeff = np.linalg.solve(A @ A.T + lam_reg * np.eye(N, dtype=np.float64), residual.T).T
        theta_new = theta_ref - coeff @ A
        diag = self._theta_projection_diag(theta_ref, theta_new, theta_bar_target, A, 0, 1.0)
        diag["theta_hydrostatic_lsq_lambda"] = float(lam_reg)
        diag["theta_hydrostatic_anomaly_only"] = bool(anomaly_only)
        return (theta_new.astype(np.float64), diag) if return_diag else theta_new.astype(np.float64)

    @staticmethod
    def _theta_projection_diag(
        theta_ref: np.ndarray,
        theta_new: np.ndarray,
        theta_bar_target: np.ndarray,
        A: np.ndarray,
        level_start: int,
        blend: float,
    ) -> dict[str, float]:
        constraint = theta_new @ A.T - theta_bar_target
        active = constraint[:, level_start:] if level_start < A.shape[0] else constraint[:, :0]
        delta = theta_new - theta_ref
        rel_delta = np.abs(delta) / np.maximum(np.abs(theta_ref), 1.0e-30)
        return {
            "theta_hydrostatic_rms_delta": float(np.sqrt(np.mean(delta**2))),
            "theta_hydrostatic_max_delta": float(np.max(np.abs(delta))),
            "theta_hydrostatic_rel_rms_delta": float(
                np.sqrt(np.mean(delta**2)) / max(float(np.sqrt(np.mean(theta_ref**2))), 1.0e-30)
            ),
            "theta_hydrostatic_max_rel_delta": float(np.max(rel_delta)),
            "theta_hydrostatic_p95_rel_delta": float(np.percentile(rel_delta, 95.0)),
            "theta_hydrostatic_constraint_rms": float(np.sqrt(np.mean(constraint**2))),
            "theta_hydrostatic_constraint_max": float(np.max(np.abs(constraint))),
            "theta_hydrostatic_active_constraint_rms": float(np.sqrt(np.mean(active**2)) if active.size else 0.0),
            "theta_hydrostatic_active_constraint_max": float(np.max(np.abs(active)) if active.size else 0.0),
            "theta_hydrostatic_theta_min": float(np.min(theta_new)),
            "theta_hydrostatic_theta_max": float(np.max(theta_new)),
            "theta_hydrostatic_level_start": int(level_start),
            "theta_hydrostatic_blend": float(blend),
        }

    def _edge_ps(self, ps: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ps_t = torch.as_tensor(ps[:, None], dtype=torch.float64)
            return self.mesh.cell_to_edge(ps_t).squeeze(-1).detach().cpu().numpy()

    def compute_u_edge(self, ps: np.ndarray, *, perturb: bool = False, wind_source: str = "U_jw") -> np.ndarray:
        """Sample zonal wind at full levels and project to edge normals."""
        wind_source = str(wind_source)
        if wind_source not in ("U_jw", "phi_gradient_wind"):
            raise ValueError("wind_source must be 'U_jw' or 'phi_gradient_wind'")
        lat_e = self.edge_lat
        lon_e = self.edge_lon
        east = np.stack([-np.sin(lon_e), np.cos(lon_e), np.zeros_like(lon_e)], axis=1)
        ps_e = self._edge_ps(ps)
        p_i_e = self.A_i[None, :] + self.B_i[None, :] * ps_e[:, None]
        p_f_e = 0.5 * (p_i_e[:, :-1] + p_i_e[:, 1:])
        eta_f_e = p_f_e / ps_e[:, None] if self.mode == "standard_variable_ps" else p_f_e / _p0
        eta_f_e = np.clip(eta_f_e, max(float(self.A_i[0]) / _p0, 1.0e-6), 1.0)
        perturb_e = _perturbation_u(lat_e, lon_e)[:, None] if perturb else 0.0
        u_zonal_base = _u_jw(lat_e[:, None], eta_f_e)
        if wind_source == "phi_gradient_wind":
            u_zonal_base = _u_from_phi_gradient_wind(lat_e[:, None], eta_f_e)
        u_zonal = u_zonal_base + perturb_e
        return (u_zonal[:, :, None] * east[:, None, :] * self.edge_normal[:, None, :]).sum(axis=2).astype(np.float64)

    def _mass_flux_projection_matrix(self) -> np.ndarray:
        if self._mass_proj_laplacian is None:
            eye = torch.eye(self.mesh.nCells, dtype=torch.float64)
            lap = self.mesh.div(self.mesh.grad(eye)).detach().cpu().numpy()
            self._mass_proj_laplacian = lap[1:, 1:]
        return self._mass_proj_laplacian

    def project_mass_flux_divergence_free(
        self,
        u_e: np.ndarray,
        ps: np.ndarray,
    ) -> np.ndarray:
        """Project layer mass flux `dp*u` onto the discrete divergence-free space."""
        with torch.no_grad():
            ps_t = torch.as_tensor(ps[:, None], dtype=torch.float64)
            dp_e = self.mesh.cell_to_edge(self.column.layer_thickness(ps_t)).detach().cpu().numpy()
        mass_flux = u_e * dp_e
        layer_div = self.mesh.div(torch.as_tensor(mass_flux, dtype=torch.float64)).detach().cpu().numpy()
        chi = np.zeros((self.mesh.nCells, self.N), dtype=np.float64)
        chi[1:, :] = np.linalg.solve(self._mass_flux_projection_matrix(), layer_div[1:, :])
        grad_chi = self.mesh.grad(torch.as_tensor(chi, dtype=torch.float64)).detach().cpu().numpy()
        return ((mass_flux - grad_chi) / np.clip(dp_e, 1.0e-30, None)).astype(np.float64)

    def make_state(
        self,
        dyn: DryHydrostaticConservativeDynamics | None = None,
        *,
        perturb: bool = False,
        project_mass_flux: bool = False,
        balance_mode: str = "analytic",
        theta_source: str = "phi_derivative",
        wind_source: str = "U_jw",
        theta_ic_mode: str = "point",
        theta_lsq_lambda: float = 0.1,
        return_info: bool = False,
    ) -> State | tuple[State, ICBuildInfo]:
        """
        Build a JW state.

        This method performs state construction only. Optimizer-based wind or
        theta tuning belongs in explicit assessment scripts, not package IC code.
        """
        balance_mode = str(balance_mode)
        theta_ic_mode = "point" if str(theta_ic_mode) == "point_sampled" else str(theta_ic_mode)
        if theta_ic_mode not in ("point", "hydrostatic_lsq"):
            raise ValueError("theta_ic_mode must be 'point', 'point_sampled', or 'hydrostatic_lsq'")

        if balance_mode != "analytic":
            raise ValueError(
                f"unknown balance_mode={balance_mode!r}; package IC construction only supports analytic state construction. "
                "Use theta_ic_mode='hydrostatic_lsq' for the diagnostic discrete-hydrostatic projection."
            )

        ps_np = self.compute_ps()
        theta_np = self.compute_theta_interfaces(ps_np, theta_source=theta_source)
        u_np = self.compute_u_edge(ps_np, perturb=perturb, wind_source=wind_source)
        theta_hydro_diag: dict[str, float] | None = None

        if theta_ic_mode == "hydrostatic_lsq":
            theta_np, theta_hydro_diag = self.compute_theta_interfaces_hydrostatic_lsq(
                ps_np,
                theta_np,
                theta_lsq_lambda=float(theta_lsq_lambda),
                return_diag=True,
            )

        if project_mass_flux:
            u_np = self.project_mass_flux_divergence_free(u_np, ps_np)

        attrs = {}
        if dyn is not None:
            attrs["gravity"] = dyn.gravity
        state = State.from_theta(
            u_e=torch.as_tensor(u_np, dtype=torch.float64),
            theta_i=torch.as_tensor(theta_np, dtype=torch.float64),
            ps=torch.as_tensor(ps_np[:, None], dtype=torch.float64),
            mesh=self.mesh,
            column=self.column,
            attrs=attrs,
        )

        info = ICBuildInfo(
            kind="JW",
            balance_mode=balance_mode,
            perturb=bool(perturb),
            project_mass_flux=bool(project_mass_flux),
            jw_ic_mode=str(self.mode),
            theta_source=str(theta_source),
            wind_source=str(wind_source),
            theta_ic_mode=theta_ic_mode,
            theta_lsq_lambda=float(theta_lsq_lambda) if theta_ic_mode == "hydrostatic_lsq" else None,
            u_max=float(state.u_e.abs().max().item()),
        )
        if theta_hydro_diag is not None:
            info.theta_hydrostatic_rms_delta = float(theta_hydro_diag["theta_hydrostatic_rms_delta"])
            info.theta_hydrostatic_max_delta = float(theta_hydro_diag["theta_hydrostatic_max_delta"])
            info.theta_hydrostatic_rel_rms_delta = float(theta_hydro_diag["theta_hydrostatic_rel_rms_delta"])
            info.theta_hydrostatic_constraint_rms = float(theta_hydro_diag["theta_hydrostatic_constraint_rms"])
            info.theta_hydrostatic_constraint_max = float(theta_hydro_diag["theta_hydrostatic_constraint_max"])
            info.theta_hydrostatic_theta_min = float(theta_hydro_diag["theta_hydrostatic_theta_min"])
            info.theta_hydrostatic_theta_max = float(theta_hydro_diag["theta_hydrostatic_theta_max"])
            info.theta_hydrostatic_level_start = int(theta_hydro_diag["theta_hydrostatic_level_start"])
            info.theta_hydrostatic_blend = float(theta_hydro_diag["theta_hydrostatic_blend"])
            info.theta_hydrostatic_max_rel_delta = float(theta_hydro_diag["theta_hydrostatic_max_rel_delta"])
            info.theta_hydrostatic_p95_rel_delta = float(theta_hydro_diag["theta_hydrostatic_p95_rel_delta"])

        return (state, info) if return_info else state


def compare_initial_states(
    analytic: State,
    balanced: State,
    dyn: DryHydrostaticConservativeDynamics,
) -> ICDifferenceReport:
    """Return normed differences between two JW-like initial states."""
    with torch.no_grad():
        theta_a = analytic.theta()
        theta_b = balanced.theta()
        du = balanced.u_e - analytic.u_e
        dtheta = theta_b - theta_a
        dps = balanced.ps - analytic.ps
        rel_theta = dtheta.abs() / theta_a.abs().clamp_min(1.0e-30)
        energy_a = dyn.total_energy(analytic)
        energy_b = dyn.total_energy(balanced)
        return ICDifferenceReport(
            rms_du=float(du.square().mean().sqrt().item()),
            max_du=float(du.abs().max().item()),
            rel_rms_du=float(du.square().mean().sqrt().item() / analytic.u_e.square().mean().sqrt().clamp_min(1.0).item()),
            rms_dtheta=float(dtheta.square().mean().sqrt().item()),
            max_dtheta=float(dtheta.abs().max().item()),
            rel_rms_dtheta=float(dtheta.square().mean().sqrt().item() / theta_a.square().mean().sqrt().clamp_min(1.0).item()),
            max_rel_dtheta=float(rel_theta.max().item()),
            p95_rel_dtheta=float(torch.quantile(rel_theta.reshape(-1), 0.95).item()),
            rms_dps=float(dps.square().mean().sqrt().item()),
            max_dps=float(dps.abs().max().item()),
            rel_rms_dps=float(dps.square().mean().sqrt().item() / analytic.ps.square().mean().sqrt().clamp_min(1.0).item()),
            analytic_u_max=float(analytic.u_e.abs().max().item()),
            balanced_u_max=float(balanced.u_e.abs().max().item()),
            analytic_energy=float(energy_a.item()),
            balanced_energy=float(energy_b.item()),
            rel_energy_difference=float(((energy_b - energy_a) / energy_a.abs().clamp_min(1.0)).item()),
        )


JablonskiWilliamsonIC = JWInitialConditions


__all__ = [
    "ICBuildInfo",
    "ICDifferenceReport",
    "JWInitialConditions",
    "JablonskiWilliamsonIC",
    "compare_initial_states",
    "_A_lat_jw",
    "_B_lat_jw",
    "_Phi_avg_jw",
    "_Phi_jw",
    "_T_avg",
    "_T_jw",
    "_dPhi_jw_dlat",
    "_eta_v",
    "_p0",
    "_phi_s_jw",
    "_ps_jw",
    "_theta_from_phi_derivative_jw",
    "_u_from_phi_gradient_wind",
    "_u_jw",
]
