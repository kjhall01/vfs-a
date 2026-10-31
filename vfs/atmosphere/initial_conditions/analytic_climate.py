"""Analytic climate-like dry initial states.

This module owns the canonical synthetic balanced-state generator used by
learned dry-core correction experiments.  The exact balanced family is
axisymmetric: smooth red-spectrum latitude profiles for surface pressure and
interface theta are converted into an analytic meridional pressure-gradient
force, then a zonal gradient-wind velocity is solved and projected onto VFS
edge normals.

General two-dimensional red-noise thermodynamic fields do not have a local
scalar gradient-wind closure.  This module therefore provides two 2-D paths:
an offline dense streamfunction solve for linear planetary/geostrophic balance,
and an optional autograd refinement that keeps the thermodynamic fields fixed
while reducing the actual VFS dry-core momentum tendency in a nondivergent
streamfunction subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from vfs.atmosphere.state import State
from vfs.atmosphere.column import CharneyPhillipsColumn


@dataclass(frozen=True)
class AnalyticClimateStateConfig:
    """Configuration for the axisymmetric red-spectrum balanced family."""

    seed: int = 0
    spectral_cutoff: int = 8
    spectral_slope: float = 2.5
    reference_surface_pressure_pa: float = 100_000.0
    ps_amplitude_pa: float = 900.0
    theta_surface_k: float = 300.0
    theta_stratification_k: float = 30.0
    theta_mode_amplitudes_k: tuple[float, ...] = (7.0, 3.5, 1.5)
    target_wind_rms_m_s: float | None = 18.0
    wind_max_abs_m_s: float = 90.0
    finite_difference_step_rad: float = 1.0e-5
    coriolis_floor_lat_deg: float = 8.0
    ps_min_pa: float = 70_000.0
    ps_max_pa: float = 120_000.0
    theta_min_k: float = 180.0
    phi_full_mode: str | None = None


@dataclass
class AnalyticClimateSample:
    """Generated analytic state plus balance diagnostics."""

    state: State
    diagnostics: dict[str, float | int | str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TwoDimensionalBalanceConfig:
    """Configuration for solved 2-D red-spectrum balance families."""

    seed: int = 0
    max_degree: int = 6
    spectral_slope: float = 2.0
    reference_surface_pressure_pa: float = 100_000.0
    ps_amplitude_pa: float = 700.0
    theta_surface_k: float = 300.0
    theta_stratification_k: float = 30.0
    theta_mode_amplitudes_k: tuple[float, ...] = (5.0, 2.5, 1.0)
    target_wind_rms_m_s: float | None = 18.0
    coriolis_floor_lat_deg: float = 8.0
    wind_max_abs_m_s: float = 1_000.0
    theta_min_k: float = 180.0
    ps_min_pa: float = 70_000.0
    ps_max_pa: float = 120_000.0
    dense_solver_max_cells: int = 2048
    solver_ridge: float = 1.0e-2
    nonlinear_iterations: int = 0
    nonlinear_update_damping: float = 0.7
    nonlinear_balance_steps: int = 0
    nonlinear_balance_lr: float = 3.0e-2
    nonlinear_balance_delta_u_reg: float = 1.0e-4
    nonlinear_balance_tolerance: float = 0.0
    stress_velocity_rms_m_s: float = 2.0
    stress_theta_amplitude_k: float = 0.6
    stress_ps_amplitude_pa: float = 80.0
    equatorial_taper: bool = True
    phi_full_mode: str | None = None


def _lat_lon(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lat = torch.asin(xyz[..., 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    return lat, lon


def _east_vector(lon: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)],
        dim=-1,
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(
        torch.finfo(values.dtype).tiny
    )


def _weighted_rms(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        (values.square() * weights).sum()
        / weights.sum().clamp_min(torch.finfo(values.dtype).tiny)
    ).sqrt()


def _legendre_values(mu: torch.Tensor, max_degree: int) -> list[torch.Tensor]:
    max_degree = int(max_degree)
    values = [torch.ones_like(mu)]
    if max_degree <= 0:
        return values
    values.append(mu)
    for ell in range(2, max_degree + 1):
        values.append(
            ((2 * ell - 1) * mu * values[-1] - (ell - 1) * values[-2]) / float(ell)
        )
    return values


def _sample_zonal_coefficients(
    *,
    seed: int,
    max_degree: int,
    spectral_slope: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    coeff = torch.zeros(int(max_degree) + 1, dtype=torch.float64)
    if int(max_degree) >= 1:
        raw = torch.randn(int(max_degree), generator=generator, dtype=torch.float64)
        degree = torch.arange(1, int(max_degree) + 1, dtype=torch.float64)
        coeff[1:] = raw * degree.pow(-float(spectral_slope))
    return coeff.to(dtype=dtype, device=device)


def _real_spherical_harmonic_basis(
    xyz: torch.Tensor,
    max_degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real SH-like basis values and degree index on unit-sphere points."""

    max_degree = int(max_degree)
    if max_degree < 1:
        raise ValueError("max_degree must be at least 1")
    dtype = xyz.dtype
    device = xyz.device
    z = xyz[..., 2].clamp(-1.0, 1.0)
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    sin_theta = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    p: list[list[torch.Tensor | None]] = [
        [None for _ in range(max_degree + 1)] for _ in range(max_degree + 1)
    ]
    p[0][0] = torch.ones_like(z)
    for m in range(1, max_degree + 1):
        assert p[m - 1][m - 1] is not None
        p[m][m] = -(2 * m - 1) * sin_theta * p[m - 1][m - 1]
    for m in range(0, max_degree):
        assert p[m][m] is not None
        p[m + 1][m] = (2 * m + 1) * z * p[m][m]
    for m in range(0, max_degree + 1):
        for ell in range(m + 2, max_degree + 1):
            assert p[ell - 1][m] is not None
            assert p[ell - 2][m] is not None
            p[ell][m] = (
                (2 * ell - 1) * z * p[ell - 1][m]
                - (ell + m - 1) * p[ell - 2][m]
            ) / float(ell - m)

    values = []
    degrees = []
    for ell in range(1, max_degree + 1):
        assert p[ell][0] is not None
        values.append(p[ell][0])
        degrees.append(ell)
        for m in range(1, ell + 1):
            assert p[ell][m] is not None
            values.append(p[ell][m] * torch.cos(float(m) * lon))
            degrees.append(ell)
            values.append(p[ell][m] * torch.sin(float(m) * lon))
            degrees.append(ell)
    basis = torch.stack(values, dim=0)
    degree_t = torch.tensor(degrees, dtype=dtype, device=device)
    return basis, degree_t


def _sample_red_coefficients(
    *,
    seed: int,
    degrees: torch.Tensor,
    spectral_slope: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    coeff = torch.randn(int(degrees.numel()), generator=generator, dtype=torch.float64)
    coeff = coeff.to(dtype=degrees.dtype, device=degrees.device)
    return coeff * degrees.clamp_min(1.0).pow(-float(spectral_slope))


class _ZonalRedProfile:
    def __init__(
        self,
        *,
        coefficients: torch.Tensor,
        normalize_lat: torch.Tensor,
        normalize_weights: torch.Tensor,
    ) -> None:
        self.coefficients = coefficients
        raw = self._raw(normalize_lat)
        mean = _weighted_mean(raw, normalize_weights)
        rms = _weighted_rms(raw - mean, normalize_weights).clamp_min(
            torch.finfo(raw.dtype).tiny
        )
        self.mean = mean
        self.rms = rms

    def _raw(self, lat: torch.Tensor) -> torch.Tensor:
        mu = torch.sin(lat)
        values = _legendre_values(mu, int(self.coefficients.numel()) - 1)
        out = torch.zeros_like(lat)
        for ell, value in enumerate(values):
            out = out + self.coefficients[ell] * value
        return out

    def __call__(self, lat: torch.Tensor) -> torch.Tensor:
        return (self._raw(lat) - self.mean) / self.rms


class _TwoDimensionalRedField:
    def __init__(
        self,
        *,
        coefficients: torch.Tensor,
        basis: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self.coefficients = coefficients
        raw = torch.einsum("m,mc->c", coefficients, basis)
        mean = _weighted_mean(raw, weights)
        rms = _weighted_rms(raw - mean, weights).clamp_min(torch.finfo(raw.dtype).tiny)
        self.values = (raw - mean) / rms


class AnalyticClimateStateGenerator:
    """Generate analytic axisymmetric gradient-wind/hydrostatic states."""

    def __init__(
        self,
        mesh,
        column: CharneyPhillipsColumn,
        *,
        config: AnalyticClimateStateConfig | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.mesh = mesh
        self.base_column = column
        self.config = AnalyticClimateStateConfig() if config is None else config
        self.dtype = dtype if dtype is not None else mesh.cell_xyz.dtype
        self.device = device if device is not None else mesh.cell_xyz.device
        self.radius = float(getattr(mesh, "radius", 6.371e6))

        if not isinstance(column, CharneyPhillipsColumn):
            raise TypeError(
                "AnalyticClimateStateGenerator requires a CharneyPhillipsColumn"
            )

        phi_full_mode = self.config.phi_full_mode or getattr(
            column, "phi_full_mode", "direct_u_gl2_adjoint"
        )
        self.column = CharneyPhillipsColumn(
            int(column.n_levels),
            column.A_interface.detach().cpu(),
            column.B_interface.detach().cpu(),
            phi_s=0.0,
            phi_full_mode=str(phi_full_mode),
        )

    def _coriolis_and_curvature(
        self, lat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        f = 2.0 * 7.292115e-5 * torch.sin(lat)
        floor = 2.0 * 7.292115e-5 * math.sin(
            math.radians(float(cfg.coriolis_floor_lat_deg))
        )
        sign = torch.where(f >= 0.0, torch.ones_like(f), -torch.ones_like(f))
        f_safe = sign * f.abs().clamp_min(float(floor))
        curve = torch.tan(lat).clamp(-8.0, 8.0) / self.radius
        return f_safe, curve

    def _profiles(self, seed: int) -> tuple[_ZonalRedProfile, list[_ZonalRedProfile]]:
        cfg = self.config
        lat_grid = torch.linspace(
            -0.5 * math.pi + 1.0e-4,
            0.5 * math.pi - 1.0e-4,
            2049,
            dtype=self.dtype,
            device=self.device,
        )
        weights = torch.cos(lat_grid).clamp_min(0.0)
        ps_coeff = _sample_zonal_coefficients(
            seed=int(seed) + 11,
            max_degree=int(cfg.spectral_cutoff),
            spectral_slope=float(cfg.spectral_slope),
            dtype=self.dtype,
            device=self.device,
        )
        theta_profiles = []
        for index, _amp in enumerate(cfg.theta_mode_amplitudes_k):
            coeff = _sample_zonal_coefficients(
                seed=int(seed) + 101 + 37 * index,
                max_degree=int(cfg.spectral_cutoff),
                spectral_slope=float(cfg.spectral_slope) + 0.08 * index,
                dtype=self.dtype,
                device=self.device,
            )
            theta_profiles.append(
                _ZonalRedProfile(
                    coefficients=coeff,
                    normalize_lat=lat_grid,
                    normalize_weights=weights,
                )
            )
        return (
            _ZonalRedProfile(
                coefficients=ps_coeff,
                normalize_lat=lat_grid,
                normalize_weights=weights,
            ),
            theta_profiles,
        )

    def _theta_vertical_modes(self) -> torch.Tensor:
        n = int(self.column.n_levels)
        eta = torch.linspace(0.0, 1.0, n + 1, dtype=self.dtype, device=self.device)
        x = 2.0 * eta - 1.0
        envelope = torch.sin(math.pi * eta).clamp_min(0.0)
        candidates = [
            envelope,
            x * envelope,
            0.5 * (3.0 * x.square() - 1.0) * envelope,
            0.5 * (5.0 * x.pow(3) - 3.0 * x) * envelope,
        ]
        modes = []
        for candidate in candidates[: len(self.config.theta_mode_amplitudes_k)]:
            rms = candidate.square().mean().sqrt().clamp_min(
                torch.finfo(self.dtype).tiny
            )
            modes.append(candidate / rms)
        if not modes:
            modes.append(torch.zeros_like(eta))
        return torch.stack(modes, dim=0)

    def _fields_at_lat(
        self,
        lat: torch.Tensor,
        *,
        seed: int,
        anomaly_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        ps_profile, theta_profiles = self._profiles(seed)
        eta = torch.linspace(
            0.0,
            1.0,
            int(self.column.n_levels) + 1,
            dtype=self.dtype,
            device=self.device,
        )
        base_theta = (
            float(cfg.theta_surface_k)
            + float(cfg.theta_stratification_k) * (1.0 - eta)
        )
        ps = (
            float(cfg.reference_surface_pressure_pa)
            + float(anomaly_scale) * float(cfg.ps_amplitude_pa) * ps_profile(lat)
        )
        ps = ps.clamp(float(cfg.ps_min_pa), float(cfg.ps_max_pa)).reshape(-1, 1)

        theta = base_theta.reshape(1, -1).expand(lat.numel(), -1).clone()
        modes = self._theta_vertical_modes()
        for index, profile in enumerate(theta_profiles):
            theta = theta + (
                float(anomaly_scale)
                * float(cfg.theta_mode_amplitudes_k[index])
                * profile(lat).reshape(-1, 1)
                * modes[index].reshape(1, -1)
            )
        theta = theta.clamp_min(float(cfg.theta_min_k))
        return ps, theta

    def _pgf_meridional(
        self,
        lat: torch.Tensor,
        *,
        seed: int,
        anomaly_scale: float,
    ) -> torch.Tensor:
        h = float(self.config.finite_difference_step_rad)
        lat_plus = (lat + h).clamp(
            -0.5 * math.pi + 1.0e-5, 0.5 * math.pi - 1.0e-5
        )
        lat_minus = (lat - h).clamp(
            -0.5 * math.pi + 1.0e-5, 0.5 * math.pi - 1.0e-5
        )
        ps_plus, theta_plus = self._fields_at_lat(
            lat_plus, seed=seed, anomaly_scale=anomaly_scale
        )
        ps_minus, theta_minus = self._fields_at_lat(
            lat_minus, seed=seed, anomaly_scale=anomaly_scale
        )
        phi_plus, alpha_b_plus = self.column.pgf_components(theta_plus, ps_plus)
        phi_minus, alpha_b_minus = self.column.pgf_components(theta_minus, ps_minus)
        distance = (
            (lat_plus - lat_minus).reshape(-1, 1).clamp_min(1.0e-12) * self.radius
        )
        dphi_dn = (phi_plus - phi_minus) / distance
        dps_dn = (ps_plus - ps_minus) / distance
        alpha_b = 0.5 * (alpha_b_plus + alpha_b_minus)
        return dphi_dn + alpha_b * dps_dn

    def _solve_zonal_gradient_wind(
        self, pgf_meridional: torch.Tensor, lat: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        f_safe, curve = self._coriolis_and_curvature(lat)

        u = (-pgf_meridional / f_safe.reshape(-1, 1)).clamp(
            -float(cfg.wind_max_abs_m_s), float(cfg.wind_max_abs_m_s)
        )
        for _ in range(8):
            residual = (
                pgf_meridional
                + f_safe.reshape(-1, 1) * u
                + curve.reshape(-1, 1) * u.square()
            )
            jac = f_safe.reshape(-1, 1) + 2.0 * curve.reshape(-1, 1) * u
            jac = torch.sign(jac) * jac.abs().clamp_min(1.0e-10)
            u = (u - residual / jac).clamp(
                -float(cfg.wind_max_abs_m_s), float(cfg.wind_max_abs_m_s)
            )
        return u

    def _build_once(
        self,
        *,
        seed: int,
        anomaly_scale: float,
    ) -> tuple[State, dict[str, torch.Tensor | float]]:
        cell_xyz = self.mesh.cell_xyz.to(dtype=self.dtype, device=self.device)
        edge_xyz = self.mesh.edge_xyz.to(dtype=self.dtype, device=self.device)
        edge_normal = self.mesh.edge_normal.to(dtype=self.dtype, device=self.device)
        cell_lat, _cell_lon = _lat_lon(cell_xyz)
        edge_lat, edge_lon = _lat_lon(edge_xyz)
        ps, theta = self._fields_at_lat(
            cell_lat, seed=seed, anomaly_scale=anomaly_scale
        )
        pgf_merid = self._pgf_meridional(
            edge_lat, seed=seed, anomaly_scale=anomaly_scale
        )
        u_zonal = self._solve_zonal_gradient_wind(pgf_merid, edge_lat)
        east = _east_vector(edge_lon)
        u_e = u_zonal * (east * edge_normal).sum(dim=-1).reshape(-1, 1)
        f_safe, curve = self._coriolis_and_curvature(edge_lat)
        residual = (
            pgf_merid
            + f_safe.reshape(-1, 1) * u_zonal
            + curve.reshape(-1, 1) * u_zonal.square()
        )
        state = State.from_theta(
            u_e.to(dtype=self.dtype),
            theta.to(dtype=self.dtype),
            ps.to(dtype=self.dtype),
            mesh=self.mesh,
            column=self.column,
            attrs={
                "gravity": 9.80665,
                "analytic_climate_family": "balanced_axisymmetric_gradient_wind",
                "analytic_balance_model": "axisymmetric_gradient_wind_hydrostatic_cp",
                "analytic_climate_seed": int(seed),
                "analytic_climate_anomaly_scale": float(anomaly_scale),
            },
        )
        return state, {
            "u_zonal": u_zonal,
            "pgf_meridional": pgf_merid,
            "gradient_wind_residual": residual,
            "anomaly_scale": float(anomaly_scale),
        }

    def sample_balanced(self, *, seed: int | None = None) -> AnalyticClimateSample:
        """Return a balanced axisymmetric red-spectrum gradient-wind sample."""

        cfg = self.config
        sample_seed = int(cfg.seed if seed is None else seed)
        anomaly_scale = 1.0
        state, tensors = self._build_once(seed=sample_seed, anomaly_scale=anomaly_scale)

        target = cfg.target_wind_rms_m_s
        if target is not None:
            u_zonal = tensors["u_zonal"]
            assert isinstance(u_zonal, torch.Tensor)
            rms = u_zonal.square().mean().sqrt().detach()
            if torch.isfinite(rms) and float(rms.cpu()) > 1.0e-8:
                anomaly_scale = max(
                    0.05, min(8.0, float(target) / float(rms.cpu()))
                )
                state, tensors = self._build_once(
                    seed=sample_seed, anomaly_scale=anomaly_scale
                )

        residual = tensors["gradient_wind_residual"]
        u_zonal = tensors["u_zonal"]
        pgf = tensors["pgf_meridional"]
        assert isinstance(residual, torch.Tensor)
        assert isinstance(u_zonal, torch.Tensor)
        assert isinstance(pgf, torch.Tensor)
        diagnostics = {
            "seed": int(sample_seed),
            "spectral_cutoff": int(cfg.spectral_cutoff),
            "spectral_slope": float(cfg.spectral_slope),
            "anomaly_scale": float(tensors["anomaly_scale"]),
            "u_zonal_rms_m_s": float(u_zonal.square().mean().sqrt().detach().cpu()),
            "u_zonal_max_abs_m_s": float(u_zonal.abs().max().detach().cpu()),
            "pgf_meridional_rms_m_s2": float(
                pgf.square().mean().sqrt().detach().cpu()
            ),
            "analytic_gradient_wind_residual_rms_m_s2": float(
                residual.square().mean().sqrt().detach().cpu()
            ),
            "analytic_gradient_wind_residual_max_abs_m_s2": float(
                residual.abs().max().detach().cpu()
            ),
            "ps_min_pa": float(state.ps.min().detach().cpu()),
            "ps_max_pa": float(state.ps.max().detach().cpu()),
            "theta_min_k": float(state.theta().min().detach().cpu()),
            "theta_max_k": float(state.theta().max().detach().cpu()),
        }
        metadata = {
            "family": "balanced",
            "balance": "axisymmetric_gradient_wind",
            "exact_balance_scope": "continuum_axisymmetric_meridional_force",
            "ps_theta_generation": "independent_zonal_red_spectral_profiles",
            "state_attrs": dict(state.attrs or {}),
        }
        return AnalyticClimateSample(
            state=state, diagnostics=diagnostics, metadata=metadata
        )


def _planetary_rotation(mesh, u_e: torch.Tensor, rotation_rate: float) -> torch.Tensor:
    coriolis = (float(rotation_rate) * mesh.coriolis_unit_vertex).to(
        dtype=u_e.dtype,
        device=u_e.device,
    )
    v0 = mesh.verticesOnEdge[:, 0]
    v1 = mesh.verticesOnEdge[:, 1]
    q_e = 0.5 * (coriolis[v0] + coriolis[v1])
    s = mesh.safe_rotation_stencil
    weights = mesh.rotation_weights_masked.to(dtype=u_e.dtype, device=u_e.device)
    u_s = u_e[..., s, :]
    q_s = q_e[s].to(dtype=u_e.dtype, device=u_e.device)
    q_t = q_e.to(dtype=u_e.dtype, device=u_e.device)[:, None]
    term = 0.5 * (q_t + q_s)[..., None] * weights[..., None] * u_s
    return -term.sum(dim=-2)


def _edge_rms(mesh, value: torch.Tensor) -> torch.Tensor:
    metric = (mesh.dc_edge * mesh.dv_edge).to(dtype=value.dtype, device=value.device)
    while metric.ndim < value.ndim:
        metric = metric.unsqueeze(-1)
    return (metric * value.square()).sum().div(metric.sum().clamp_min(1.0)).sqrt()


def _edge_mean_square(mesh, value: torch.Tensor) -> torch.Tensor:
    metric = (mesh.dc_edge * mesh.dv_edge).to(dtype=value.dtype, device=value.device)
    while metric.ndim < value.ndim:
        metric = metric.unsqueeze(-1)
    return (metric * value.square()).sum() / metric.sum().clamp_min(
        torch.finfo(value.dtype).tiny
    )


class TwoDimensionalBalanceSolver:
    """Dense streamfunction balance solver for moderate VFS meshes."""

    def __init__(
        self,
        mesh,
        *,
        rotation_rate: float = 7.292115e-5,
        ridge: float = 1.0e-8,
        max_cells: int = 2048,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.mesh = mesh
        self.rotation_rate = float(rotation_rate)
        self.ridge = float(ridge)
        self.max_cells = int(max_cells)
        self.dtype = dtype if dtype is not None else mesh.cell_xyz.dtype
        self._operator: torch.Tensor | None = None
        self._normal: torch.Tensor | None = None
        self._sqrt_metric: torch.Tensor | None = None

    def _build_operator(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(self.mesh.nCells) > self.max_cells:
            raise ValueError(
                "2-D dense balance solve is limited to "
                f"{self.max_cells} cells by config, got {int(self.mesh.nCells)}"
            )
        dtype = like.dtype
        device = like.device
        if (
            self._operator is not None
            and self._operator.dtype == dtype
            and self._operator.device == device
        ):
            assert self._sqrt_metric is not None
            return self._operator, self._sqrt_metric

        identity = torch.eye(int(self.mesh.nVertices), dtype=dtype, device=device)
        stream_wind = self.mesh.curl_adjoint(identity)
        operator = _planetary_rotation(
            self.mesh, stream_wind, self.rotation_rate
        )
        metric = (self.mesh.dc_edge * self.mesh.dv_edge).to(dtype=dtype, device=device)
        sqrt_metric = metric.clamp_min(torch.finfo(dtype).tiny).sqrt().reshape(-1, 1)
        self._operator = operator
        self._sqrt_metric = sqrt_metric
        self._normal = None
        return operator, sqrt_metric

    def _normal_matrix(
        self, operator: torch.Tensor, sqrt_metric: torch.Tensor
    ) -> torch.Tensor:
        if (
            self._normal is not None
            and self._normal.dtype == operator.dtype
            and self._normal.device == operator.device
        ):
            return self._normal
        weighted = operator * sqrt_metric
        normal = weighted.transpose(0, 1) @ weighted
        diag_scale = normal.diagonal().mean().clamp_min(
            torch.finfo(operator.dtype).tiny
        )
        normal = normal + float(self.ridge) * diag_scale * torch.eye(
            normal.shape[0], dtype=operator.dtype, device=operator.device
        )
        self._normal = normal
        return normal

    def solve_streamfunction(self, target_acceleration: torch.Tensor) -> torch.Tensor:
        """Solve ``planetary_rotation(curl_adjoint(psi)) ~= target``."""

        target = target_acceleration.to(dtype=self.dtype)
        operator, sqrt_metric = self._build_operator(target)
        normal = self._normal_matrix(operator, sqrt_metric)
        weighted = operator * sqrt_metric
        rhs = weighted.transpose(0, 1) @ (target * sqrt_metric)
        return torch.linalg.solve(normal, rhs)

    def solve_wind(
        self,
        pgf: torch.Tensor,
        *,
        nonlinear_iterations: int = 0,
        nonlinear_update_damping: float = 0.7,
        wind_max_abs: float = 90.0,
        return_streamfunction: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, float]]
    ):
        target = -pgf
        psi = self.solve_streamfunction(target)
        for _ in range(max(0, int(nonlinear_iterations))):
            u = self.mesh.curl_adjoint(psi)
            full_residual = (
                self.mesh.trisk_pv_flux_rotation(u, self.rotation_rate) + pgf
            )
            delta = self.solve_streamfunction(-full_residual)
            psi = psi + float(nonlinear_update_damping) * delta
        u = self.mesh.curl_adjoint(psi)
        u_max = u.abs().max().clamp_min(torch.finfo(u.dtype).tiny)
        if float(u_max.detach().cpu()) > float(wind_max_abs):
            scale = float(wind_max_abs) / u_max
            u = u * scale
            psi = psi * scale
        linear_residual = _planetary_rotation(self.mesh, u, self.rotation_rate) + pgf
        full_residual = self.mesh.trisk_pv_flux_rotation(u, self.rotation_rate) + pgf
        diagnostics = {
            "linear_balance_residual_rms_m_s2": float(
                _edge_rms(self.mesh, linear_residual).detach().cpu()
            ),
            "full_balance_residual_rms_m_s2": float(
                _edge_rms(self.mesh, full_residual).detach().cpu()
            ),
            "pgf_rms_m_s2": float(_edge_rms(self.mesh, pgf).detach().cpu()),
            "divergence_rms_s1": float(
                self.mesh.div(u).square().mean().sqrt().detach().cpu()
            ),
            "streamfunction_solver": "dense_streamfunction_least_squares",
        }
        if return_streamfunction:
            return u, psi, diagnostics
        return u, diagnostics


class TwoDimensionalBalancedClimateGenerator:
    """Generate 2-D red-spectrum states with a solved C-grid balance wind."""

    def __init__(
        self,
        mesh,
        column: CharneyPhillipsColumn,
        *,
        config: TwoDimensionalBalanceConfig | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.mesh = mesh
        self.base_column = column
        self.config = TwoDimensionalBalanceConfig() if config is None else config
        self.dtype = dtype if dtype is not None else mesh.cell_xyz.dtype
        self.device = device if device is not None else mesh.cell_xyz.device
        phi_full_mode = self.config.phi_full_mode or getattr(
            column, "phi_full_mode", "direct_u_gl2_adjoint"
        )
        self.column = CharneyPhillipsColumn(
            int(column.n_levels),
            column.A_interface.detach().cpu(),
            column.B_interface.detach().cpu(),
            phi_s=0.0,
            phi_full_mode=str(phi_full_mode),
        )
        cell_xyz = self.mesh.cell_xyz.to(dtype=self.dtype, device=self.device)
        self.basis, self.degrees = _real_spherical_harmonic_basis(
            cell_xyz, int(self.config.max_degree)
        )
        self.area = self.mesh.cell_area.to(dtype=self.dtype, device=self.device)
        self.solver = TwoDimensionalBalanceSolver(
            self.mesh,
            ridge=float(self.config.solver_ridge),
            max_cells=int(self.config.dense_solver_max_cells),
            dtype=self.dtype,
        )

    def _field(self, seed: int, *, slope_offset: float = 0.0) -> torch.Tensor:
        coeff = _sample_red_coefficients(
            seed=int(seed),
            degrees=self.degrees,
            spectral_slope=float(self.config.spectral_slope) + float(slope_offset),
        )
        return _TwoDimensionalRedField(
            coefficients=coeff,
            basis=self.basis,
            weights=self.area,
        ).values

    def _tapered_field(self, seed: int, *, slope_offset: float = 0.0) -> torch.Tensor:
        values = self._field(seed, slope_offset=slope_offset)
        if not bool(self.config.equatorial_taper):
            return values
        z_abs = self.mesh.cell_xyz[:, 2].to(dtype=self.dtype, device=self.device).abs()
        floor = math.sin(math.radians(float(self.config.coriolis_floor_lat_deg)))
        taper = torch.tanh((z_abs / max(floor, 1.0e-6)).square())
        values = values * taper
        values = values - _weighted_mean(values, self.area)
        return values / _weighted_rms(values, self.area).clamp_min(
            torch.finfo(values.dtype).tiny
        )

    def _theta_vertical_modes(self) -> torch.Tensor:
        n = int(self.column.n_levels)
        eta = torch.linspace(0.0, 1.0, n + 1, dtype=self.dtype, device=self.device)
        x = 2.0 * eta - 1.0
        envelope = torch.sin(math.pi * eta).clamp_min(0.0)
        candidates = [
            envelope,
            x * envelope,
            0.5 * (3.0 * x.square() - 1.0) * envelope,
            0.5 * (5.0 * x.pow(3) - 3.0 * x) * envelope,
        ]
        modes = []
        for candidate in candidates[: len(self.config.theta_mode_amplitudes_k)]:
            modes.append(
                candidate
                / candidate.square().mean().sqrt().clamp_min(
                    torch.finfo(self.dtype).tiny
                )
            )
        return torch.stack(modes, dim=0)

    def _fields(
        self, seed: int, *, anomaly_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        ps_pattern = self._tapered_field(int(seed) + 17)
        ps = (
            float(cfg.reference_surface_pressure_pa)
            + float(anomaly_scale) * float(cfg.ps_amplitude_pa) * ps_pattern
        ).clamp(float(cfg.ps_min_pa), float(cfg.ps_max_pa))
        eta = torch.linspace(
            0.0,
            1.0,
            int(self.column.n_levels) + 1,
            dtype=self.dtype,
            device=self.device,
        )
        theta = (
            float(cfg.theta_surface_k)
            + float(cfg.theta_stratification_k) * (1.0 - eta)
        ).reshape(1, -1).expand(int(self.mesh.nCells), -1).clone()
        modes = self._theta_vertical_modes()
        for index, amplitude in enumerate(cfg.theta_mode_amplitudes_k):
            theta = theta + (
                float(anomaly_scale)
                * float(amplitude)
                * self._tapered_field(
                    int(seed) + 101 + 53 * index, slope_offset=0.08 * index
                )
                .reshape(-1, 1)
                * modes[index].reshape(1, -1)
            )
        return ps.reshape(-1, 1), theta.clamp_min(float(cfg.theta_min_k))

    def _pgf(self, theta: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        phi_full, alpha_b = self.column.pgf_components(theta, ps)
        return -self.mesh.grad(phi_full) - self.mesh.cell_to_edge(
            alpha_b
        ) * self.mesh.grad(ps)

    def _momentum_residual_from_psi(
        self,
        psi: torch.Tensor,
        *,
        theta: torch.Tensor,
        ps: torch.Tensor,
        dyn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_e = self.mesh.curl_adjoint(psi)
        state = State.from_theta(
            u_e,
            theta,
            ps,
            mesh=self.mesh,
            column=self.column,
            attrs={"gravity": 9.80665},
        )
        terms = dyn.tendency_terms(state)["u_e"]
        return terms["total"], u_e

    def _refine_nonlinear_gradient_balance(
        self,
        psi0: torch.Tensor,
        *,
        theta: torch.Tensor,
        ps: torch.Tensor,
        pgf: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        steps = max(0, int(cfg.nonlinear_balance_steps))
        if steps <= 0:
            return self.mesh.curl_adjoint(psi0).detach(), {}

        raise ValueError("nonlinear balance tuning for the removed canonical dry core is no longer supported")

    def _build_balanced(
        self,
        *,
        seed: int,
        anomaly_scale: float,
    ) -> tuple[State, dict[str, float]]:
        cfg = self.config
        ps, theta = self._fields(seed, anomaly_scale=anomaly_scale)
        pgf = self._pgf(theta, ps)
        u_e, psi, solver_diag = self.solver.solve_wind(
            pgf,
            nonlinear_iterations=int(cfg.nonlinear_iterations),
            nonlinear_update_damping=float(cfg.nonlinear_update_damping),
            wind_max_abs=float(cfg.wind_max_abs_m_s),
            return_streamfunction=True,
        )
        refine_diag: dict[str, float] = {}
        if int(cfg.nonlinear_balance_steps) > 0:
            u_e, refine_diag = self._refine_nonlinear_gradient_balance(
                psi, theta=theta, ps=ps, pgf=pgf
            )
        family = (
            "balanced_2d_gradient_red_noise"
            if int(cfg.nonlinear_balance_steps) > 0
            else "balanced_2d_red_noise"
        )
        balance_model = (
            "2d_autograd_gradient_balance_solve"
            if int(cfg.nonlinear_balance_steps) > 0
            else "2d_streamfunction_balance_solve"
        )
        state = State.from_theta(
            u_e.detach().to(dtype=self.dtype),
            theta.detach().to(dtype=self.dtype),
            ps.detach().to(dtype=self.dtype),
            mesh=self.mesh,
            column=self.column,
            attrs={
                "gravity": 9.80665,
                "analytic_climate_family": family,
                "analytic_balance_model": balance_model,
                "analytic_climate_seed": int(seed),
                "analytic_climate_anomaly_scale": float(anomaly_scale),
            },
        )
        diagnostics = {
            **solver_diag,
            **refine_diag,
            "seed": int(seed),
            "max_degree": int(cfg.max_degree),
            "spectral_slope": float(cfg.spectral_slope),
            "anomaly_scale": float(anomaly_scale),
            "u_rms_m_s": float(_edge_rms(self.mesh, u_e).detach().cpu()),
            "u_max_abs_m_s": float(u_e.abs().max().detach().cpu()),
            "ps_min_pa": float(ps.min().detach().cpu()),
            "ps_max_pa": float(ps.max().detach().cpu()),
            "theta_min_k": float(theta.min().detach().cpu()),
            "theta_max_k": float(theta.max().detach().cpu()),
        }
        return state, diagnostics

    def sample_balanced(self, *, seed: int | None = None) -> AnalyticClimateSample:
        cfg = self.config
        sample_seed = int(cfg.seed if seed is None else seed)
        anomaly_scale = 1.0
        target = cfg.target_wind_rms_m_s
        state, diagnostics = self._build_balanced(
            seed=sample_seed, anomaly_scale=anomaly_scale
        )
        for _ in range(3):
            if target is None or diagnostics["u_rms_m_s"] <= 1.0e-8:
                break
            factor = float(target) / float(diagnostics["u_rms_m_s"])
            if abs(math.log(max(factor, 1.0e-12))) < 0.05:
                break
            anomaly_scale = max(
                1.0e-4, min(8.0, float(anomaly_scale) * max(0.2, min(5.0, factor)))
            )
            state, diagnostics = self._build_balanced(
                seed=sample_seed, anomaly_scale=anomaly_scale
            )
        metadata = {
            "family": str(state.attrs.get("analytic_climate_family", "balanced_2d")),
            "balance": str(state.attrs.get("analytic_balance_model", "unknown")),
            "ps_theta_generation": (
                "independent_2d_real_spherical_harmonic_red_spectrum"
            ),
            "state_attrs": dict(state.attrs or {}),
        }
        return AnalyticClimateSample(
            state=state, diagnostics=diagnostics, metadata=metadata
        )

    def sample_stress(self, *, seed: int | None = None) -> AnalyticClimateSample:
        cfg = self.config
        sample_seed = int(cfg.seed if seed is None else seed)
        balanced = self.sample_balanced(seed=sample_seed)
        state = balanced.state
        velocity_pattern = self._tapered_field(sample_seed + 911).reshape(-1, 1)
        # Map cell stress to vertices by nearest smooth recomputation on vertex xyz.
        vertex_basis, vertex_degrees = _real_spherical_harmonic_basis(
            self.mesh.vertex_xyz.to(dtype=self.dtype, device=self.device),
            int(cfg.max_degree),
        )
        coeff = _sample_red_coefficients(
            seed=sample_seed + 977,
            degrees=vertex_degrees,
            spectral_slope=float(cfg.spectral_slope),
        )
        raw_vertex = torch.einsum("m,mv->v", coeff, vertex_basis)
        raw_vertex = raw_vertex - raw_vertex.mean()
        raw_vertex = raw_vertex / raw_vertex.square().mean().sqrt().clamp_min(
            torch.finfo(self.dtype).tiny
        )
        perturb_u = self.mesh.curl_adjoint(
            raw_vertex.reshape(-1, 1).expand(-1, int(self.column.n_levels))
        )
        perturb_u = perturb_u * (
            float(cfg.stress_velocity_rms_m_s)
            / _edge_rms(self.mesh, perturb_u).clamp_min(torch.finfo(self.dtype).tiny)
        )
        theta = state.theta() + float(cfg.stress_theta_amplitude_k) * velocity_pattern
        ps = state.ps + float(cfg.stress_ps_amplitude_pa) * velocity_pattern
        stress_family = (
            "stress_2d_gradient_red_noise"
            if int(cfg.nonlinear_balance_steps) > 0
            else "stress_2d_red_noise"
        )
        stress_model = (
            "near_balanced_2d_gradient_stress"
            if int(cfg.nonlinear_balance_steps) > 0
            else "near_balanced_2d_streamfunction_stress"
        )
        stressed = State.from_theta(
            state.u_e + perturb_u,
            theta.clamp_min(float(cfg.theta_min_k)),
            ps.clamp(float(cfg.ps_min_pa), float(cfg.ps_max_pa)),
            mesh=state.mesh,
            column=state.column,
            attrs={
                **dict(state.attrs or {}),
                "analytic_climate_family": stress_family,
                "analytic_balance_model": stress_model,
            },
        )
        pgf = self._pgf(stressed.theta(), stressed.ps)
        residual = self.mesh.trisk_pv_flux_rotation(stressed.u_e, 7.292115e-5) + pgf
        diagnostics = {
            **balanced.diagnostics,
            "stress_velocity_rms_m_s": float(
                _edge_rms(self.mesh, perturb_u).detach().cpu()
            ),
            "stress_balance_residual_rms_m_s2": float(
                _edge_rms(self.mesh, residual).detach().cpu()
            ),
            "stress_divergence_rms_s1": float(
                self.mesh.div(stressed.u_e).square().mean().sqrt().detach().cpu()
            ),
        }
        return AnalyticClimateSample(
            state=stressed,
            diagnostics=diagnostics,
            metadata={
                "family": "stress_2d",
                "base_family": "balanced_2d",
                "state_attrs": dict(stressed.attrs or {}),
            },
        )


def make_analytic_climate_state(
    mesh,
    column: CharneyPhillipsColumn,
    *,
    seed: int = 0,
    dtype: torch.dtype | None = None,
    config: AnalyticClimateStateConfig | None = None,
) -> State:
    """Convenience wrapper returning only the balanced State."""

    generator = AnalyticClimateStateGenerator(
        mesh,
        column,
        config=config,
        dtype=dtype,
    )
    return generator.sample_balanced(seed=int(seed)).state


__all__ = [
    "AnalyticClimateSample",
    "AnalyticClimateStateConfig",
    "AnalyticClimateStateGenerator",
    "TwoDimensionalBalanceConfig",
    "TwoDimensionalBalanceSolver",
    "TwoDimensionalBalancedClimateGenerator",
    "make_analytic_climate_state",
]
