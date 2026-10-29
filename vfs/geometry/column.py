"""
Charney-Phillips vertical column geometry for hydrostatic dynamics.

State layout
------------
All tensors follow the [..., nSpatial, nLayers] convention used by
VoronoiMesh operators:

  u   (edge-normal wind):     [..., nEdges,  N]    full levels  k = 0..N-1
  θ   (potential temperature):[..., nCells, N+1]   interfaces   k = 0..N
  ps  (surface pressure):     [..., nCells,  1]    surface scalar

Vertical index convention
--------------------------
  k = 0   →  model top     (lowest  pressure, highest altitude)
  k = N   →  surface       (highest pressure = ps)

  N   full levels:   k = 0..N-1  (midpoints between adjacent interfaces)
  N+1 interfaces:    k = 0..N

Hybrid sigma-pressure coordinate
----------------------------------
  p_interface[k] = A[k] + B[k] * ps

  Required boundary conditions:
    A[0] > 0,  B[0] = 0   →  pure pressure surface at model top
    A[N] = 0,  B[N] = 1   →  pure sigma surface at the ground = ps

  The factory function make_hybrid_coefficients() generates a standard
  set of coefficients satisfying these conditions.

Why Charney-Phillips?
----------------------
The Lorenz vertical staggering places θ and u on the same full levels.
The discrete hydrostatic operator then has a null space — the
"computational mode" — that is unphysical and grows in long runs.

In the Charney-Phillips (CP) staggering, θ is placed at interfaces.
The hydrostatic equation dφ = -θ dπ is then evaluated where θ is
native, making the operator full rank with no null space.

Hydrostatic reconstruction
---------------------------
Starting from the surface:

  φ_interface[N] = φ_s           (topographic boundary condition)

  φ_interface[k] = φ_interface[k+1]
                 + θ̄[k] · (π_interface[k+1] - π_interface[k])

where

  θ̄[k] = ½ (θ_interface[k] + θ_interface[k+1])

  π_interface[k] = c_p · (p_interface[k] / p₀)^κ

The arithmetic mean for θ̄ is not just convenient — it is the
energy-consistent choice.  See the Adjointness note below.

Full-level geopotential is the simple mean of its bounding interfaces:

  φ_full[k] = ½ (φ_interface[k] + φ_interface[k+1])

Pressure-gradient force on sigma surfaces
------------------------------------------
On a hybrid-sigma surface at full level k, the horizontal PGF is not
simply ∇φ_full.  A correction for the slope of the sigma surface is
required:

  PGF_k = ∇_η φ_full_k  +  α_full_k · B_full_k · ∇_η ps

where α = R_d T / p is the specific volume.  The method pgf_components()
returns (φ_full, α·B_full) so that the dynamics can call mesh.grad on
both and sum:

  pgf_e = mesh.grad(phi_full) + (alpha_B * mesh.grad(ps)[..., :1])

Note that mesh.grad(ps) has shape [..., nEdges, 1] which broadcasts
against alpha_B of shape [..., nCells, N] after the cell-to-edge step.

Adjointness of the hydrostatic operator
-----------------------------------------
Define the linear operator H : R^{N+1} → R^{N+1} (treating φ_s as fixed):

  (Hθ)[k] = Σ_{j=k}^{N-1}  ½(θ[j] + θ[j+1]) · δπ[j]

where δπ[j] = π_i[j+1] - π_i[j] > 0 is the Exner thickness of layer j.

Under the *unweighted* Euclidean inner product on R^{N+1}, the adjoint
H^T satisfies:

  (H^T ψ)[j] = ½ δπ[j-1] · Σ_{k=0}^{j-1} ψ[k]   (j > 0)
             + ½ δπ[j]   · Σ_{k=0}^{j}   ψ[k]   (j < N)

i.e. it is an upward prefix-sum operator scaled by half the neighbouring
δπ values.  The full expression is documented in hydrostatic_adjoint_rhs().

Energy conservation requirement: the θ tendency due to adiabatic
compression must equal -H^T applied to the geopotential tendency.
Specifically:

  ∂θ_i[k] / ∂t |_adiabatic = -(H^T · d(φ_i)/dt)[k]

The smoke test verifies this property numerically using autograd.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

R_D  = 287.05    # J kg⁻¹ K⁻¹  dry air gas constant
C_P  = 1004.0    # J kg⁻¹ K⁻¹  specific heat at constant pressure
P0   = 1.0e5     # Pa           reference pressure
GRAV = 9.80665   # m s⁻²        standard gravity


# ---------------------------------------------------------------------------
# Hybrid coefficient factory
# ---------------------------------------------------------------------------

def make_hybrid_coefficients(
    n_levels: int,
    p_top: float = 5000.0,
    sigma_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate hybrid sigma-pressure interface coefficients.

    Interface pressures are defined by:
        p_i[k] = A[k] + B[k] * ps

    The transition is parameterised by sigma_power:
        B[k] = (k/N)^sigma_power      (0 at top, 1 at surface)
        A[k] = p_top · (1 - B[k])     (p_top at top, 0 at surface)

    This gives:
        p_i[0] = p_top   (model top, independent of ps)
        p_i[N] = ps      (surface, pure sigma)
        p_i[k] = p_top + (k/N)^n · (ps - p_top)   (smooth transition)

    Parameters
    ----------
    n_levels     : N, number of full levels (N+1 interfaces).
    p_top        : Pressure at model top in Pa (default 5000 Pa = 50 hPa).
    sigma_power  : Controls transition steepness.
                   1 = linear, 2 = quadratic (standard, default).
                   Higher values concentrate levels near the surface.

    Returns
    -------
    A : (N+1,) array of A coefficients in Pa.
    B : (N+1,) array of B coefficients (dimensionless, 0..1).
    """
    N = int(n_levels)
    eta = np.linspace(0.0, 1.0, N + 1, dtype=np.float64)   # 0..1
    B = eta ** float(sigma_power)
    A = float(p_top) * (1.0 - B)
    # Enforce exact boundary conditions to machine precision
    A[0] = float(p_top);  B[0] = 0.0
    A[-1] = 0.0;          B[-1] = 1.0
    return A, B


# ---------------------------------------------------------------------------
# CharneyPhillipsColumn
# ---------------------------------------------------------------------------

class CharneyPhillipsColumn(nn.Module):
    """
    Vertical column geometry for hydrostatic CP dynamics.

    See module docstring for a full description of the conventions and
    the adjointness requirements.

    Parameters
    ----------
    n_levels : int
        Number of full levels N.  There are N+1 interfaces.
    A : array_like, shape (N+1,)
        Hybrid A coefficients [Pa].  Must satisfy A[0] = p_top > 0, A[N] = 0.
    B : array_like, shape (N+1,)
        Hybrid B coefficients.  Must satisfy B[0] = 0, B[N] = 1.
    phi_s : array_like or float
        Surface geopotential [m² s⁻²].  Can be:
          - A scalar (0.0 for flat topography).
          - A 1-D array of shape (nCells,) for terrain.
        Stored as a buffer.  The column is tied to a specific horizontal
        grid once phi_s has a spatial extent.
    R_d, c_p, p0, g : float
        Physical constants.  Defaults are standard dry-air values.
    """

    def __init__(
        self,
        n_levels: int,
        A: np.ndarray | torch.Tensor,
        B: np.ndarray | torch.Tensor,
        phi_s: np.ndarray | torch.Tensor | float = 0.0,
        *,
        phi_full_mode: str = "interface_mean",
        R_d: float = R_D,
        c_p: float = C_P,
        p0:  float = P0,
        g:   float = GRAV,
    ) -> None:
        super().__init__()

        N = int(n_levels)
        if N < 1:
            raise ValueError(f"n_levels must be >= 1, got {N}")

        A = torch.as_tensor(A, dtype=torch.float64).reshape(-1)
        B = torch.as_tensor(B, dtype=torch.float64).reshape(-1)

        if A.shape != (N + 1,):
            raise ValueError(f"A must have shape ({N+1},), got {tuple(A.shape)}")
        if B.shape != (N + 1,):
            raise ValueError(f"B must have shape ({N+1},), got {tuple(B.shape)}")
        if float(B[0].abs()) > 1e-10:
            raise ValueError("B[0] must be 0 (pure pressure surface at model top)")
        if float((B[-1] - 1.0).abs()) > 1e-10:
            raise ValueError("B[N] must be 1 (pure sigma at ground)")
        if float(A[-1].abs()) > 1e-10:
            raise ValueError("A[N] must be 0")
        if float(A[0]) <= 0.0:
            raise ValueError("A[0] = p_top must be > 0")

        if isinstance(phi_s, (int, float)):
            phi_s_t = torch.tensor(float(phi_s), dtype=torch.float64)
        else:
            phi_s_t = torch.as_tensor(phi_s, dtype=torch.float64)

        self.n_levels: int = N
        self.R_d:  float = float(R_d)
        self.c_p:  float = float(c_p)
        self.p0:   float = float(p0)
        self.g:    float = float(g)
        self.kappa: float = float(R_d / c_p)   # ≈ 0.2857 for dry air
        self.phi_full_mode: str = str(phi_full_mode)
        if self.phi_full_mode not in (
            "interface_mean",
            "linear_pfull_adjoint",
            "direct_u_linear_adjoint",
            "direct_u_gl2_adjoint",
        ):
            raise ValueError(
                "phi_full_mode must be 'interface_mean', 'linear_pfull_adjoint', "
                "'direct_u_linear_adjoint', or 'direct_u_gl2_adjoint', "
                f"got {self.phi_full_mode!r}"
            )

        self.register_buffer("A_interface", A)              # [N+1]
        self.register_buffer("B_interface", B)              # [N+1]
        # B at full levels: average of the two bounding interface values.
        # Used in the sigma-surface PGF correction.
        self.register_buffer("B_full", 0.5 * (B[:-1] + B[1:]))  # [N]
        self.register_buffer("phi_s", phi_s_t)             # [] or [nCells]

    # ------------------------------------------------------------------
    # Vertical operators — all inputs/outputs follow [..., nSpatial, nK]
    # ------------------------------------------------------------------

    def interface_pressure(self, ps: torch.Tensor) -> torch.Tensor:
        """
        Pressure at all N+1 interfaces.

        p_i[..., k] = A[k] + B[k] · ps[..., 0]

        Input:  ps   [..., nCells, 1]
        Output: p_i  [..., nCells, N+1]
        """
        # A_interface, B_interface: [N+1]
        # ps[..., 0]: [..., nCells] → broadcast over interface index
        return self.A_interface + self.B_interface * ps

    def layer_thickness(self, ps: torch.Tensor) -> torch.Tensor:
        """
        Pressure thickness of each full level (always positive).

        δp[..., k] = p_i[..., k+1] - p_i[..., k]  > 0

        k+1 is deeper (higher pressure) than k.

        Input:  ps   [..., nCells, 1]
        Output: δp   [..., nCells, N]
        """
        p_i = self.interface_pressure(ps)
        return p_i[..., 1:] - p_i[..., :-1]

    def exner_interface(self, ps: torch.Tensor) -> torch.Tensor:
        """
        Exner pressure at interfaces: π = c_p · (p/p₀)^κ.

        Input:  ps   [..., nCells, 1]
        Output: π_i  [..., nCells, N+1]
        """
        p_i = self.interface_pressure(ps)
        return self.exner_pressure(p_i)

    def exner_pressure(self, p: torch.Tensor) -> torch.Tensor:
        """Exner pressure π = c_p · (p/p₀)^κ for arbitrary pressure p."""
        return self.c_p * (p / self.p0) ** self.kappa

    def full_pressure(self, ps: torch.Tensor) -> torch.Tensor:
        """
        Actual layer pressure midpoint p_full = 0.5*(p_i[k]+p_i[k+1]).

        This is distinct from the arithmetic Exner midpoint used by the
        historical interface_mean full-level placement.
        """
        p_i = self.interface_pressure(ps)
        return 0.5 * (p_i[..., :-1] + p_i[..., 1:])

    def exner_full_pmid(self, ps: torch.Tensor) -> torch.Tensor:
        """Exner pressure evaluated at the actual layer pressure midpoint."""
        return self.exner_pressure(self.full_pressure(ps))

    def exner_full(self, ps: torch.Tensor) -> torch.Tensor:
        """
        Exner pressure at full levels: mean of bounding interfaces.

        Input:  ps   [..., nCells, 1]
        Output: π    [..., nCells, N]
        """
        pi_i = self.exner_interface(ps)
        return 0.5 * (pi_i[..., :-1] + pi_i[..., 1:])

    def geopotential_interface(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Geopotential at all N+1 interfaces.

        Integrates dφ = -θ dπ upward from the surface using the CP
        layer-mean θ̄[k] = ½(θ_i[k] + θ_i[k+1]).

        Boundary condition: φ_interface[N] = φ_s.

        Vectorised as a suffix cumulative sum — no Python loop.

        Input:  theta_i  [..., nCells, N+1]   θ at interfaces
                ps       [..., nCells, 1]
        Output: phi_i    [..., nCells, N+1]
        """
        pi_i     = self.exner_interface(ps)                        # [..., N+1]
        delta_pi = pi_i[..., 1:] - pi_i[..., :-1]                 # [..., N], > 0

        # θ̄[k] = ½(θ[k] + θ[k+1])  — layer mean in the CP staggering
        theta_avg = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])  # [..., N]

        # integrand[k] = θ̄[k] · δπ[k]  (the geopotential increment per layer)
        integrand = theta_avg * delta_pi                           # [..., N]

        # suffix_sum[k] = Σ_{j=k}^{N-1} integrand[j]
        #
        # φ_interface[k] = φ_s + suffix_sum[k]
        # φ_interface[N] = φ_s + 0 = φ_s  ✓
        #
        # Implement as: flip → cumsum → flip, then append zero.
        integrand_r = torch.flip(integrand, dims=[-1])
        suffix_sum  = torch.flip(torch.cumsum(integrand_r, dim=-1), dims=[-1])

        phi_s = self.phi_s.unsqueeze(-1)                          # [nCells, 1] or [1]

        # Append a zero column for the surface interface (k=N).
        # Derive zeros from suffix_sum so the batch dims match automatically.
        zeros = torch.zeros_like(suffix_sum[..., :1])             # [..., nCells, 1]
        increment = torch.cat([suffix_sum, zeros], dim=-1)        # [..., nCells, N+1]
        return phi_s + increment

    def geopotential_full_interface_mean(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Geopotential at full levels: mean of bounding interfaces.

        φ_full[k] = ½(φ_interface[k] + φ_interface[k+1])

        Input:  theta_i  [..., nCells, N+1]
                ps       [..., nCells, 1]
        Output: phi      [..., nCells, N]
        """
        phi_i = self.geopotential_interface(theta_i, ps)
        return 0.5 * (phi_i[..., :-1] + phi_i[..., 1:])

    def geopotential_full_linear_pfull(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full-level geopotential evaluated at actual layer pressure midpoint.

        This keeps the standard CP interface hydrostatic reconstruction and
        surface boundary condition, but places the full-level force point at
        p_full[k] = 0.5*(p_i[k]+p_i[k+1]):

            phi_full[k] = phi_i[k+1]
                        + theta_bar[k] * (pi_i[k+1] - pi(p_full[k]))

        The within-layer geopotential is therefore linear in Exner pressure
        with the same theta_bar used by the CP hydrostatic interface operator.
        """
        phi_i = self.geopotential_interface(theta_i, ps)
        pi_i = self.exner_interface(ps)
        pi_full = self.exner_full_pmid(ps)
        theta_bar = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
        return phi_i[..., 1:] + theta_bar * (pi_i[..., 1:] - pi_full)

    def theta_full_pmid_linear_exner(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Theta linearly interpolated in Exner pressure to p_full.

        This is used by direct_u_gl2_adjoint for both alpha and internal
        energy placement.
        """
        pi_i = self.exner_interface(ps)
        pi_u = self.exner_full_pmid(ps)
        delta = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(1.0e-300)
        s = (pi_u - pi_i[..., :-1]) / delta
        return theta_i[..., :-1] + s * (theta_i[..., 1:] - theta_i[..., :-1])

    def geopotential_full_direct_u_gl2(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct-to-u/full-level geopotential using GL2 within the partial layer.

        Full-layer contributions below k are the exact integral of linearly
        interpolated theta over the layer, which equals the CP trapezoid
        theta_bar * delta_pi.  The partial integral from pi_u[k] to the lower
        interface pi_i[k+1] is evaluated with two Gauss-Legendre nodes:

            x = +/- 1/sqrt(3), weights = 1
            pi_q = 0.5*(pi_u + pi_bot) + 0.5*(pi_bot - pi_u)*x

        Since theta(pi) is linear in Exner pressure, GL2 is exact here.
        """
        pi_i = self.exner_interface(ps)
        pi_u = self.exner_full_pmid(ps)
        delta_pi = pi_i[..., 1:] - pi_i[..., :-1]
        theta_top = theta_i[..., :-1]
        theta_bot = theta_i[..., 1:]
        theta_bar = 0.5 * (theta_top + theta_bot)

        layer_integral = theta_bar * delta_pi
        suffix = torch.flip(torch.cumsum(torch.flip(layer_integral, dims=[-1]), dim=-1), dims=[-1])
        zeros = torch.zeros_like(suffix[..., :1])
        phi_i = self.phi_s.to(dtype=theta_i.dtype, device=theta_i.device).unsqueeze(-1) + torch.cat([suffix, zeros], dim=-1)

        pi_bot = pi_i[..., 1:]
        half = 0.5 * (pi_bot - pi_u)
        mid = 0.5 * (pi_bot + pi_u)
        node = 1.0 / math.sqrt(3.0)
        q0 = mid - half * node
        q1 = mid + half * node
        denom = delta_pi.clamp_min(1.0e-300)
        s0 = (q0 - pi_i[..., :-1]) / denom
        s1 = (q1 - pi_i[..., :-1]) / denom
        theta_q0 = theta_top + s0 * (theta_bot - theta_top)
        theta_q1 = theta_top + s1 * (theta_bot - theta_top)
        partial = half * (theta_q0 + theta_q1)
        return phi_i[..., 1:] + partial

    def geopotential_full(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """Geopotential at the active full-level placement."""
        if self.phi_full_mode == "interface_mean":
            return self.geopotential_full_interface_mean(theta_i, ps)
        if self.phi_full_mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
            return self.geopotential_full_linear_pfull(theta_i, ps)
        if self.phi_full_mode == "direct_u_gl2_adjoint":
            return self.geopotential_full_direct_u_gl2(theta_i, ps)
        raise RuntimeError(f"unsupported phi_full_mode={self.phi_full_mode!r}")

    def temperature_full(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temperature at full levels: T = θ · (π / c_p).

        Uses the same layer-mean θ̄ as the hydrostatic reconstruction
        to keep the thermodynamics consistent.

        Input:  theta_i  [..., nCells, N+1]
                ps       [..., nCells, 1]
        Output: T        [..., nCells, N]
        """
        if self.phi_full_mode == "direct_u_gl2_adjoint":
            theta_full = self.theta_full_pmid_linear_exner(theta_i, ps)
        else:
            theta_full = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
        if self.phi_full_mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint", "direct_u_gl2_adjoint"):
            pi_full = self.exner_full_pmid(ps)
        else:
            pi_full = self.exner_full(ps)
        return theta_full * (pi_full / self.c_p)

    def specific_volume_full(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Specific volume α = R_d T / p at full levels.

        Needed for the PGF correction on sigma surfaces:

            PGF_k = mesh.grad(phi_full_k)
                  + alpha_full_k * B_full_k * mesh.grad(ps)

        Input:  theta_i  [..., nCells, N+1]
                ps       [..., nCells, 1]
        Output: alpha    [..., nCells, N]
        """
        T      = self.temperature_full(theta_i, ps)
        p_full = self.full_pressure(ps)
        return self.R_d * T / p_full#.clamp_min(1.0)

    def pgf_components(
        self,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Components needed to assemble the horizontal PGF.

        On a hybrid-sigma surface at full level k, the full PGF is:

            PGF_e_k = mesh.grad(phi_full)[..., k]
                    + alpha_B[..., k] * mesh.grad(ps)[..., 0]

        where alpha_B = alpha_full * B_full.

        The second term corrects for the slope of the sigma surface.
        For a flat surface (B_full = 0 everywhere, pure pressure coordinate),
        it vanishes identically.

        Usage in dynamics:
            phi_full, alpha_B = col.pgf_components(theta_i, ps)
            # phi_full: [..., nCells, N]
            # alpha_B:  [..., nCells, N]
            pgf_phi = mesh.grad(phi_full)                     # [..., nEdges, N]
            pgf_ps  = mesh.grad(ps)                           # [..., nEdges, 1]
            pgf     = pgf_phi + alpha_B_edge * pgf_ps
            # where alpha_B_edge = mesh.cell_to_edge(alpha_B)

        Returns
        -------
        phi_full : [..., nCells, N]
        alpha_B  : [..., nCells, N]   = alpha_full * B_full[k]
        """
        phi_full   = self.geopotential_full(theta_i, ps)
        alpha_full = self.specific_volume_full(theta_i, ps)
        # B_full: [N] — broadcast over all leading dims and nCells
        alpha_B = alpha_full * self.B_full
        return phi_full, alpha_B

    def hydrostatic_adjoint_rhs(
        self,
        psi_i: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply H^T: the adjoint of the geopotential operator.

        H maps θ_i → δφ_i (geopotential increment, φ_s fixed).
        H^T maps ψ_i → δθ_i under the Euclidean inner product.

        Derivation
        ----------
        (Hθ)[k] = Σ_{j=k}^{N-1} ½(θ[j] + θ[j+1]) · δπ[j]

        Swap summation order in <Hθ, ψ>:

          <Hθ, ψ> = Σ_{j=0}^{N-1} ½ · δπ[j] · (θ[j] + θ[j+1]) · S_full[j]

        where S_full[j] = Σ_{k=0}^{j} ψ[k]  (prefix sum *including* ψ[j]).

        Collecting contributions to each θ[m]:
          - θ[j]   term at j=m   contributes  ½ · δπ[m]   · S_full[m]
          - θ[j+1] term at j=m-1 contributes  ½ · δπ[m-1] · S_full[m-1]

        Note S_full[m-1] = S_shift[m] = Σ_{k=0}^{m-1} ψ[k]  (one step behind).

        Therefore:
          (H^T ψ)[m] = ½ · δπ[m]   · S_full[m]    (0 if m = N)
                     + ½ · δπ[m-1] · S_shift[m]   (0 if m = 0)

        where S_shift[m] = S_full[m] - ψ[m].

        Input:  psi_i  [..., nCells, N+1]
                ps     [..., nCells, 1]
        Output:        [..., nCells, N+1]
        """
        pi_i     = self.exner_interface(ps)
        delta_pi = pi_i[..., 1:] - pi_i[..., :-1]    # [..., N], > 0

        # S_full[m]  = Σ_{k=0}^{m}   ψ[k]  (cumsum)
        # S_shift[m] = Σ_{k=0}^{m-1} ψ[k]  (cumsum shifted right by one)
        S_full  = torch.cumsum(psi_i, dim=-1)          # [..., N+1]
        S_shift = torch.cat([
            torch.zeros_like(psi_i[..., :1]),
            S_full[..., :-1],
        ], dim=-1)                                     # [..., N+1]

        # Pad delta_pi with zeros at the boundaries
        zero     = torch.zeros_like(delta_pi[..., :1])
        dp_below = torch.cat([delta_pi, zero], dim=-1) # δπ[m],   0 at m=N
        dp_above = torch.cat([zero, delta_pi], dim=-1) # δπ[m-1], 0 at m=0

        return 0.5 * (S_full * dp_below + S_shift * dp_above)
