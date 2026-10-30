"""Residual-checked Krylov solves and hydrostatic Schur preconditioning.

GMRES uses the caller's physically scaled inner product (dry or moist). The
Schur factorization eliminates wind exactly; geometric multilevel cycles are
right preconditioners, not replacement equations. Geometry restriction preserves
pressure-weighted contents and prescribed terrain. No timestep formulas live here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import torch
from ...core.state import State
from ...geometry.mesh.multilevel import VoronoiMeshHierarchy

@dataclass(frozen=True)
class IMEXSolverConfig:
    """Controls for the matrix-free tensor-state ARK solve."""

    rtol: float | None = None
    atol: float = 0.0
    max_iterations: int = 40
    restart: int = 20
    reorthogonalize: bool = True
    implicit_operator: str = "coupled"
    preconditioner: str = "none"
    schur_tolerance_safety: float = 0.8
    multilevel_coarsening: int = 3
    multilevel_max_levels: int = 4
    multilevel_min_frequency: int = 3
    multilevel_stencil_size: int = 3
    multilevel_smoothing_iterations: int = 1
    multilevel_jacobi_weight: float = 0.85
    compile_kernels: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = None

    def resolved_rtol(self, dtype: torch.dtype) -> float:
        return float(self.rtol) if self.rtol is not None else (1.0e-8 if dtype == torch.float64 else 1.0e-5)

    def validate(self) -> None:
        if self.rtol is not None and self.rtol <= 0.0:
            raise ValueError("rtol must be positive")
        if self.atol < 0.0:
            raise ValueError("atol must be nonnegative")
        if self.max_iterations < 1 or self.restart < 1:
            raise ValueError("max_iterations and restart must be positive")
        if self.implicit_operator not in {"coupled", "fast_wave", "gravity_wave"}:
            raise ValueError("implicit_operator must be 'coupled', 'fast_wave', or 'gravity_wave'")
        if self.preconditioner not in {"none", "hydrostatic_schur", "multilevel_hydrostatic_schur"}:
            raise ValueError(
                "preconditioner must be 'none', 'hydrostatic_schur', or "
                "'multilevel_hydrostatic_schur'"
            )
        if not 0.0 < self.schur_tolerance_safety <= 1.0:
            raise ValueError("schur_tolerance_safety must be in (0, 1]")
        if self.multilevel_coarsening < 2:
            raise ValueError("multilevel_coarsening must be at least two")
        if self.multilevel_max_levels < 2:
            raise ValueError("multilevel_max_levels must be at least two")
        if self.multilevel_min_frequency < 1:
            raise ValueError("multilevel_min_frequency must be positive")
        if self.multilevel_stencil_size < 1:
            raise ValueError("multilevel_stencil_size must be positive")
        if self.multilevel_smoothing_iterations < 1:
            raise ValueError("multilevel_smoothing_iterations must be positive")
        if not 0.0 < self.multilevel_jacobi_weight < 1.0:
            raise ValueError("multilevel_jacobi_weight must be in (0, 1)")
        if self.compile_backend is not None and not str(self.compile_backend).strip():
            raise ValueError("compile_backend must be non-empty when provided")
        if self.compile_mode is not None and not str(self.compile_mode).strip():
            raise ValueError("compile_mode must be non-empty when provided")


@dataclass(frozen=True)
class IMEXStageInfo:
    stage: int
    implicit_coefficient: float
    iterations: int
    matvecs: int
    preconditioner_applications: int
    recycled_dimension: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    converged: bool


@dataclass(frozen=True)
class IMEXStepInfo:
    method: str
    linear_operator: str
    preconditioner: str
    dt: float
    stages: tuple[IMEXStageInfo, ...]
    converged: bool
    total_iterations: int
    total_matvecs: int
    total_preconditioner_applications: int


class IMEXConvergenceError(RuntimeError):
    """Raised when an adaptive ARK stage does not meet its residual target."""

    def __init__(self, stage_info: IMEXStageInfo) -> None:
        super().__init__(
            f"IMEX stage {stage_info.stage} failed after {stage_info.iterations} GMRES iterations "
            f"with relative residual {stage_info.relative_residual:.3e}"
        )
        self.stage_info = stage_info


def _values(state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return state.u_e, state.mtheta_i, state.ps


class _StateInnerProduct:
    """Fixed, field-aware scaling for Krylov convergence decisions."""

    def __init__(self, anchor: State) -> None:
        u, mtheta, ps = _values(anchor)
        tiny = torch.finfo(u.dtype).tiny
        self.u_scale = u.detach().square().mean().sqrt().clamp_min(torch.as_tensor(10.0, dtype=u.dtype, device=u.device))
        mtheta_floor = torch.as_tensor(1.0, dtype=mtheta.dtype, device=mtheta.device)
        self.mtheta_scale = mtheta.detach().square().mean().sqrt().clamp_min(mtheta_floor)
        self.ps_scale = ps.detach().square().mean().sqrt().clamp_min(torch.as_tensor(1.0e5, dtype=ps.dtype, device=ps.device))
        self.tiny = tiny

    def dot(self, left: State, right: State) -> torch.Tensor:
        lu, lm, lp = _values(left)
        ru, rm, rp = _values(right)
        return (
            (lu * ru).mean() / self.u_scale.square()
            + (lm * rm).mean() / self.mtheta_scale.square()
            + (lp * rp).mean() / self.ps_scale.square()
        )

    def norm(self, value: State) -> torch.Tensor:
        return self.dot(value, value).clamp_min(0.0).sqrt()


@dataclass(frozen=True)
class _KrylovRecycle:
    """One restarted-GMRES subspace for the next ARK stage of the same block."""

    residual_basis: tuple[State, ...]
    solution_basis: tuple[State, ...]
    hessenberg: torch.Tensor


@dataclass(frozen=True)
class _GMRESInfo:
    solution: State
    iterations: int
    matvecs: int
    preconditioner_applications: int
    recycled_dimension: int
    initial_residual: torch.Tensor
    final_residual: torch.Tensor
    converged: bool
    recycle: _KrylovRecycle | None


def _gmres(
    matvec: Callable[[State], State],
    rhs: State,
    inner: _StateInnerProduct,
    *,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    preconditioner: Callable[[State], State] | None = None,
    recycle: _KrylovRecycle | None = None,
) -> _GMRESInfo:
    """Restarted flexible GMRES with incremental Givens residual estimates.

    Adaptive iterations use the donor's incremental least-squares update: a
    true matrix residual is evaluated once per restart cycle rather than once
    per Arnoldi column. The full ARK-stage residual remains the acceptance
    criterion in :func:`_hydrostatic_schur_solve`. No fixed-iteration or
    unchecked stage acceptance path is retained.
    """

    x = rhs.zeros_like()
    rhs_norm = inner.norm(rhs)
    target = torch.maximum(
        torch.as_tensor(float(atol), dtype=rhs_norm.dtype, device=rhs_norm.device),
        torch.as_tensor(float(rtol), dtype=rhs_norm.dtype, device=rhs_norm.device) * rhs_norm,
    )
    recycled_dimension = 0
    if recycle is None:
        residual = rhs
    else:
        projected_rhs = torch.stack([inner.dot(vector, rhs) for vector in recycle.residual_basis])
        coefficients = torch.linalg.lstsq(recycle.hessenberg, projected_rhs).solution
        for coefficient, vector in zip(coefficients, recycle.solution_basis, strict=True):
            x = x + coefficient * vector
        projected_operator = rhs.zeros_like()
        for coefficient, vector in zip(
            recycle.hessenberg @ coefficients,
            recycle.residual_basis,
            strict=True,
        ):
            projected_operator = projected_operator + coefficient * vector
        residual = rhs - projected_operator
        recycled_dimension = len(recycle.solution_basis)
    iterations = 0
    matvecs = 0
    preconditioner_applications = 0
    recycle_out = recycle

    while iterations < int(max_iterations):
        beta = inner.norm(residual)
        if bool((beta == 0.0).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                beta,
                beta,
                True,
                recycle_out,
            )
        if bool((beta <= target).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                beta,
                beta,
                True,
                recycle_out,
            )

        width = min(int(restart), int(max_iterations) - iterations)
        basis: list[State] = [residual / beta]
        solution_basis: list[State] = []
        hessenberg = torch.zeros((width + 1, width), dtype=beta.dtype, device=beta.device)
        zero = beta * 0.0
        rotated_rhs = [beta] + [zero for _ in range(width)]
        rotation_cosines: list[torch.Tensor] = []
        rotation_sines: list[torch.Tensor] = []
        triangular_columns: list[list[torch.Tensor]] = []
        active_columns = 0

        for column in range(width):
            solution_vector = basis[column] if preconditioner is None else preconditioner(basis[column])
            if preconditioner is not None:
                preconditioner_applications += 1
            solution_basis.append(solution_vector)
            work = matvec(solution_vector)
            matvecs += 1
            for row, vector in enumerate(basis):
                coefficient = inner.dot(vector, work)
                hessenberg[row, column] = coefficient
                work = work - coefficient * vector
            if reorthogonalize:
                for row, vector in enumerate(basis):
                    coefficient = inner.dot(vector, work)
                    hessenberg[row, column] = hessenberg[row, column] + coefficient
                    work = work - coefficient * vector

            next_norm = inner.norm(work)
            hessenberg[column + 1, column] = next_norm
            arnoldi_breakdown = bool((next_norm <= torch.finfo(beta.dtype).eps).detach().cpu())
            if not arnoldi_breakdown:
                basis.append(work / next_norm)

            column_values = [hessenberg[row, column] for row in range(column + 2)]
            for row in range(column):
                upper, lower = column_values[row], column_values[row + 1]
                cosine, sine = rotation_cosines[row], rotation_sines[row]
                column_values[row] = cosine * upper + sine * lower
                column_values[row + 1] = -sine * upper + cosine * lower
            diagonal, subdiagonal = column_values[column], column_values[column + 1]
            radius = torch.hypot(diagonal, subdiagonal)
            if bool((radius <= torch.finfo(beta.dtype).eps).detach().cpu()):
                cosine, sine = torch.ones_like(radius), torch.zeros_like(radius)
            else:
                cosine, sine = diagonal / radius, subdiagonal / radius
            rotation_cosines.append(cosine)
            rotation_sines.append(sine)
            column_values[column] = cosine * diagonal + sine * subdiagonal
            column_values[column + 1] = zero
            triangular_columns.append(column_values[: column + 1])
            rotated_value = rotated_rhs[column]
            rotated_rhs[column] = cosine * rotated_value
            rotated_rhs[column + 1] = -sine * rotated_value

            iterations += 1
            active_columns = column + 1
            if bool((rotated_rhs[active_columns].abs() <= target).detach().cpu()) or arnoldi_breakdown:
                break

        coefficients = [zero for _ in range(active_columns)]
        for row in range(active_columns - 1, -1, -1):
            value = rotated_rhs[row]
            for column in range(row + 1, active_columns):
                value = value - triangular_columns[column][row] * coefficients[column]
            diagonal = triangular_columns[row][row]
            coefficients[row] = (
                zero
                if bool((diagonal.abs() <= torch.finfo(beta.dtype).eps).detach().cpu())
                else value / diagonal
            )
        for coefficient, vector in zip(coefficients, solution_basis, strict=True):
            x = x + coefficient * vector
        recycle_out = _KrylovRecycle(
            residual_basis=tuple(basis),
            solution_basis=tuple(solution_basis),
            hessenberg=hessenberg[: len(basis), :active_columns].clone(),
        )
        residual = rhs - matvec(x)
        matvecs += 1
        final = inner.norm(residual)
        if bool((final <= target).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                rhs_norm,
                final,
                True,
                recycle_out,
            )

    final = inner.norm(residual)
    return _GMRESInfo(
        x,
        iterations,
        matvecs,
        preconditioner_applications,
        recycled_dimension,
        rhs_norm,
        final,
        bool((final <= target).detach().cpu()),
        recycle_out,
    )


class _HydrostaticSchurSystem:
    """Exact reduced factorization of the analytic gravity-wave stage block."""

    def __init__(self, operator: DryPELinearOperator, factor: float) -> None:
        if operator.implicit_operator != "gravity_wave":
            raise ValueError("hydrostatic_schur requires implicit_operator='gravity_wave'")
        self.operator = operator
        self.factor = float(factor)
        self.anchor = operator.anchor
        self.zero = self.anchor.zeros_like()

    def make_state(self, u: torch.Tensor, mtheta: torch.Tensor, ps: torch.Tensor) -> State:
        return State(
            u,
            mtheta,
            ps,
            None if self.anchor.tracer_m is None else torch.zeros_like(self.anchor.tracer_m),
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )

    def thermodynamic(self, mtheta: torch.Tensor, ps: torch.Tensor) -> State:
        return self.make_state(self.zero.u_e, mtheta, ps)

    def wind_response(self, value: State) -> torch.Tensor:
        return self.operator.gravity_wave_wind(value.mtheta_i, value.ps)

    def thermodynamic_response(self, u: torch.Tensor) -> State:
        mtheta, ps = self.operator.gravity_wave_thermodynamic(u)
        return self.thermodynamic(mtheta, ps)

    def rhs(self, value: State) -> State:
        """Eliminate wind from the full stage right-hand side."""

        wind_rhs = self.thermodynamic_response(value.u_e)
        return self.thermodynamic(
            value.mtheta_i + self.factor * wind_rhs.mtheta_i,
            value.ps + self.factor * wind_rhs.ps,
        )

    def matvec(self, value: State) -> State:
        """Apply ``I - factor**2 H G`` to a thermodynamic perturbation."""

        feedback = self.thermodynamic_response(self.wind_response(value))
        return self.thermodynamic(
            value.mtheta_i - self.factor * self.factor * feedback.mtheta_i,
            value.ps - self.factor * self.factor * feedback.ps,
        )

    def back_substitute(self, rhs: State, thermodynamic: State) -> State:
        """Recover wind after solving the reduced thermodynamic system."""

        return State(
            rhs.u_e + self.factor * self.wind_response(thermodynamic),
            thermodynamic.mtheta_i,
            thermodynamic.ps,
            # The analytic gravity-wave block has zero tracer tendency.
            # Its exact stage inverse is therefore the identity on tracers.
            None if rhs.tracer_m is None else rhs.tracer_m,
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )


@dataclass(frozen=True)
class _ModalHelmholtzSetup:
    """Cached vertical-mode coefficients for a geometric Schur V-cycle."""

    transform: torch.Tensor
    reconstruct: torch.Tensor
    diagonal_eigenvalues: torch.Tensor
    offdiagonal_eigenvalues: torch.Tensor
    partner_indices: torch.Tensor
    laplacian_diagonals: tuple[torch.Tensor, ...]
    coarse_horizontal_eigenvalues: torch.Tensor
    coarse_horizontal_eigenvectors: torch.Tensor
    coarse_sqrt_area: torch.Tensor


class _MultilevelHydrostaticSchurPreconditioner:
    """Modal geometric V-cycle for the reduced analytic Schur system.

    A coarse VFS state supplies the horizontal/vertical coupling fit; the
    finest reduced system remains the exact frozen analytic Schur operator.
    This is therefore a right preconditioner, not a replacement equation or a
    new dynamics tendency.
    """

    def __init__(
        self,
        system: _HydrostaticSchurSystem,
        hierarchy: VoronoiMeshHierarchy,
        inner: _StateInnerProduct,
        *,
        smoothing_iterations: int,
        jacobi_weight: float,
        modal_setup: _ModalHelmholtzSetup | None = None,
    ) -> None:
        if hierarchy.meshes[0] is not system.anchor.mesh:
            raise ValueError("multilevel hierarchy does not match the Schur operator mesh")
        self.system = system
        self.hierarchy = hierarchy
        self.inner = inner
        self.smoothing_iterations = int(smoothing_iterations)
        self.jacobi_weight = float(jacobi_weight)
        self.mtheta_scale = inner.mtheta_scale
        self.ps_scale = inner.ps_scale
        if modal_setup is None:
            self._initialize_modal_helmholtz()
        else:
            self._load_modal_setup(modal_setup)

    def modal_setup(self) -> _ModalHelmholtzSetup:
        return _ModalHelmholtzSetup(
            self.modal_transform,
            self.modal_reconstruct,
            self.modal_diagonal_eigenvalues,
            self.modal_offdiagonal_eigenvalues,
            self.modal_partner_indices,
            self.modal_laplacian_diagonals,
            self.coarse_horizontal_eigenvalues,
            self.coarse_horizontal_eigenvectors,
            self.coarse_sqrt_area,
        )

    def _load_modal_setup(self, setup: _ModalHelmholtzSetup) -> None:
        self.modal_transform = setup.transform
        self.modal_reconstruct = setup.reconstruct
        self.modal_diagonal_eigenvalues = setup.diagonal_eigenvalues
        self.modal_offdiagonal_eigenvalues = setup.offdiagonal_eigenvalues
        self.modal_partner_indices = setup.partner_indices
        self.modal_laplacian_diagonals = setup.laplacian_diagonals
        self.coarse_horizontal_eigenvalues = setup.coarse_horizontal_eigenvalues
        self.coarse_horizontal_eigenvectors = setup.coarse_horizontal_eigenvectors
        self.coarse_sqrt_area = setup.coarse_sqrt_area

    @staticmethod
    def _scalar_laplacian(mesh, value: torch.Tensor) -> torch.Tensor:
        return -(mesh.radius * mesh.radius) * mesh.div(mesh.grad(value))

    @staticmethod
    def _scalar_laplacian_diagonal(mesh, reference: torch.Tensor) -> torch.Tensor:
        edge_coefficient = (mesh.dv_edge / mesh.dc_edge)[mesh.safe_edges_on_cell]
        edge_coefficient = torch.where(mesh.valid_edges_on_cell, edge_coefficient, torch.zeros_like(edge_coefficient))
        return (
            mesh.radius
            * mesh.radius
            * edge_coefficient.sum(dim=-1)
            / mesh.cell_area
        ).to(dtype=reference.dtype, device=reference.device)

    def _pack_normalized(self, value: State) -> torch.Tensor:
        return torch.cat(
            (
                value.mtheta_i / self.mtheta_scale,
                value.ps / self.ps_scale,
            ),
            dim=-1,
        )

    def _unpack_normalized(self, value: torch.Tensor) -> State:
        return self.system.thermodynamic(
            value[..., :-1] * self.mtheta_scale,
            value[..., -1:] * self.ps_scale,
        )

    def _coarse_system(self) -> _HydrostaticSchurSystem:
        from .dry_ark import DryPELinearOperator
        from ...geometry.mesh.multilevel import restrict_state
        anchor = restrict_state(self.system.anchor, self.hierarchy)
        operator = DryPELinearOperator(
            self.system.operator.dynamics,
            anchor,
            implicit_operator="gravity_wave",
            tendency=self.system.operator.tendency,
        )
        return _HydrostaticSchurSystem(operator, self.system.factor)

    def _initialize_modal_helmholtz(self) -> None:
        coarse_system = self._coarse_system()
        grid = self.hierarchy.meshes[-1]
        anchor = coarse_system.anchor
        n_interfaces = anchor.mtheta_i.shape[-1]
        n_modes = n_interfaces + 1
        phi = grid.cell_xyz[:, 0].to(dtype=anchor.ps.dtype, device=anchor.ps.device).unsqueeze(-1)
        laplacian_phi = self._scalar_laplacian(grid, phi)
        area = grid.cell_area.to(dtype=phi.dtype, device=phi.device).unsqueeze(-1)
        area = area / area.sum().clamp_min(torch.finfo(phi.dtype).tiny)
        denominator = (laplacian_phi.square() * area).sum().clamp_min(torch.finfo(phi.dtype).tiny)
        columns: list[torch.Tensor] = []
        for mode in range(n_modes):
            mtheta = torch.zeros_like(anchor.mtheta_i)
            ps = torch.zeros_like(anchor.ps)
            if mode < n_interfaces:
                mtheta[..., :, mode] = phi[..., 0] * self.mtheta_scale
            else:
                ps[..., :, 0] = phi[..., 0] * self.ps_scale
            value = coarse_system.thermodynamic(mtheta, ps)
            response = self._pack_normalized(coarse_system.matvec(value) - value)
            coefficient = (response * laplacian_phi * area).sum(dim=-2) / denominator
            while coefficient.ndim > 1:
                coefficient = coefficient.mean(dim=0)
            columns.append(coefficient)
        coupling = torch.stack(columns, dim=1).detach()
        eigenvalues, eigenvectors = torch.linalg.eig(coupling.to(device="cpu"))
        eigenvectors_inverse = torch.linalg.inv(eigenvectors)
        tolerance = 128.0 * torch.finfo(coupling.dtype).eps * eigenvalues.abs().max().clamp_min(1.0)
        transform_rows: list[torch.Tensor] = []
        reconstruct_columns: list[torch.Tensor] = []
        diagonal_values: list[torch.Tensor] = []
        offdiagonal_values: list[torch.Tensor] = []
        partner_indices: list[int] = []
        for mode in range(n_modes):
            imaginary = eigenvalues[mode].imag
            if float(imaginary.abs()) <= float(tolerance):
                channel = len(transform_rows)
                transform_rows.append(eigenvectors_inverse[mode].real)
                reconstruct_columns.append(eigenvectors[:, mode].real)
                diagonal_values.append(eigenvalues[mode].real)
                offdiagonal_values.append(torch.zeros_like(imaginary))
                partner_indices.append(channel)
            elif float(imaginary) > 0.0:
                channel = len(transform_rows)
                transform_rows.extend((eigenvectors_inverse[mode].real, eigenvectors_inverse[mode].imag))
                reconstruct_columns.extend((2.0 * eigenvectors[:, mode].real, -2.0 * eigenvectors[:, mode].imag))
                diagonal_values.extend((eigenvalues[mode].real, eigenvalues[mode].real))
                offdiagonal_values.extend((-imaginary, imaginary))
                partner_indices.extend((channel + 1, channel))
        if len(transform_rows) != n_modes:
            raise RuntimeError("vertical Schur modes did not form real conjugate blocks")
        self.modal_transform = torch.stack(transform_rows).to(dtype=phi.dtype, device=phi.device)
        self.modal_reconstruct = torch.stack(reconstruct_columns, dim=1).to(dtype=phi.dtype, device=phi.device)
        self.modal_diagonal_eigenvalues = torch.stack(diagonal_values).to(dtype=phi.dtype, device=phi.device)
        self.modal_offdiagonal_eigenvalues = torch.stack(offdiagonal_values).to(dtype=phi.dtype, device=phi.device)
        self.modal_partner_indices = torch.as_tensor(partner_indices, dtype=torch.long, device=phi.device)
        self.modal_laplacian_diagonals = tuple(
            self._scalar_laplacian_diagonal(mesh, phi) for mesh in self.hierarchy.meshes
        )

        coarse_count = grid.nCells
        identity = torch.eye(coarse_count, dtype=phi.dtype, device=phi.device)
        laplacian_matrix = self._scalar_laplacian(grid, identity).mT
        sqrt_area = grid.cell_area.to(dtype=phi.dtype, device=phi.device).sqrt()
        symmetric_laplacian = sqrt_area.unsqueeze(-1) * laplacian_matrix / sqrt_area.unsqueeze(0).clamp_min(
            torch.finfo(phi.dtype).tiny
        )
        symmetric_laplacian = 0.5 * (symmetric_laplacian + symmetric_laplacian.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_laplacian.to(device="cpu"))
        self.coarse_horizontal_eigenvalues = eigenvalues.clamp_min(0.0).to(dtype=phi.dtype, device=phi.device)
        self.coarse_horizontal_eigenvectors = eigenvectors.to(dtype=phi.dtype, device=phi.device)
        self.coarse_sqrt_area = sqrt_area

    def _modal_matvec(self, level: int, value: torch.Tensor) -> torch.Tensor:
        laplacian = self._scalar_laplacian(self.hierarchy.meshes[level], value)
        diagonal = self.modal_diagonal_eigenvalues.to(dtype=value.dtype, device=value.device)
        offdiagonal = self.modal_offdiagonal_eigenvalues.to(dtype=value.dtype, device=value.device)
        partners = self.modal_partner_indices.to(device=value.device)
        return value + diagonal * laplacian + offdiagonal * laplacian[..., partners]

    def _modal_coarse_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        sqrt_area = self.coarse_sqrt_area.to(dtype=rhs.dtype, device=rhs.device)
        area_view = sqrt_area.reshape((1,) * (rhs.ndim - 2) + (-1, 1))
        weighted = rhs * area_view
        eigenvectors = self.coarse_horizontal_eigenvectors.to(dtype=rhs.dtype, device=rhs.device)
        coefficients = torch.einsum("ch,...cm->...mh", eigenvectors, weighted)
        horizontal = self.coarse_horizontal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device)
        diagonal = 1.0 + self.modal_diagonal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device).unsqueeze(-1) * horizontal
        offdiagonal = self.modal_offdiagonal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device).unsqueeze(-1) * horizontal
        partners = self.modal_partner_indices.to(device=rhs.device)
        partner_diagonal = diagonal[partners]
        partner_offdiagonal = offdiagonal[partners]
        scale = diagonal * partner_diagonal - offdiagonal * partner_offdiagonal
        coefficients = (partner_diagonal * coefficients - offdiagonal * coefficients[..., partners, :]) / scale
        weighted_solution = torch.einsum("ch,...mh->...cm", eigenvectors, coefficients)
        return weighted_solution / area_view

    def _modal_v_cycle(self, level: int, rhs: torch.Tensor) -> torch.Tensor:
        if level == len(self.hierarchy.meshes) - 1:
            return self._modal_coarse_solve(rhs)
        laplacian_diagonal = self.modal_laplacian_diagonals[level].to(dtype=rhs.dtype, device=rhs.device)
        diagonal = 1.0 + laplacian_diagonal.unsqueeze(-1) * self.modal_diagonal_eigenvalues.to(
            dtype=rhs.dtype, device=rhs.device
        )
        offdiagonal = laplacian_diagonal.unsqueeze(-1) * self.modal_offdiagonal_eigenvalues.to(
            dtype=rhs.dtype, device=rhs.device
        )
        partners = self.modal_partner_indices.to(device=rhs.device)
        partner_diagonal = diagonal[..., partners]
        partner_offdiagonal = offdiagonal[..., partners]
        diagonal_scale = diagonal * partner_diagonal - offdiagonal * partner_offdiagonal

        def smooth(current: torch.Tensor) -> torch.Tensor:
            return self.jacobi_weight * (
                partner_diagonal * current - offdiagonal * current[..., partners]
            ) / diagonal_scale

        correction = torch.zeros_like(rhs)
        residual = rhs
        for _ in range(self.smoothing_iterations):
            increment = smooth(residual)
            correction = correction + increment
            residual = residual - self._modal_matvec(level, increment)
        coarse_rhs = self.hierarchy.transfers[level].restrict(residual)
        assert isinstance(coarse_rhs, torch.Tensor)
        coarse_correction = self._modal_v_cycle(level + 1, coarse_rhs)
        prolonged = self.hierarchy.transfers[level].prolong(coarse_correction)
        assert isinstance(prolonged, torch.Tensor)
        correction = correction + prolonged
        residual = residual - self._modal_matvec(level, prolonged)
        for _ in range(self.smoothing_iterations):
            increment = smooth(residual)
            correction = correction + increment
            residual = residual - self._modal_matvec(level, increment)
        return correction

    def __call__(self, value: State) -> State:
        normalized = self._pack_normalized(value)
        modal = torch.einsum(
            "ij,...cj->...ci",
            self.modal_transform.to(dtype=normalized.dtype, device=normalized.device),
            normalized,
        )
        modal_correction = self._modal_v_cycle(0, modal)
        normalized_correction = torch.einsum(
            "ij,...cj->...ci",
            self.modal_reconstruct.to(dtype=modal_correction.dtype, device=modal_correction.device),
            modal_correction,
        )
        return self._unpack_normalized(normalized_correction)


def _hydrostatic_schur_solve(
    operator: DryPELinearOperator,
    rhs: State,
    inner: _StateInnerProduct,
    *,
    factor: float,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    tolerance_safety: float,
    preconditioner: Callable[[State], State] | None = None,
    recycle: _KrylovRecycle | None = None,
) -> _GMRESInfo:
    """Solve the analytic gravity-wave stage by block Schur elimination.

    For the validated gravity-wave block, ``L = [[0, G], [H, 0]]`` where
    ``G`` maps thermodynamic perturbations to wind and ``H`` maps wind to the
    centered thermodynamic/continuity response.  Eliminating wind yields the
    lower-dimensional exact system ``(I - factor**2 H G) z = rhs_z + factor H
    rhs_u``.  This is an exact factorization of the *ported analytic block*;
    only the iterative reduced solve is approximate.
    """

    system = _HydrostaticSchurSystem(operator, factor)
    reduced_rhs = system.rhs(rhs)

    full_rhs_norm = inner.norm(rhs)
    full_target = torch.maximum(
        torch.as_tensor(float(atol), dtype=full_rhs_norm.dtype, device=full_rhs_norm.device),
        torch.as_tensor(float(rtol), dtype=full_rhs_norm.dtype, device=full_rhs_norm.device) * full_rhs_norm,
    )
    reduced_rhs_norm = inner.norm(reduced_rhs)
    reduced_rtol = float(
        (
            torch.as_tensor(float(tolerance_safety), dtype=full_target.dtype, device=full_target.device)
            * full_target
            / reduced_rhs_norm.clamp_min(inner.tiny)
        )
        .clamp(min=torch.finfo(full_target.dtype).eps, max=1.0)
        .detach()
        .cpu()
    )
    reduced = _gmres(
        system.matvec,
        reduced_rhs,
        inner,
        rtol=reduced_rtol,
        atol=0.0,
        max_iterations=max_iterations,
        restart=restart,
        reorthogonalize=reorthogonalize,
        preconditioner=preconditioner,
        recycle=recycle,
    )
    correction = system.back_substitute(rhs, reduced.solution)
    full_residual = rhs - (correction - factor * operator.apply(correction))
    final_norm = inner.norm(full_residual)
    return _GMRESInfo(
        solution=correction,
        iterations=reduced.iterations,
        matvecs=reduced.matvecs + 1,
        preconditioner_applications=reduced.preconditioner_applications,
        recycled_dimension=reduced.recycled_dimension,
        initial_residual=full_rhs_norm,
        final_residual=final_norm,
        converged=bool((final_norm <= full_target).detach().cpu()),
        recycle=reduced.recycle,
    )

