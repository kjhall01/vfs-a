"""Restarted flexible GMRES, independent of component state and units.

Vectors provide zeros_like and linear arithmetic; the supplied inner product
owns scaling and dot/norm. Atmospheric stage acceptance stays with atmosphere.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
import torch

Vector = TypeVar("Vector")

@dataclass(frozen=True)
class _KrylovRecycle:
    """One restarted-GMRES subspace for the next ARK stage of the same block."""

    residual_basis: tuple[Vector, ...]
    solution_basis: tuple[Vector, ...]
    hessenberg: torch.Tensor


@dataclass(frozen=True)
class _GMRESInfo:
    solution: Vector
    iterations: int
    matvecs: int
    preconditioner_applications: int
    recycled_dimension: int
    initial_residual: torch.Tensor
    final_residual: torch.Tensor
    converged: bool
    recycle: _KrylovRecycle | None


def _gmres(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    inner: Any,
    *,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    preconditioner: Callable[[Vector], Vector] | None = None,
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
        basis: list[Vector] = [residual / beta]
        solution_basis: list[Vector] = []
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


