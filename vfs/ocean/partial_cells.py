"""Partial bottom cells on common reference levels, with split wet faces.

Depths increase downwards from mean sea level. A zero thickness is padding
and has no associated mass, kinetic energy or flux. Edge orientation is
from the first to the second cell in each pair.
"""

from dataclasses import dataclass

import torch

from .ale import remap_contents


def reference_thickness(depth, nominal, *, merge_fraction=0.25):
    """Extend the deepest nominal level into trenches. Merge small cuts upwards."""
    if depth.ndim != 1 or nominal.ndim != 1 or nominal.numel() == 0:
        raise ValueError(
            "expected a vector of depths and a vector of nominal thicknesses"
        )
    if (
        not nominal.is_floating_point()
        or depth.dtype != nominal.dtype
        or depth.device != nominal.device
        or not torch.isfinite(depth).all()
        or not torch.isfinite(nominal).all()
        or (depth < 0).any()
        or (nominal <= 0).any()
        or not 0 <= merge_fraction <= 1
    ):
        raise ValueError("invalid depths, nominal thicknesses or merge fraction")
    tops = torch.cat((nominal.new_zeros(1), nominal.cumsum(0)[:-1]))
    available = (depth[:, None] - tops).clamp_min(0)
    h = torch.cat(
        (torch.minimum(available[:, :-1], nominal[:-1]), available[:, -1:]), 1
    )
    cuts = (h[:, 1:] > 0) & (h[:, 1:] < merge_fraction * nominal[1:])
    transfer = torch.where(cuts, h[:, 1:], 0)
    zero = torch.zeros_like(h[:, :1])
    return h + torch.cat((transfer, zero), 1) - torch.cat((zero, transfer), 1)


def _counts(h):
    if h.ndim != 2 or h.shape[1] == 0 or not h.is_floating_point():
        raise ValueError("expected floating point (columns, levels) thickness")
    wet = h > 0
    if (
        not torch.isfinite(h).all()
        or (h < 0).any()
        or (wet[:, 1:] & ~wet[:, :-1]).any()
    ):
        raise ValueError("wet layers must form a finite positive prefix")
    return wet.sum(1)


def remap_active_contents(old_h, content, new_h, *, method="plm"):
    """Conservative ALE of layer inventories, including changes in active count."""
    old_count, new_count = _counts(old_h), _counts(new_h)
    if (
        old_h.shape[0] != new_h.shape[0]
        or old_h.dtype != new_h.dtype
        or old_h.device != new_h.device
        or content.dtype != old_h.dtype
        or content.device != old_h.device
        or content.shape[:2] != old_h.shape
        or content.ndim not in (2, 3)
        or method not in ("plm", "pcm")
    ):
        raise ValueError("incompatible remap inputs")
    if ((old_count == 0) != (new_count == 0)).any() or (content[old_h == 0] != 0).any():
        raise ValueError("ALE cannot replace dry columns or discard inactive inventory")
    out = content.new_zeros((*new_h.shape, *content.shape[2:]))
    pairs = torch.stack((old_count, new_count), 1)
    for old, new in pairs.unique(dim=0).tolist():
        if old:
            rows = (old_count == old) & (new_count == new)
            out[rows, :new] = remap_contents(
                old_h[rows, :old], content[rows, :old], new_h[rows, :new], method=method
            )
    return out


def layer_interfaces(depth, h, count=None):
    """Physical depths of the layer interfaces, relative to mean sea level."""
    if count is None:
        count = _counts(h)
    # Anchor interfaces at the prescribed seabed as the free surface moves.
    cumulative = torch.cat((torch.zeros_like(h[:, :1]), h.cumsum(1)), 1)
    surface = depth - cumulative[:, -1]
    # At rest, subtracting independently accumulated column depths can
    # turn one shared level into two faces a few ulps apart. Resolve
    # displacements only above the roundoff scale of the bed datum.
    tol = 8 * torch.finfo(h.dtype).eps * torch.maximum(depth, cumulative[:, -1])
    surface = torch.where(surface.abs() <= tol, surface - surface.detach(), surface)
    interfaces = surface[:, None] + cumulative
    at_bed = torch.arange(h.shape[1] + 1, device=h.device)[None, :] >= count[:, None]
    interfaces = torch.where(at_bed, depth[:, None], interfaces)
    if ((interfaces.diff(dim=1) <= 0) & (h > 0)).any():
        raise ValueError("wet interfaces indistinguishable at this precision")
    return interfaces


@dataclass(frozen=True)
class WetFaces:
    """Packed intersections, ordered by edge then depth. Owners index flattened h."""

    edge: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    opening: torch.Tensor
    n_edges: int
    cell_shape: torch.Size

    @classmethod
    def from_columns(cls, depth, h, cells):
        count = _counts(h)
        if (
            depth.shape != h.shape[:1]
            or depth.dtype != h.dtype
            or depth.device != h.device
            or not torch.isfinite(depth).all()
            or (depth < 0).any()
            or cells.ndim != 2
            or cells.shape[1] != 2
            or cells.dtype != torch.long
            or cells.device != h.device
            or ((cells < -1) | (cells >= len(h))).any()
        ):
            raise ValueError("invalid bathymetry or edge connectivity")
        interfaces = layer_interfaces(depth, h, count)
        valid_edges = torch.where((cells >= 0).all(1))[0]
        neighbors = cells[valid_edges]
        left, right = interfaces[neighbors[:, 0]], interfaces[neighbors[:, 1]]
        top = torch.maximum(left[:, :1], right[:, :1])
        bottom = torch.minimum(left[:, -1:], right[:, -1:])
        bounds = torch.cat((left, right), 1).clamp(min=top, max=bottom).sort(1).values
        widths = bounds.diff(dim=1)
        edge, slot = torch.where(widths > 0)
        starts = bounds[:, :-1].contiguous()
        # The lower endpoint can round onto the next interface for thin cuts.
        # Locate by the upper endpoint, assigning exact ties to the layer below.
        li = torch.searchsorted(left.contiguous(), starts, right=True)[edge, slot] - 1
        ri = torch.searchsorted(right.contiguous(), starts, right=True)[edge, slot] - 1
        return cls(
            valid_edges[edge],
            neighbors[edge, 0] * h.shape[1] + li,
            neighbors[edge, 1] * h.shape[1] + ri,
            widths[edge, slot],
            len(cells),
            h.shape,
        )

    def exchange(self, flux):
        """Integrated signed influx per cell and layer, for one oriented flux per opening."""
        if flux.shape != self.opening.shape:
            raise ValueError("expected a flux for every wet face segment")
        out = flux.new_zeros(self.cell_shape.numel())
        return (
            out.index_add(0, self.left, -flux)
            .index_add(0, self.right, flux)
            .view(self.cell_shape)
        )

    def jump(self, conjugate):
        """Adjoint to exchange in inventory coordinates."""
        if conjugate.shape != self.cell_shape:
            raise ValueError("conjugate must match cell geometry")
        flat = conjugate.flatten()
        return flat[self.right] - flat[self.left]

    def volume_flux(self, velocity, edge_length):
        return velocity * edge_length[self.edge] * self.opening

    def kinetic_energy(self, velocity, dual_area, rho0):
        return 0.5 * rho0 * dual_area[self.edge] * self.opening * velocity.square()

    def remap_velocity(self, velocity, target, dual_area, rho0):
        """Return new velocity and signed KE change per edge (joules).

        For vertical rezoning on the same mesh with fixed column totals.
        PCM conserves integrated normal momentum and dissipates unresolved shear.
        """
        if target.n_edges != self.n_edges or velocity.shape != self.opening.shape:
            raise ValueError("incompatible edge layout")

        def padded(faces):
            counts = faces.edge.bincount(minlength=self.n_edges)
            n = int(counts.max()) if counts.numel() else 0
            h = faces.opening.new_zeros((self.n_edges, max(1, n)))
            offset = counts.cumsum(0) - counts
            slot = torch.arange(len(faces.edge), device=h.device) - offset[faces.edge]
            index = (faces.edge, slot)
            h[index] = faces.opening
            return h, index

        old, src = padded(self)
        new, dst = padded(target)
        momentum = torch.zeros_like(old)
        momentum[src] = velocity * self.opening
        mapped = (
            remap_active_contents(old, momentum, new, method="pcm")[dst]
            / target.opening
        )
        change = velocity.new_zeros(self.n_edges)
        change = change.index_add(
            0, target.edge, target.kinetic_energy(mapped, dual_area, rho0)
        )
        change = change.index_add(
            0, self.edge, -self.kinetic_energy(velocity, dual_area, rho0)
        )
        return mapped, change


@dataclass(frozen=True)
class ShelfRemap:
    temp_h: torch.Tensor
    salt_h: torch.Tensor
    velocity: torch.Tensor
    faces: WetFaces
    edge_kinetic_change: torch.Tensor
    column_energy_residual: torch.Tensor
    column_entropy_change: torch.Tensor


def remap_roquet(
    depth,
    old_h,
    new_h,
    temp_h,
    salt_h,
    velocity,
    cells,
    cell_area,
    dual_area,
    energy,
    *,
    method="plm",
):
    """Rezone at fixed bathymetry using the supplied RoquetEnergy and its calorics.

    Split each edge's KE change equally between its two adjacent columns.
    Free-surface energy is invariant under this fixed-volume operation.
    Budget units are joules and J/K. Reservoir contents (TKE, EKE) can be
    remapped separately with remap_active_contents at fixed inventory.
    """
    from .stratified.remap import minimum_change_step

    old = WetFaces.from_columns(depth, old_h, cells)
    new = WetFaces.from_columns(depth, new_h, cells)
    mapped = remap_active_contents(
        old_h, torch.stack((temp_h, salt_h), -1), new_h, method=method
    )
    u, dke = old.remap_velocity(velocity, new, dual_area, energy.rho0)
    mass_area = energy.rho0 * cell_area
    if (~torch.isfinite(mass_area) | (mass_area <= 0)).any() or (
        ~torch.isfinite(dual_area) | (dual_area <= 0)
    ).any():
        raise ValueError("positive finite areas required")
    kinetic = torch.zeros_like(depth)
    for side in cells.T:
        valid = side >= 0
        kinetic = kinetic.index_add(0, side[valid], 0.5 * dke[valid])
    cp = energy.config.cp
    nodes, weights = energy._quadrature(old_h)

    def quantities(h, ht, hs):
        divisor = torch.where(h > 0, h, 1.0)
        t, s = ht / divisor, hs / divisor
        if ((h > 0) & ((t <= -2.5) | (t >= 40.0) | (s < 0.0) | (s > 42.0))).any():
            raise ValueError("ALE candidate outside Roquet caloric domain")
        bottom = h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - h
        z = bottom[..., None] + h[..., None] * nodes
        psi, pt, _, _ = energy.parcel_potential(t[..., None], s[..., None], z)
        enthalpy = h * (cp * t + (psi * weights).sum(-1))
        entropy = h * cp * torch.log1p(t / 273.15)
        return enthalpy, entropy, cp + (pt * weights).sum(-1), cp / (273.15 + t)

    old_e, old_s, _, _ = quantities(old_h, temp_h, salt_h)
    wet = new_h > 0
    denom = torch.where(wet, new_h, 1.0)
    theta0 = mapped[..., 0] / denom
    theta = theta0
    salt = mapped[..., 1]
    eps = torch.finfo(theta.dtype).eps
    etol = 256 * eps * (mass_area * old_e.abs().sum(1) + kinetic.abs()).clamp_min(1.0)
    stol = 512 * eps * (mass_area * old_s.abs().sum(1)).clamp_min(1.0)
    count = wet.sum(1)
    for iteration in range(25):
        ht = new_h * theta
        e, s, grad, ds = quantities(new_h, ht, salt)
        residual = mass_area * (e.sum(1) - old_e.sum(1)) + kinetic
        entropy = mass_area * (s.sum(1) - old_s.sum(1))
        converged = (residual.abs() <= etol) & (entropy >= -stol)
        if converged.all():
            return ShelfRemap(ht, salt, u, new, dke, residual, entropy)
        if iteration == 24:
            break
        candidate = theta.clone()
        for n in count[~converged].unique().tolist():
            if n == 0:
                raise ValueError("dry column has a nonzero energy budget")
            rows = (count == n) & ~converged
            index = (rows, slice(0, n))
            h = new_h[index]
            total = h.sum(1)
            w = h / total[:, None]
            mass = mass_area[rows] * total
            step, _ = minimum_change_step(
                theta[index] - theta0[index],
                w * grad[index],
                w * ds[index],
                residual[rows] / mass,
                entropy[rows] / mass,
                w,
                0.0,
            )
            candidate[index] = theta[index] + step
        theta = candidate
    raise FloatingPointError(
        "partial-cell ALE could not close column energy and entropy"
    )
