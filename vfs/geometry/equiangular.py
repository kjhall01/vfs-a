"""Conservative C-grid diagnostics on a global spherical latitude–longitude grid."""
from __future__ import annotations

import math
import torch

from .remap import LatLonGrid


class EquiangularGrid(LatLonGrid):
    """Global cell-centered grid with matched finite-volume operators.

    Scalars have shape (..., nlat, nlon). East faces have the same shape:
    index j lies between cells j and j+1 (periodic). North faces have shape
    (..., nlat+1, nlon), including the two zero-area polar boundaries.
    Face components are positive eastward/northward. Geometry starts in CPU
    float64; use .to() before supplying fields with matching dtype/device.
    """

    def __init__(self, nlat: int, nlon: int, *, radius: float = 6.371e6):
        if isinstance(nlat, bool) or not isinstance(nlat, int) or nlat < 2:
            raise ValueError("nlat must be an integer >= 2")
        if isinstance(nlon, bool) or not isinstance(nlon, int) or nlon < 3:
            raise ValueError("nlon must be an integer >= 3")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        lat_b = torch.linspace(-90., 90., nlat + 1, dtype=torch.float64)
        lon_b = torch.linspace(0., 360., nlon + 1, dtype=torch.float64)
        super().__init__((lat_b[:-1]+lat_b[1:])/2,
                         (lon_b[:-1]+lon_b[1:])/2, radius=radius)
        dphi, dlam = math.pi/nlat, 2*math.pi/nlon
        # Meridional faces follow meridians; zonal faces follow parallels.
        east_length = torch.full(self.shape, radius*dphi, dtype=torch.float64)
        north_length = (radius*dlam*torch.cos(torch.deg2rad(lat_b)))[:, None].expand(nlat+1, nlon).clone()
        north_length[[0, -1]] = 0.  # Exact zero, not cos(pi/2) roundoff.
        east_distance = (radius*dlam*torch.cos(torch.deg2rad(self.lat_deg)))[:, None].expand(self.shape).clone()
        north_distance = torch.full((nlat+1, nlon), radius*dphi, dtype=torch.float64)
        # Dual cells span consecutive cell-center latitudes. Each pole is one
        # cap, stored as repeated values with 1/nlon of its area per entry.
        center_sin = torch.sin(torch.deg2rad(self.lat_deg))
        dual_bounds = torch.cat((center_sin.new_tensor([-1.]), center_sin,
                                 center_sin.new_tensor([1.])))
        corner_area = (radius**2*dlam*torch.diff(dual_bounds))[:, None].expand(nlat+1, nlon).clone()
        lower_fraction = ((center_sin-torch.sin(torch.deg2rad(lat_b[:-1])))
                          / torch.diff(torch.sin(torch.deg2rad(lat_b))))[:, None]
        for name, value in dict(corner_area=corner_area, lower_corner_fraction=lower_fraction,
                                lat_bounds_deg=lat_b, lon_bounds_deg=lon_b,
                                east_face_length=east_length, north_face_length=north_length,
                                east_distance=east_distance, north_distance=north_distance,
                                east_weight=east_length*east_distance,
                                north_weight=north_length*north_distance).items():
            self.register_buffer(name, value)

    @classmethod
    def regular(cls, nlat: int, nlon: int, *, radius: float = 6.371e6):
        """Construct a global grid; regional/pole-centered layouts are not supported."""
        return cls(nlat, nlon, radius=radius)

    def _check(self, value, shape):
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or tuple(value.shape[-2:]) != shape:
            raise ValueError(f"expected a tensor with trailing shape {shape}")
        if value.dtype != self.cell_area.dtype or value.device != self.cell_area.device:
            raise ValueError("field dtype/device must match grid; move grid explicitly with .to()")

    def integral(self, density: torch.Tensor) -> torch.Tensor:
        """Area integral over the last two axes; input is a quantity per m²."""
        self._check(density, self.shape)
        return (density*self.cell_area).sum(dim=(-2, -1))

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        """Area-weighted mean, retaining all leading axes."""
        return self.integral(value)/self.cell_area.sum()

    def grad(self, scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell scalar -> east/north face derivatives (scalar units per metre).

        Polar values are zero placeholders on zero-area faces, not a physical
        boundary condition on a finite-length boundary.
        """
        self._check(scalar, self.shape)
        east = (scalar.roll(-1, dims=-1)-scalar)/self.east_distance
        interior = (scalar[..., 1:, :]-scalar[..., :-1, :])/self.north_distance[1:-1]
        zero = torch.zeros_like(scalar[..., :1, :])
        return east, torch.cat((zero, interior, zero), dim=-2)

    def div(self, east: torch.Tensor, north: torch.Tensor) -> torch.Tensor:
        """Normal flux densities -> cell divergence (input units per metre).

        Supply flux per unit face length, not an already integrated transport.
        Finite polar-face values contribute zero. Leading axes must agree.
        """
        self._check(east, self.shape)
        self._check(north, (self.nlat+1, self.nlon))
        if east.shape[:-2] != north.shape[:-2]:
            raise ValueError("east/north leading axes must agree")
        e, n = east*self.east_face_length, north*self.north_face_length
        return (e-e.roll(1, dims=-1)+n[..., 1:, :]-n[..., :-1, :])/self.cell_area

    def laplacian(self, scalar: torch.Tensor) -> torch.Tensor:
        """Matched scalar Laplacian div(grad(scalar)); negative semidefinite."""
        return self.div(*self.grad(scalar))

    def face_inner(self, east_a, north_a, east_b, north_b) -> torch.Tensor:
        """Metric-weighted face inner product paired with cell-area integration."""
        for x in (east_a, east_b):
            self._check(x, self.shape)
        for x in (north_a, north_b):
            self._check(x, (self.nlat+1, self.nlon))
        if len({x.shape[:-2] for x in (east_a, north_a, east_b, north_b)}) != 1:
            raise ValueError("face fields' leading axes must agree")
        return ((east_a*east_b*self.east_weight).sum((-2, -1))
                +(north_a*north_b*self.north_weight).sum((-2, -1)))

    def wind_to_faces(self, eastward, northward) -> tuple[torch.Tensor, torch.Tensor]:
        """Arithmetic interpolation of collocated local wind components.

        This is an explicit diagnostic adapter, not a conservative regridding
        operator or a kinetic-energy-preserving velocity reconstruction.
        """
        self._check(eastward, self.shape)
        self._check(northward, self.shape)
        if eastward.shape != northward.shape:
            raise ValueError("wind component shapes must agree")
        east = .5*(eastward+eastward.roll(-1, dims=-1))
        middle = .5*(northward[..., :-1, :]+northward[..., 1:, :])
        zero = torch.zeros_like(northward[..., :1, :])
        return east, torch.cat((zero, middle, zero), dim=-2)

    def curl(self, east: torch.Tensor, north: torch.Tensor) -> torch.Tensor:
        """Outward curl from circulation around dual cells, shape (..., nlat+1, nlon).

        Interior (i,j) is at latitude boundary i and longitude boundary j+1.
        Both polar rows repeat a single cap-average value. corner_area assigns
        each repeated entry 1/nlon of the cap area, avoiding double counting.
        Input units per metre; stress in N/m² gives curl in N/m³.
        """
        self._check(east, self.shape)
        self._check(north, (self.nlat+1, self.nlon))
        if east.shape[:-2] != north.shape[:-2]:
            raise ValueError("east/north leading axes must agree")
        # Tangential line integrals on the dual boundary. For grad(q), each
        # is a difference of endpoint values, so closed loops telescope.
        e = east*self.east_distance
        n = north[..., 1:-1, :]*self.north_distance[1:-1]
        interior = (n.roll(-1, dims=-1)-n+e[..., :-1, :]-e[..., 1:, :]) / self.corner_area[1:-1]
        south = -e[..., :1, :].sum(-1, keepdim=True)/self.corner_area[0].sum()
        north_cap = e[..., -1:, :].sum(-1, keepdim=True)/self.corner_area[-1].sum()
        return torch.cat((south.expand_as(e[..., :1, :]), interior,
                          north_cap.expand_as(e[..., :1, :])), dim=-2)

    def corner_integral(self, scalar: torch.Tensor) -> torch.Tensor:
        """Dual-area integral; repeated polar values carry fractional cap areas."""
        self._check(scalar, (self.nlat+1, self.nlon))
        return (scalar*self.corner_area).sum((-2, -1))

    def corners_to_cells(self, scalar: torch.Tensor) -> torch.Tensor:
        """Area-overlap averaging of a piecewise-constant dual-cell field.

        Conserves the area integral. Polar rows are first averaged across
        longitude because each represents one cap. This smoothing does not
        recover cell-center point values exactly.
        """
        self._check(scalar, (self.nlat+1, self.nlon))
        south = scalar[..., :1, :].mean(-1, keepdim=True).expand_as(scalar[..., :1, :])
        north = scalar[..., -1:, :].mean(-1, keepdim=True).expand_as(scalar[..., -1:, :])
        caps = torch.cat((south, scalar[..., 1:-1, :], north), dim=-2)
        zonal = .5*(caps+caps.roll(1, dims=-1))
        return (self.lower_corner_fraction*zonal[..., :-1, :]
                +(1-self.lower_corner_fraction)*zonal[..., 1:, :])

    def curl_from_collocated(self, eastward: torch.Tensor, northward: torch.Tensor,
                             *, location: str = "cell") -> torch.Tensor:
        """Curl of collocated wind/stress, with explicit interpolation and averaging.

        location='corner' retains the native circulation result; 'cell' applies
        conservative dual-to-cell averaging. No density or Coriolis division is
        applied: wind-stress curl is not itself Ekman pumping.
        """
        if location not in ("cell", "corner"):
            raise ValueError("location must be 'cell' or 'corner'")
        result = self.curl(*self.wind_to_faces(eastward, northward))
        return self.corners_to_cells(result) if location == "cell" else result
