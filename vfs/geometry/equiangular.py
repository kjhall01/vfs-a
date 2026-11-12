"""Conservative C-grid diagnostics on a global spherical latitude–longitude grid."""
from __future__ import annotations

import math
import numpy as np
import torch

from .remap import LatLonGrid


def _fixed_tensor(value):
    """Own geometry storage, including read-only/reversed NumPy coordinates."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu',dtype=torch.float64).clone()
    return torch.from_numpy(np.array(value,dtype=np.float64,copy=True))


class EquiangularGrid(LatLonGrid):
    """Global regular latitude–longitude diagnostics with matched FV operators.

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
        self._initialize((lat_b[:-1]+lat_b[1:])/2,
                         (lon_b[:-1]+lon_b[1:])/2, radius, lat_b, lon_b)

    @classmethod
    def regular(cls, nlat: int, nlon: int, *, radius: float = 6.371e6):
        """Construct a global grid whose samples exclude the poles."""
        return cls(nlat, nlon, radius=radius)

    @classmethod
    def from_coordinates(cls, latitude, longitude, *, radius: float = 6.371e6,
                         lat_bounds=None, lon_bounds=None):
        """Global regular coordinates in degrees, retaining input array order.

        Latitude may increase or decrease, with either both poles sampled or
        half-step centers excluding them. Longitude must increase eastward
        modulo 360, with no repeated endpoint; offsets and wrapped seams work.
        Optional bounds are n+1 edges or contiguous (n,2) pairs in axis order.
        Only midpoint control volumes (clipped at the poles) are supported.
        Coordinates/bounds are fixed geometry, detached and copied to CPU.
        """
        def coordinate(x, name, minimum):
            x = _fixed_tensor(x)
            if x.ndim != 1 or len(x) < minimum or not torch.isfinite(x).all():
                raise ValueError(f'{name} must be a finite 1D coordinate with >= {minimum} points')
            return x
        lat = coordinate(latitude, 'latitude', 2)
        lon = coordinate(longitude, 'longitude', 3)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be finite and positive')
        # Validate the complete periodic ring, including its closing interval.
        dl = torch.remainder(lon.roll(-1)-lon, 360.)
        step_lon = 360./len(lon)
        if not torch.allclose(dl, torch.full_like(dl,step_lon), atol=1e-6, rtol=0):
            raise ValueError('longitude must be a complete regular eastward ring without a duplicate seam')
        dlat = torch.diff(lat)
        direction = 1 if dlat[0] > 0 else -1
        centered = torch.linspace(-90.+90./len(lat),90.-90./len(lat),len(lat),dtype=lat.dtype)
        poles = torch.linspace(-90.,90.,len(lat),dtype=lat.dtype)
        ordered = lat if direction == 1 else lat.flip(0)
        if not (torch.allclose(ordered,centered,atol=1e-6,rtol=0)
                or (len(lat)>=3 and torch.allclose(ordered,poles,atol=1e-6,rtol=0))):
            raise ValueError('latitude must cover the globe regularly, at half-step centers or both poles')
        # Snap pole endpoints only within the validated coordinate tolerance.
        if torch.allclose(ordered,poles,atol=1e-6,rtol=0):
            lat[0],lat[-1] = -90.*direction,90.*direction
        lat_b = torch.cat((lat.new_tensor([-90.*direction]),.5*(lat[:-1]+lat[1:]),lat.new_tensor([90.*direction])))
        lon_b = lon[0]-step_lon/2 + torch.arange(len(lon)+1,dtype=lon.dtype)*step_lon
        def check_bounds(value, expected, name, periodic=False):
            if value is None:
                return
            b = _fixed_tensor(value)
            target = torch.stack((expected[:-1],expected[1:]),-1) if b.ndim==2 else expected
            if b.shape != target.shape or not torch.isfinite(b).all():
                raise ValueError(f'{name} must have shape (n+1,) or (n,2)')
            delta = b-target
            if periodic:
                delta = torch.remainder(delta+180.,360.)-180.
            if not torch.allclose(delta,torch.zeros_like(delta),atol=1e-6,rtol=0):
                raise ValueError(f'{name} must describe contiguous midpoint control volumes')
        check_bounds(lat_bounds,lat_b,'lat_bounds')
        check_bounds(lon_bounds,lon_b,'lon_bounds',True)
        grid = cls.__new__(cls)
        grid._initialize(lat,lon,radius,lat_b,lon_b)
        return grid

    def _initialize(self, lat, lon, radius, lat_b, lon_b):
        # LatLonGrid's original constructor assumes an unwrapped increasing
        # longitude axis. Retain its interface but build the validated geometry
        # here, including descending latitude and a wrapped longitude seam.
        torch.nn.Module.__init__(self)
        self.nlat, self.nlon, self.radius = len(lat), len(lon), float(radius)
        self.register_buffer('lat_deg',lat)
        self.register_buffer('lon_deg',lon)
        self.latitude_direction = 1 if lat[-1] > lat[0] else -1
        self.pole_centered = bool(lat[0].abs() == 90. and lat[-1].abs() == 90.)
        dlam = 2*math.pi/self.nlon
        dphi = torch.deg2rad(torch.diff(lat_b).abs())
        area = (radius**2*dlam*torch.diff(lat_b.deg2rad().sin()).abs())[:,None].expand(self.shape).clone()
        self.register_buffer('cell_area',area)
        east_length = (radius*dphi)[:,None].expand(self.shape).clone()
        north_length = (radius*dlam*torch.cos(lat_b.deg2rad()))[:,None].expand(self.nlat+1,self.nlon).clone()
        north_length[[0,-1]] = 0.
        east_distance = (radius*dlam*torch.cos(lat.deg2rad()))[:,None].expand(self.shape).clone()
        distances = torch.cat((dphi[:1],torch.diff(lat.deg2rad()).abs(),dphi[-1:]))
        north_distance = (radius*distances)[:,None].expand(self.nlat+1,self.nlon).clone()
        center_sin = lat.deg2rad().sin()
        dual_bounds = torch.cat((center_sin.new_tensor([-self.latitude_direction]),center_sin,
                                 center_sin.new_tensor([self.latitude_direction])))
        corner_area = (radius**2*dlam*torch.diff(dual_bounds).abs())[:,None].expand(self.nlat+1,self.nlon).clone()
        fraction = ((center_sin-lat_b[:-1].deg2rad().sin())
                    / torch.diff(lat_b.deg2rad().sin()))[:,None]
        if self.pole_centered:
            # Each polar scalar row represents ONE cap. Internal meridional
            # sector faces are removed, not divided by cos(90 degrees).
            east_length[[0,-1]] = 0.
            east_distance[[0,-1]] = 0.
        for name,value in dict(corner_area=corner_area,lower_corner_fraction=fraction,
                               lat_bounds_deg=lat_b,lon_bounds_deg=lon_b,
                               east_face_length=east_length,north_face_length=north_length,
                               east_distance=east_distance,north_distance=north_distance,
                               east_weight=east_length*east_distance,
                               north_weight=north_length*north_distance).items():
            self.register_buffer(name,value)

    def _polar_mean(self, value):
        """One scalar degree of freedom per pole; preserve its area integral."""
        if not self.pole_centered:
            return value
        first = value[..., :1, :].mean(-1,keepdim=True).expand_as(value[..., :1, :])
        last = value[..., -1:, :].mean(-1,keepdim=True).expand_as(value[..., -1:, :])
        return torch.cat((first,value[...,1:-1,:],last),dim=-2)

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

        Endpoint face values are zero placeholders on zero-area faces, not a
        physical boundary condition on a finite-length boundary. Pole-sampled
        scalar rows are averaged across longitude before differentiation.
        """
        self._check(scalar, self.shape)
        scalar = self._polar_mean(scalar)
        distance = torch.where(self.east_distance > 0,self.east_distance,1.)
        east = (scalar.roll(-1, dims=-1)-scalar)/distance
        interior = self.latitude_direction*(scalar[..., 1:, :]-scalar[..., :-1, :])/self.north_distance[1:-1]
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
        result = (e-e.roll(1, dims=-1)+self.latitude_direction*(n[..., 1:, :]-n[..., :-1, :]))/self.cell_area
        return self._polar_mean(result)

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

        Polar northward components retain their longitude-dependent basis;
        they must not be averaged like scalars. Collapsed east faces are zero.
        This is an explicit diagnostic adapter, not a conservative regridding
        operator or a kinetic-energy-preserving velocity reconstruction.
        """
        self._check(eastward, self.shape)
        self._check(northward, self.shape)
        if eastward.shape != northward.shape:
            raise ValueError("wind component shapes must agree")
        east = .5*(eastward+eastward.roll(-1, dims=-1))
        if self.pole_centered:
            zero = torch.zeros_like(east[..., :1, :])
            east = torch.cat((zero,east[...,1:-1,:],zero),dim=-2)
        middle = .5*(northward[..., :-1, :]+northward[..., 1:, :])
        zero = torch.zeros_like(northward[..., :1, :])
        return east, torch.cat((zero, middle, zero), dim=-2)

    def curl(self, east: torch.Tensor, north: torch.Tensor) -> torch.Tensor:
        """Outward curl from circulation around dual cells, shape (..., nlat+1, nlon).

        Interior (i,j) is at latitude boundary i and longitude boundary j+1.
        For a centered layout, endpoint rows repeat cap-average values with
        fractional cap areas. For a pole-sampled layout these endpoint dual
        cells have zero area and return zero placeholders; use the default
        cell output of curl_from_collocated for meaningful polar cap values.
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
        interior = (n.roll(-1, dims=-1)-n+self.latitude_direction*(e[..., :-1, :]-e[..., 1:, :])) / self.corner_area[1:-1]
        if self.pole_centered:
            # The endpoint dual cells have collapsed to zero area; interior
            # dual cells adjacent to the pole are triangular sectors.
            south = north_cap = torch.zeros_like(e[..., :1, :1])
        else:
            south = -self.latitude_direction*e[..., :1, :].sum(-1, keepdim=True)/self.corner_area[0].sum()
            north_cap = self.latitude_direction*e[..., -1:, :].sum(-1, keepdim=True)/self.corner_area[-1].sum()
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
        return self._polar_mean(self.lower_corner_fraction*zonal[..., :-1, :]
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

    def div_from_collocated(self, eastward: torch.Tensor, northward: torch.Tensor) -> torch.Tensor:
        """Divergence on the supplied scalar layout after component interpolation."""
        return self.div(*self.wind_to_faces(eastward,northward))
