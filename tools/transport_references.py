"""Independent spherical C5 specifications and cell-average references.

Williamson et al. (1992) solid rotation; Nair & Lauritzen (2010), Case 4.
See verification/INCREMENT_1C.md for equations, units and adaptations.
These analytical definitions are deliberately outside the numerical package.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

PERIOD = 12 * 86400.
FLOWS = ('rotation_equatorial', 'rotation_tilted', 'NL2010_case4')


def axis(flow):
    alpha = 0. if flow == 'rotation_equatorial' else math.pi / 4
    return np.array([-math.sin(alpha), 0., math.cos(alpha)])


def velocity(xyz, time, flow, radius, period=PERIOD):
    """Cartesian tangential velocity [point,3], m/s, on unit xyz.

    Longitude increases east from +x toward +y, latitude north toward +z.
    Case 4 uses t*=5t/T, lambda'=lambda-2*pi*t/T and k=2. No projection
    or discrete streamfunction correction is applied to this analytical field.
    """
    if flow not in FLOWS:
        raise ValueError(f'Unknown prescribed flow: {flow}')
    xyz = np.asarray(xyz)
    if flow.startswith('rotation'):
        return 2 * np.pi * radius / period * np.cross(axis(flow), xyz)
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    lat = np.arctan2(xyz[..., 2], np.hypot(xyz[..., 0], xyz[..., 1]))
    shifted = lon - 2 * np.pi * time / period
    amplitude = np.cos(np.pi * time / period)
    u = (2 * np.sin(shifted)**2 * np.sin(2 * lat) * amplitude
         + 2 * np.pi / 5 * np.cos(lat)) * 5 * radius / period
    v = 2 * np.sin(2 * shifted) * np.cos(lat) * amplitude * 5 * radius / period
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), -1)
    north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), -1)
    return u[..., None] * east + v[..., None] * north


def departure(xyz, time, flow, period=PERIOD, rtol=2e-11):
    """Inverse characteristic on the unit sphere, independently of VFS.

    Exact Rodrigues inverse for rotation; exact identity after Case 4's full
    period. At intermediate times use DOP853 in dimensionless time, integrating
    reference trajectories backward (not negative production Euler budgets).
    """
    if flow not in FLOWS:
        raise ValueError(f'Unknown prescribed flow: {flow}')
    xyz = np.asarray(xyz)
    if time == 0 or time == period:
        return xyz.copy()
    if flow.startswith('rotation'):
        a, angle = axis(flow), -2 * np.pi * time / period
        return (xyz*np.cos(angle) + np.cross(a, xyz)*np.sin(angle)
                + (xyz @ a)[..., None]*a*(1-np.cos(angle)))
    def rhs(t, flat):
        points = flat.reshape(xyz.shape)
        return velocity(points, t, flow, 1., 1.).ravel()
    solution = solve_ivp(rhs, (time / period, 0.), xyz.ravel(), method='DOP853',
                         rtol=rtol, atol=rtol / 100)
    if not solution.success:
        raise RuntimeError(solution.message)
    points = solution.y[:, -1].reshape(xyz.shape)
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


def profile(xyz, kind='smooth'):
    """Admissible water fractions [point,3], kg/kg, not pressure contents.

    Gaussian hills from NL2010 (b=5, hmax=1); positive vapor is prescribed
    as part of the initial condition, not a runtime floor. Compact condensate
    has exact zero support outside angular radius .5 radians.
    """
    xyz = np.asarray(xyz)
    if kind == 'constant':
        return np.broadcast_to([.01, .001, .0001], (*xyz.shape[:-1], 3)).copy()
    centers = np.array([[np.cos(lon), np.sin(lon), 0.] for lon in (5*np.pi/6, 7*np.pi/6)])
    distance2 = ((xyz[..., None, :] - centers)**2).sum(-1)
    hills = np.exp(-5 * distance2).sum(-1)
    if kind == 'smooth':
        cloud = hills
    elif kind == 'empty_condensate':
        angle = np.arctan2(np.linalg.norm(np.cross(xyz[..., None, :], centers), axis=-1), xyz @ centers.T)
        cloud = np.where(angle < .5, .5*(1 + np.cos(np.pi*angle/.5)), 0.).sum(-1)
    else:
        raise ValueError(f'Unknown profile: {kind}')
    return np.stack((.01 + .001*hills, .001*cloud, .0001*cloud), -1)


def quadrature(mesh, order):
    """Independent Gauss-Duffy integration over spherical Voronoi cells.

    Triangulate each polygon with its generator and adjacent dual vertices.
    Normalize planar barycentric points onto the sphere; Jacobian is
    |det(a,b,c)|/|a+u(b-a)+v(c-a)|^3. Duffy v=(1-u)*eta contributes (1-u).
    Weights are solid angles. Divide by independently exact spherical triangle
    areas, not by a quadrature sum that could conceal area integration error.
    """
    nodes, weights = leggauss(order)
    u, eta = np.meshgrid((nodes+1)/2, (nodes+1)/2, indexing='ij')
    w = np.outer(weights/2, weights/2)*(1-u)
    u, v, w = u.ravel(), ((1-u)*eta).ravel(), w.ravel()
    points, areas, owners, exact = [], [], [], []
    cells, vertices = mesh.cell_xyz.numpy(), mesh.vertex_xyz.numpy()
    for cell, a in enumerate(cells):
        ring = mesh.verticesOnCell[cell, :int(mesh.nEdgesOnCell[cell])].numpy()
        total = 0.
        for b, c in zip(vertices[ring], np.roll(vertices[ring], -1, axis=0)):
            determinant = abs(np.dot(a, np.cross(b, c)))
            total += 2*np.arctan2(determinant, 1+np.dot(a,b)+np.dot(b,c)+np.dot(c,a))
            raw = a + u[:, None]*(b-a) + v[:, None]*(c-a)
            norm = np.linalg.norm(raw, axis=-1)
            points.append(raw/norm[:, None])
            areas.append(w*determinant/norm**3)
            owners.append(np.full(w.size, cell))
        exact.append(total)
    return np.concatenate(points), np.concatenate(areas), np.concatenate(owners), np.array(exact)


def cell_average(mesh, flow, time, kind, order, rtol=2e-11):
    points, weights, owner, area = quadrature(mesh, order)
    values = profile(departure(points, time, flow, rtol=rtol), kind)
    average = np.stack([np.bincount(owner, weights*values[:, k], minlength=mesh.nCells)/area
                        for k in range(3)], -1)
    area_error = np.max(abs(np.bincount(owner, weights, minlength=mesh.nCells)/area-1))
    return average, dict(order=order, maximum_relative_area_quadrature_error=float(area_error),
                        production_area_relative_difference=float(np.max(abs(area*mesh.radius**2/mesh.cell_area.numpy()-1))))
