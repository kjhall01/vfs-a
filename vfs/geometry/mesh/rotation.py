"""Compatible horizontal PV-flux rotation, shared by atmospheric/ocean layers.

For positive carrier thickness h, F=I(h)u and q=(C u+f)/P(h).
Acceleration is -Q(q)F, where Q uses symmetric edge-PV averaging and is skew
in the edge-area metric. Thus <I(h)u, acceleration>_edge_area=0. Constant q
also gives C acceleration = q P(-D F), provided CJ=PD. This is horizontal
uniform-PV compatibility, not a general enstrophy or Jacobi guarantee.
"""
import torch


def initialize_dual_transfer(mesh):
    """Cache geometry-only indices/weights once, also after mesh-file loading."""
    if mesh.rotation_scheme != 'compatible':
        return
    valid = mesh.verticesOnCell >= 0
    cells = torch.arange(mesh.nCells, device=mesh.cell_area.device)[:,None].expand_as(valid)[valid]
    vertices = mesh.verticesOnCell[valid]
    weights = mesh.thuburn_corner_area[valid] * mesh.radius**2 / mesh.dual_area[vertices]
    mesh.register_buffer('dual_transfer_cells', cells)
    mesh.register_buffer('dual_transfer_vertices', vertices)
    mesh.register_buffer('dual_transfer_weights', weights)


def cell_to_dual(mesh, value):
    if mesh.rotation_scheme != 'compatible':
        raise ValueError('dual mass transfer requires rotation_scheme="compatible"')
    if value.ndim != 2 or value.shape[0] != mesh.nCells:
        raise ValueError('dual transfer expects [cell, layer] tensors')
    if value.dtype != mesh.cell_area.dtype or value.device != mesh.cell_area.device:
        raise ValueError('dual transfer dtype/device must match mesh')
    return value.new_zeros((mesh.nVertices,value.shape[1])).index_add(
        0, mesh.dual_transfer_vertices,
        mesh.dual_transfer_weights[:,None]*value[mesh.dual_transfer_cells])


def compatible_rotation(mesh, u, thickness, rotation_rate):
    dual_mass = cell_to_dual(mesh, thickness)
    if not torch.isfinite(dual_mass).all() or (dual_mass <= 0).any():
        raise ValueError('compatible rotation requires finite positive dual carrier thickness')
    if u.shape != (mesh.nEdges,thickness.shape[1]):
        raise ValueError('velocity must have shape [edge, layer] matching thickness')
    if u.dtype != thickness.dtype or u.device != thickness.device:
        raise ValueError('velocity and thickness dtype/device must match')
    absolute = mesh.curl(u) + float(rotation_rate)*mesh.coriolis_unit_vertex[:,None]
    pv = absolute/dual_mass
    v0,v1 = mesh.verticesOnEdge.T
    qe = .5*(pv[v0]+pv[v1])
    stencil = mesh.safe_rotation_stencil
    flux = mesh.cell_to_edge(thickness)*u
    coefficients = -.5*(qe[:,None,:]+qe[stencil])*mesh.rotation_weights_masked[:,:,None]
    return (coefficients*flux[stencil]).sum(1)
