"""Opt-in regional compilation for the coupled CPU/CUDA float64 campaign.

Compile repeated tensor algebra, with the same functions and parameters. The
shared mesh kernels serve atmosphere, ocean and ice. Atmospheric Hamiltonian
gradients and Roquet calorics/energy serve both dynamics and coupled physics.
State validation, positivity limiting, nonlinear convergence, PyAM21 control
flow/radiation scheduling and file I/O remain eager. No global monkey-patching.
"""
import torch


MESH_KERNELS = ('_grad_values', '_div_values', '_cell_to_edge_values', '_edge_to_cell_values')


def configure_compilation(adapter, receiver, *, enabled=False, backend='inductor'):
    """Configure one model pair before stepping; compilation itself is lazy.

    ``backend`` is an explicit Python verification hook. The campaign switch
    always selects Inductor. Full-graph capture rejects unsupported regions;
    there is no fallback that silently runs the requested kernels eagerly.
    """
    manifest = dict(
        policy='coupled-regions-v1', enabled=enabled,
        backend=backend if enabled else None,
        fullgraph=True if enabled else None,
        dynamic=True if enabled else None,
        targets=(
            [f'mesh.{name}' for name in MESH_KERNELS]
            + ['atmosphere.mixture_gradients', 'ocean.stratification',
               'ocean.parcel_potential', 'ocean.coefficients', 'ocean.caloric_values']
            if enabled else []
        ),
    )
    previous = getattr(adapter, 'compilation_manifest', None)
    if previous is not None:
        if previous == manifest:
            return manifest
        if previous['enabled']:
            raise ValueError('rebuild the coupled models to change compilation mode/backend')
    if enabled:
        options = dict(backend=backend, fullgraph=True, dynamic=True)
        mesh = adapter.model.mesh
        for name in MESH_KERNELS:
            setattr(mesh, name, torch.compile(getattr(mesh, name), **options))
        kernel = adapter.model.dynamics.kernel
        kernel.mixture_gradients = torch.compile(kernel.mixture_gradients, **options)
        receiver.H.model.compile_thermodynamics(backend=backend)
    adapter.compilation_manifest = manifest
    return manifest
