"""Atmospheric wave-anchor restriction; mesh weights are shared geometry."""
import torch


def restrict_state(anchor, hierarchy):
    """Coarse wave anchor; tracer layout is [..., cell, full, tracer].

    Restrict contents, pressure and terrain with the existing area-adjoint
    weights; zero coarse wind. Preserve batch axes and leave input untouched.
    """
    from ..state import State
    initial = anchor
    for transfer, mesh in zip(hierarchy.transfers, hierarchy.meshes[1:]):
        mtheta = transfer.restrict(anchor.mtheta_i)
        ps = transfer.restrict(anchor.ps)
        tracer = (
            None
            if anchor.tracer_m is None
            else transfer.restrict(
                anchor.tracer_m,
                horizontal_axis=-3,  # [..., cell, full, tracer]
            )
        )
        assert isinstance(mtheta, torch.Tensor) and isinstance(ps, torch.Tensor)
        assert tracer is None or isinstance(tracer, torch.Tensor)
        column = anchor.column
        if getattr(column.phi_s, "ndim", 0) > 0:
            phi_s = transfer.restrict(column.phi_s)
            assert isinstance(phi_s, torch.Tensor)
            column = type(column)(
                column.n_levels,
                column.A_interface,
                column.B_interface,
                phi_s=phi_s,
                phi_full_mode=column.phi_full_mode,
                R_d=column.R_d,
                c_p=column.c_p,
                p0=column.p0,
                g=column.g,
            ).to(device=mtheta.device)
        u_shape = (*anchor.u_e.shape[:-2], mesh.nEdges, anchor.u_e.shape[-1])
        anchor = State(
            torch.zeros(u_shape, dtype=anchor.u_e.dtype, device=anchor.u_e.device),
            mtheta,
            ps,
            tracer,
            mesh=mesh,
            column=column,
            attrs=initial.attrs,
        )
    return anchor
