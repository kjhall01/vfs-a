from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("pyam21")
from test_coupled_campaign import config, source, synthetic
from test_pyam21_workflow import same_tree
from workflows.coupled import campaign as c
from workflows.coupled.shelf_redi import CoupledShelfRedi


@torch.no_grad()
def test_projected_shelf_redi_coupled_and_restart(config, source):
    cfg = replace(config, shelf_redi=True, ocean_levels=6)
    land = np.arange(cfg.cells) == 0
    depths = np.linspace(8.0, cfg.ocean_depth_m, cfg.cells)
    geography = dict(land=land, depth_m=depths, height_m=1200.0 * land)
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(
            cfg, source, Path("synthetic"), geography=geography
        )
    model = receiver.H.model
    state = sea.coupled.ocean
    op = CoupledShelfRedi(model, state, c.rest_thickness(cfg), c.gm_redi_operator(cfg))
    torch.manual_seed(212)
    x = torch.randn_like(state.ocean.h)[..., None]
    g = op.apply(x)
    y = torch.randn_like(g)
    torch.testing.assert_close(
        (g * y).sum(), (x * op.transpose(y)).sum(), atol=1e-14, rtol=1e-12
    )
    assert (g.square() * op.rows.measure[:, None]).sum() > 0
    assert op.transpose(y)[land].count_nonzero() == 0
    torch.testing.assert_close(
        op.apply(torch.ones_like(x)), torch.zeros_like(g), rtol=0, atol=1e-13
    )
    # Force a positive-field withdrawal that exceeds the native inventory.
    transfer = 1.0e10 * y
    amounts = torch.ones_like(x)
    scale = op.positive_flux_scale(transfer, amounts)
    assert (scale < 1.0).any()
    updated = amounts + op.transpose(scale * transfer)
    assert updated.min() >= 0.0
    torch.testing.assert_close(updated.sum(), amounts.sum(), rtol=2.0e-15, atol=1.0e-13)
    saved = c.pack(adapter, air, receiver, sea, "shelf", {})
    with patch.object(
        CoupledShelfRedi, "apply", autospec=True, side_effect=CoupledShelfRedi.apply
    ) as ran:
        a, s, report = c.advance(adapter, air, receiver, sea, cfg.physics_dt, cfg)
        assert ran.call_count == report["gmredi_steps"] > 0
    ra, rs, _ = c.unpack(saved, adapter, receiver, sea, "shelf")
    b, t, replay = c.advance(adapter, ra, receiver, rs, cfg.physics_dt, cfg)
    same_tree(
        c.pack(adapter, a, receiver, s, "shelf", {}),
        c.pack(adapter, b, receiver, t, "shelf", {}),
    )
    assert report == replay and s.land is not None


def test_mesh_order_check(tmp_path):
    path = tmp_path / "geography.npz"
    np.savez(
        path,
        land=[True, False],
        height_m=[100.0, 0.0],
        depth_m=[0.0, 80.0],
        metadata='{"mesh_sha256":"expected"}',
    )
    with pytest.raises(ValueError, match="mesh_sha256"):
        c.load_geography(path, mesh_sha256="different")
    assert c.load_geography(path, mesh_sha256="expected")["height_m"][0] == 100.0
