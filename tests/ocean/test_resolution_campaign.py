"""Campaign geometry and evidence comparisons must retain their physical meaning."""
import sys
import torch
from test_linear_column import load
from test_lagged_column import lag
sys.modules.setdefault('lagged', lag)
from test_consistent_column import consistent
sys.modules.setdefault('consistent', consistent)
mesh_campaign = load('resolution_mesh')

def test_vertical_refinement_preserves_column_geometry():
    coarse, a, _ = mesh_campaign.setup(12, 32)
    fine, b, _ = mesh_campaign.setup(12, 64)
    assert bool((b.h > 0).all())
    torch.testing.assert_close(a.h, b.h.reshape(12, 32, 2).sum(-1), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(a.h.sum(-1), torch.full((12,), 100., dtype=a.h.dtype))
    torch.testing.assert_close(coarse.mesh.cell_area, fine.mesh.cell_area)
