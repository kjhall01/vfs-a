"""Regression for information-rich strict carrier CFL rejection.

This test does not change the physical timestep or relax the positivity guard.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vfs.ocean.transport import HorizontalTransport


def test_outflow_rejection_reports_limiting_cell_layer_and_dt():
    dtype = torch.float64
    mesh = SimpleNamespace(cell_area=torch.ones(2, dtype=dtype))
    transport = HorizontalTransport(mesh)
    state = SimpleNamespace(
        h=torch.ones((2, 1), dtype=dtype),
        u_e=torch.tensor([[2.0]], dtype=dtype),
    )
    graph = (
        torch.tensor([1.0, 1.0], dtype=dtype),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2.0], dtype=dtype),
    )
    with patch.object(transport, "graph", return_value=graph):
        with pytest.raises(FloatingPointError) as caught:
            transport.advance(state, 1.0)
    msg = str(caught.value)
    assert "actual carrier outflow exceeds Euler budget" in msg
    assert "cell=0 layer=0" in msg
    assert "dt=1s" in msg
    assert "max_outflow_ratio=" in msg
    assert "local_dt_limit=" in msg
    # The permitted donor outflow is 0.9 * (1 - 0.001) = 0.8991 m3.
    # For an outflow rate of 2 m3/s, the local stage limit is 0.44955 s.
    assert "local_dt_limit=0.44955s" in msg
