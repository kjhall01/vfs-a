"""Non-invasive nested RHS instrumentation through the real coupled step."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

pytest.importorskip("pyam21")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflows.coupled import campaign as c
from tools.profile_coupled import SectionClock
from tools.profile_ocean_rhs import RHSInstrument
from test_coupled_campaign import source, config, synthetic


def test_ocean_rhs_instrument_preserves_coupled_state_and_covers_real_rhs(
    source, config, tmp_path
):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    device = adapter.model.mesh.cell_area.device
    with torch.no_grad():
        a1, s1, r1 = c.advance(adapter, air, receiver, sea,
                               config.physics_dt, config)
        clock = SectionClock(device)
        with RHSInstrument(clock, adapter, receiver):
            a2, s2, r2 = c.advance(adapter, air, receiver, sea,
                                   config.physics_dt, config)
    for field in ("u_e", "ps", "entropy_content", "water_content"):
        assert torch.equal(
            getattr(a1.atmosphere, field), getattr(a2.atmosphere, field)
        ), field
    for field in ("h", "u_e", "temp_h", "salt_h", "tracer_h"):
        assert torch.equal(
            getattr(s1.coupled.ocean.ocean, field),
            getattr(s2.coupled.ocean.ocean, field)
        ), field
    assert torch.equal(s1.coupled.ice.area, s2.coupled.ice.area)
    assert r1 == r2
    counts = {k: v["inclusive"]["calls"] for k, v in clock.results().items()}
    for region in (
        "ocean_ice_rhs", "rhs_coupled_ocean_ice",
        "rhs_ocean_temperature_tendency", "rhs_temperature_entropy_fields",
        "rhs_conservative_transport_total", "rhs_parent_gibbs_transport",
        "rhs_ice_transport", "rhs_ice_ridging",
    ):
        assert counts.get(region, 0) > 0, (region, counts)
    assert counts["rhs_coupled_ocean_ice"] == counts["ocean_ice_rhs"]
    assert counts["rhs_ocean_temperature_tendency"] == counts["ocean_ice_rhs"]
