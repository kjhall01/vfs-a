"""Initialization and reference checks; comparisons impose no accuracy gate."""
import sys
from pathlib import Path
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'),str(Path(__file__).resolve().parents[2]/'validation/ocean_column')]
from matched_eos import build,anomaly

@pytest.mark.parametrize('kind,layers',[('wave',2),('jet',4),('mixing',16)])
def test_matching_preserves_geometry_salt_velocity_and_buoyancy(kind,layers):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    j,l,a,b,h0,meta,f=build(kind,16,layers,False)
    for key in ('h','u_e','salt_h','tracer_h'):
        torch.testing.assert_close(getattr(a.ocean,key),getattr(b.ocean,key),rtol=0,atol=0)
    assert (anomaly(j,a.ocean)-anomaly(l,b.ocean)).abs().max()<3e-13
    assert (a.ocean.h>0).all() and (b.ocean.h>0).all()
    if kind=='wave':
        assert 0<meta['omega'] and meta['fast_speed']>0
        assert meta['vector'][0]*meta['vector'][1]<0
    # Neither prognostic state aliases the other temperature content.
    assert a.ocean.temp_h.data_ptr()!=b.ocean.temp_h.data_ptr()


def test_mld_uses_first_crossing_and_interpolation():
    from analyze_matched import profile_mld
    assert profile_mld([0.,.02,.04,.1],depth=40.)==20.
    assert profile_mld([0.,.02,.01,.02],depth=40.)==40.
    assert profile_mld([0.,.06,.01,.1],depth=40.)==10.
