"""The endpoint diagnostic must observe failures without changing physics."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("pyam21")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.diagnose_pyam21_endpoint import ERROR, capture_failure, endpoint_report
from workflows.pyam21.adapter import AM2Adapter
from pyam21.physics.moist import MoistProcesses
from test_pyam21_workflow import initial, model, same_tree


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_report_localizes_temperature_and_cloud_errors(dtype):
    T = torch.full((3, 4), 280.0, dtype=dtype)
    R = torch.zeros((3, 4, 4), dtype=dtype)
    R[..., 0] = 0.01
    R[..., 3] = 0.1
    out = dict(tdt=torch.zeros_like(T), rdt=torch.zeros_like(R),
               endpoint_temperature=T.clone(), endpoint_tracers=R.clone())
    out["endpoint_temperature"][2, 1] += 0.1
    out["endpoint_tracers"][1, 3, 3] += 0.01
    before = {k: v.clone() for k, v in out.items()}
    result = endpoint_report(dict(T=T, tracers=R), out, 1200)
    assert result["failing_columns"] == 2
    assert result["fields"]["temperature"]["cell"] == 2
    assert result["fields"]["temperature"]["level"] == 1
    assert result["fields"]["cloud_fraction"]["cell"] == 1
    assert result["fields"]["cloud_fraction"]["violations"] == 1
    assert result["fields"]["vapor"]["violations"] == 0
    assert result["fields"]["liquid"]["tolerance"] == pytest.approx(
        128 * torch.finfo(dtype).eps * float(R[..., :3].abs().sum(-1).max()))
    for key in out:
        assert torch.equal(out[key], before[key])


def test_capture_preserves_successful_real_physics(model, tmp_path):
    adapter, state = initial(model, cloudy=True)
    expected, expected_report = adapter.apply(state, 60)
    with capture_failure(AM2Adapter, MoistProcesses, tmp_path):
        actual, report = adapter.apply(state, 60)
    same_tree(adapter.pack(expected), adapter.pack(actual))
    same_tree(expected_report.__dict__, report.__dict__)
    assert not list(tmp_path.iterdir())


def test_capture_keeps_actual_guard_and_saves_replayable_moist_input(model, tmp_path):
    adapter, state = initial(model, cloudy=True)
    before = adapter.pack(state)
    up = adapter.physics.up

    def inconsistent(*args, **kwargs):
        output = up(*args, **kwargs)
        output["moist"]["qin"] = output["moist"]["qin"].clone()
        output["moist"]["qin"][3, 2] += 0.001
        return output

    with patch.object(adapter.physics, "up", inconsistent):
        with capture_failure(AM2Adapter, MoistProcesses, tmp_path):
            with pytest.raises(ValueError, match=ERROR):
                adapter.apply(state, 60)
    same_tree(before, adapter.pack(state))
    report = json.loads((tmp_path / "endpoint_failure.json").read_text())
    assert report["status"] == "endpoint_mismatch_captured"
    assert report["fields"]["vapor"]["cell"] == 3
    assert report["fields"]["vapor"]["level"] == 2
    assert report["fields"]["vapor"]["violations"] == 1
    bundle = torch.load(tmp_path / "endpoint_failure.pt", weights_only=True)
    for arg in bundle["moist_call"]["args"]:
        if isinstance(arg, torch.Tensor):
            assert arg.device.type == "cpu"
            assert arg.shape[0] == len(report["selected_columns"])
    # Actual upstream physics can be replayed from this small capture without
    # a global mesh, optics tables, model initialization or restart bypass.
    physics = MoistProcesses.from_nml(bundle["namelist"])
    replay = physics(*bundle["moist_call"]["args"], **bundle["moist_call"]["kwargs"])
    for key in ("tin", "rdt", "tdt", "lsc_liquid", "lsc_ice", "lsc_cloud_area"):
        torch.testing.assert_close(replay[key], bundle["moist_call"]["result"][key], rtol=0, atol=0)
