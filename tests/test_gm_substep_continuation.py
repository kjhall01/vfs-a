"""Selection and source provenance for the day-15 GM/Redi branch."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gm_parcel_limiter as gm
from resume_gm_ice import find_day15_restart


def test_discover_only_unique_ten_step_bounded_ice_parent(tmp_path):
    def write_run(name, manifest):
        run = tmp_path / "vfs-a" / name
        (run / "restarts").mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps(manifest))
        path = run / "restarts/restart_01296000.pt"
        path.touch()
        return path

    sources = {**gm.BOUNDED_ICE_SOURCES, gm.EXPERIMENT_SOURCE: gm.ICE_FIX_LIMITER_SHA256}
    parent = {"configuration": {"gm_redi": True, "eddy_substeps": 10}, "source_sha256": sources}
    write_run("malformed", [])
    write_run("bad-source", {**parent, "source_sha256": None})
    write_run("different-ice", {**parent, "source_sha256": {}})
    write_run("two-substeps", {**parent, "configuration": {"gm_redi": True, "eddy_substeps": 2}})
    with pytest.raises(ValueError, match="found none"):
        find_day15_restart(tmp_path)
    original = write_run("baseline", parent)
    assert find_day15_restart(tmp_path) == original
    write_run("other-baseline", parent)
    with pytest.raises(ValueError, match="Specify --resume"):
        find_day15_restart(tmp_path)


def test_substep_bridge_checks_complete_parent_and_child_source_maps():
    pytest.importorskip("pyam21")
    from run_gm_day9 import load_campaign

    current = gm.experiment_sources(load_campaign().source_hashes())
    parent = {**current, gm.CAMPAIGN_SOURCE: gm.ORIGINAL_CAMPAIGN_SHA256,
              gm.EXPERIMENT_SOURCE: gm.ICE_FIX_LIMITER_SHA256}
    audit = gm.verify_fork(parent, current, two_substeps=True)
    assert set(audit["modified_sources"]) == {gm.EXPERIMENT_SOURCE, gm.CAMPAIGN_SOURCE}
    assert not gm.verify_fork(current, current, two_substeps=True)["modified_sources"]
    for key in ("vfs/__init__.py", gm.CAMPAIGN_SOURCE, gm.EXPERIMENT_SOURCE):
        with pytest.raises(ValueError, match="Unrecognized"):
            gm.verify_fork({**parent, key: "changed"}, current, two_substeps=True)
    day11 = {**parent, **gm.DAY11_ICE_SOURCES, gm.EXPERIMENT_SOURCE: gm.DAY11_LIMITER_SHA256}
    with pytest.raises(ValueError, match="Unrecognized"):
        gm.verify_fork(day11, current, two_substeps=True)
    with pytest.raises(ValueError, match="audited substep"):
        gm.verify_fork(parent, {**current, gm.CAMPAIGN_SOURCE: "changed"}, two_substeps=True)
