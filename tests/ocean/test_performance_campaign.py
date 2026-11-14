"""Check throughput accounting separately from costly scientific workloads."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/'validation/performance/run.py'
spec=importlib.util.spec_from_file_location('performance_run',path)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def test_sypd_uses_wall_time_and_all_evolving_steps():
    # Three 600 s steps in 6 wall seconds: 300 simulated seconds per wall second.
    result=bench.summarize([{'wall_s':6.,'process_cpu_s':24.}],3,600.)
    assert result['median_step_wall_s']==2.
    assert result['sypd']==pytest.approx(300/365)
    assert result['wall_hours_per_simulated_day']==pytest.approx(.08)


def test_timing_spread_and_median_are_per_step():
    result=bench.summarize([{'wall_s':x} for x in (12.,4.,8.)],2,60.)
    assert result['median_step_wall_s']==4.
    assert result['min_step_wall_s']==2.
    assert result['max_step_wall_s']==6.


def test_atomic_json_rejects_nonfinite_numbers(tmp_path):
    path=tmp_path/'result.json'
    bench.write(path,{'status':'completed'})
    with pytest.raises(ValueError):bench.write(path,{'sypd':float('nan')})
    assert path.read_text()=='{\n  "status": "completed"\n}\n'
