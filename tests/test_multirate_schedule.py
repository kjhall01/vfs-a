"""Discrete event-scheduler tests; independent of PyAM21 and ocean physics."""
import pytest

from workflows.coupled.schedule import event_due, ocean_events


def test_legacy_schedule_reproduces_one_closure_per_ocean_step():
    events = list(ocean_events(1200, 600, 600, 600))
    assert [(e.end_seconds, e.dynamics_dt, e.mixing_dt, e.mesoscale_dt)
            for e in events] == [(600, 600, 600, 600), (1200, 600, 600, 600)]


def test_independent_mixing_and_mesoscale_cadences():
    events = list(ocean_events(3600, 600, 1200, 1800))
    assert [e.end_seconds for e in events] == [600, 1200, 1800, 2400, 3000, 3600]
    assert [e.mixing_dt for e in events] == [None, 1200, None, 1200, None, 1200]
    assert [e.mesoscale_dt for e in events] == [None, None, 1800, None, None, 1800]
    assert sum(e.dynamics_dt for e in events) == 3600
    assert sum(e.mixing_dt or 0 for e in events) == 3600
    assert sum(e.mesoscale_dt or 0 for e in events) == 3600


def test_short_final_segment_flushes_all_closure_time():
    events = list(ocean_events(1500, 600, 1200, 1200))
    assert [e.dynamics_dt for e in events] == [600, 600, 300]
    assert [e.mixing_dt for e in events] == [None, 1200, 300]
    assert [e.mesoscale_dt for e in events] == [None, 1200, 300]


@pytest.mark.parametrize("values", [
    (0, 600, 600, 600),
    (1200, 0, 600, 600),
    (1200, 600, 601, 600),
    (1200, 600, 600, 1199),
    (1200, 600, True, 600),
])
def test_invalid_ocean_event_intervals(values):
    with pytest.raises(ValueError):
        list(ocean_events(*values))


def test_absolute_ale_schedule_is_checkpoint_independent():
    assert not event_due(1200, 2400)
    assert event_due(2400, 2400)
    assert event_due(4800, 2400)
    assert not event_due(3600, 2400)
    with pytest.raises(ValueError):
        event_due(1200.0, 2400)
