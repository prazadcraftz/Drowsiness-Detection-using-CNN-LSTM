"""Deterministic worker tests; injected backends only, never real audio."""

from threading import Event, get_ident

import pytest

from driver_monitor.alerts import AlertManager
from driver_monitor.metrics import SessionMetrics
from test_session_metrics import decision


def drain(manager, count):
    with manager._condition:
        assert manager._condition.wait_for(
            lambda: manager.completed_playback_count + manager.failed_playback_count >= count, timeout=2)


def test_entry_once_repeated_alert_no_restart_and_clean_shutdown():
    calls = []
    with AlertManager(backend=lambda: calls.append(get_ident())) as manager:
        manager.update(decision(0,"ALERT"))
        drain(manager,1)
        for t in (1,2,3,10):
            manager.update(decision(t,"ALERT"))
        assert len(calls) == manager.audible_trigger_count == 1
        assert calls[0] != get_ident()
        worker = manager._worker
    assert not worker.is_alive()
    manager.close()
    assert manager.snapshot()["closed"]


@pytest.mark.parametrize("reentry,expected", [(4.999,1),(5,2),(5.001,2)])
def test_cooldown_boundaries(reentry,expected):
    with AlertManager(backend=lambda: None) as manager:
        manager.update(decision(0,"ALERT"))
        drain(manager,1)
        manager.update(decision(1))
        manager.update(decision(reentry,"ALERT"))
        if expected == 2:
            drain(manager,2)
        assert manager.audible_trigger_count == expected
        assert manager.suppressed_trigger_count == 2-expected
        assert manager.snapshot(reentry)["cooldown_remaining_seconds"] == pytest.approx(5 if expected==2 else .001)


def test_suppression_does_not_suppress_logical_event():
    metrics = SessionMetrics()
    with AlertManager(backend=lambda: None) as manager:
        for t,state in [(0,"ALERT"),(1,"ATTENTIVE"),(2,"ALERT")]:
            d = decision(t,state)
            metrics.update(d)
            manager.update(d)
            if t == 0:
                drain(manager,1)
        summary = metrics.snapshot(manager.snapshot(2))
    assert summary["events"]["alert"]["count"] == 2
    assert summary["audio"]["audible_trigger_count"] == summary["audio"]["suppressed_trigger_count"] == 1


def test_disabled_never_starts_worker_or_backend():
    with AlertManager(enabled=False,backend=lambda: pytest.fail("Audio disabled")) as manager:
        manager.update(decision(0,"ALERT"))
        manager.update(decision(1,"ALERT"))
        assert manager._worker is None and manager.audible_trigger_count == 0
        assert manager.disabled_trigger_count == 1


def test_backend_failure_recorded_without_crashing_or_losing_logical_policy():
    def fail():
        raise OSError("no audio device")
    with AlertManager(backend=fail) as manager:
        d = decision(0,"ALERT")
        manager.update(d)
        drain(manager,1)
        manager.update(decision(1,"ALERT"))
        assert d.driver.alert_active and manager.last_error == "no audio device"
        assert manager.failed_playback_count == 1 and manager.audible_trigger_count == 1


def test_blocking_playback_never_blocks_update_and_busy_queue_is_bounded():
    entered, release = Event(), Event()
    def backend():
        entered.set()
        assert release.wait(timeout=2)
    manager = AlertManager(cooldown_seconds=0,backend=backend)
    try:
        manager.update(decision(0,"ALERT"))
        assert entered.wait(timeout=2)
        for i in range(1,100):
            manager.update(decision(i,"ALERT" if i%2==0 else "ATTENTIVE"))
        assert manager.snapshot()["currently_playing"]
        assert manager.audible_trigger_count == 1 and manager.suppressed_trigger_count == 49
        assert not manager.snapshot()["pending"]
    finally:
        release.set()
        manager.close()
    assert not manager._worker.is_alive()


def test_close_without_worker_and_update_after_close():
    manager = AlertManager(backend=lambda: None)
    manager.close()
    manager.close()
    with pytest.raises(RuntimeError,match="closed"):
        manager.update(decision(0,"ALERT"))


@pytest.mark.parametrize("value", [-1,float("nan"),float("inf"),True,"5"])
def test_invalid_cooldown(value):
    with pytest.raises(ValueError,match="alert_cooldown"):
        AlertManager(cooldown_seconds=value)


def test_clock_and_cooldown_do_not_drift_on_invalid_update():
    with AlertManager(enabled=False) as manager:
        manager.update(decision(0,"ALERT"))
        with pytest.raises(ValueError):
            manager.update(decision(0))
        manager.update(decision(1,"ALERT"))
        assert manager.disabled_trigger_count == 1


def test_worker_start_failure_is_nonfatal(monkeypatch):
    import driver_monitor.alerts.alert_manager as module
    def fail(self):
        raise RuntimeError("thread unavailable")
    monkeypatch.setattr(module.Thread,"start",fail)
    with AlertManager(backend=lambda: None) as manager:
        manager.update(decision(0,"ALERT"))
        assert manager.last_error == "thread unavailable" and manager._worker is None


def test_no_worker_leak_after_multiple_managers():
    workers = []
    for i in range(10):
        with AlertManager(backend=lambda: None) as manager:
            manager.update(decision(0,"ALERT"))
            drain(manager,1)
            workers.append(manager._worker)
    assert all(not worker.is_alive() for worker in workers)
