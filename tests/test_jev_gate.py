import asyncio

from supervisor.jev_gate import JevGate, JevVerdict, _default_local_fallback, sanitize_payload


def test_sanitize_payload_drops_unlisted_keys():
    payload = {
        "joint_positions": [0.1] * 6,
        "patient_id": "12345",           # must never leave the machine
        "raw_image": b"pretend-pixels",  # must never leave the machine
        "rcm_deviation_m": 0.0001,
    }
    sanitized = sanitize_payload(payload)
    assert "patient_id" not in sanitized
    assert "raw_image" not in sanitized
    assert sanitized["joint_positions"] == [0.1] * 6
    assert sanitized["rcm_deviation_m"] == 0.0001


def test_default_local_fallback_vetoes_large_rcm_deviation():
    verdict = _default_local_fallback({"rcm_deviation_m": 0.01})  # 10 mm
    assert verdict.ok is False
    assert verdict.source == "local_fallback"


def test_default_local_fallback_allows_small_rcm_deviation():
    verdict = _default_local_fallback({"rcm_deviation_m": 0.0001})  # 0.1 mm
    assert verdict.ok is True


def test_get_verdict_before_any_check_is_non_blocking_default():
    gate = JevGate(endpoint_url="https://example.invalid/jev", api_key="test")
    v = gate.get_verdict()
    assert v.source == "no_verdict_yet"


def test_check_falls_back_when_query_times_out():
    async def slow_query(plan_summary):
        await asyncio.sleep(1.0)  # much slower than the 50 ms timeout below
        return JevVerdict(ok=True, slow_factor=1.0, confidence=1.0, source="jev", latency_ms=1000.0)

    gate = JevGate(endpoint_url="https://example.invalid/jev", api_key="test", timeout_s=0.05)
    gate._query_jev = slow_query  # bypass the real network call for this test

    verdict = asyncio.run(gate._check({"rcm_deviation_m": 0.0001}))
    assert verdict.source == "local_fallback"


def test_run_forever_updates_cached_verdict_from_submitted_plan():
    async def fake_query(plan_summary):
        return JevVerdict(ok=True, slow_factor=1.0, confidence=0.9, source="jev", latency_ms=1.0)

    gate = JevGate(endpoint_url="https://example.invalid/jev", api_key="test", min_interval_s=0.01, timeout_s=0.2)
    gate._query_jev = fake_query

    async def scenario():
        task = asyncio.create_task(gate.run_forever())
        gate.submit_plan({"rcm_deviation_m": 0.0001, "episode_step": 0})
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    verdict = gate.get_verdict()
    assert verdict.source == "jev"
    assert verdict.ok is True


def test_get_verdict_is_synchronous_and_returns_immediately():
    # The whole point of the gate: the control loop must never await this.
    gate = JevGate(endpoint_url="https://example.invalid/jev", api_key="test")
    import inspect

    assert not inspect.iscoroutinefunction(gate.get_verdict)
    assert not inspect.iscoroutinefunction(gate.submit_plan)
