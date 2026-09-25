"""Async gate around the Jev (TypeSafe System One) plan-checking API.

Hard constraints from the evaluation plan (§3):
  - Jev answers in ~230-350 ms over the network. The control loop's planner
    budget is 100 ms total (§4). So Jev can NEVER be called synchronously
    from the control loop — it runs as a background task that checks
    whatever plan was most recently submitted, at <=2 Hz, and the control
    loop only ever reads the last cached verdict (non-blocking, ~microseconds).
  - When the network is slow, down, or the rate limiter hasn't produced a
    fresh verdict yet, a locally distilled fallback supervisor answers
    instead. The control loop never stalls waiting on either.
  - No patient data leaves the machine: `sanitize_payload` whitelists a
    fixed set of numeric fields (joint angles, planned action, predicted
    tip position) and drops everything else, images included, before a
    request is built.

This module has no torch/GPU dependency and uses only the standard
library, so it runs equally well on the cloud scaffolding container and on
the robot workstation.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# Fields allowed to leave the machine in a Jev request. Deliberately a
# small, numeric, non-identifying whitelist — extend it only with fields
# that are still just plan/state numbers, never raw sensor data or
# anything that could identify a patient or a procedure.
_ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        "joint_positions",
        "planned_action",
        "predicted_tip_xyz",
        "goal_tip_xyz",
        "rcm_deviation_m",
        "task_id",
        "episode_step",
    }
)


def sanitize_payload(plan_summary: dict) -> dict:
    """Drop every key not on the whitelist (images, ids, free-text, or
    anything else) before a plan summary is allowed to leave the machine."""
    return {k: v for k, v in plan_summary.items() if k in _ALLOWED_PAYLOAD_KEYS}


@dataclass
class JevVerdict:
    ok: bool                 # False => veto: don't execute the pending plan
    slow_factor: float        # >1.0 => planner should reduce speed/step size
    confidence: float
    source: str               # "jev" | "local_fallback" | "no_verdict_yet"
    latency_ms: float
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.timestamp


def _default_local_fallback(plan_summary: dict) -> JevVerdict:
    """Placeholder local supervisor: a cheap, explainable rule so the gate
    is never silently a no-op when the network is unavailable. Replace with
    a real distilled model (trained to mimic Jev's own verdicts on logged
    traffic) before relying on this in evaluation — see docs/evaluation_plan.md §3.
    """
    t0 = time.monotonic()
    dev = plan_summary.get("rcm_deviation_m", 0.0)
    veto = dev is not None and dev > 0.003  # 3 mm, matches the precision success threshold
    return JevVerdict(
        ok=not veto,
        slow_factor=1.0,
        confidence=0.5,  # deliberately low: this is a coarse stand-in, not the real supervisor
        source="local_fallback",
        latency_ms=(time.monotonic() - t0) * 1000.0,
    )


class JevGate:
    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        min_interval_s: float = 0.5,  # <=2 Hz
        timeout_s: float = 0.4,        # bounds Jev's ~230-350 ms typical latency
        local_fallback: Callable[[dict], JevVerdict] | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.local_fallback = local_fallback or _default_local_fallback

        self._latest_plan: dict | None = None
        self._latest_verdict = JevVerdict(
            ok=True, slow_factor=1.0, confidence=0.0, source="no_verdict_yet", latency_ms=0.0
        )
        self._lock = asyncio.Lock()

    # -- called from the (fast) control/planning loop -----------------
    def submit_plan(self, plan_summary: dict) -> None:
        """Cheap, synchronous, no I/O: just records the latest plan for the
        background loop to pick up. Call this every planner tick."""
        self._latest_plan = plan_summary

    def get_verdict(self) -> JevVerdict:
        """Non-blocking read of the last verdict. Always returns
        immediately — this is the only method the control loop calls."""
        return self._latest_verdict

    # -- background supervisor loop ------------------------------------
    async def run_forever(self) -> None:
        """Run as an asyncio task alongside the control loop, never awaited
        by it. Repeatedly: wait out the rate limit, check the latest
        submitted plan, and update the cached verdict."""
        while True:
            start = time.monotonic()
            if self._latest_plan is not None:
                verdict = await self._check(self._latest_plan)
                async with self._lock:
                    self._latest_verdict = verdict
            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0.0, self.min_interval_s - elapsed))

    async def _check(self, plan_summary: dict) -> JevVerdict:
        try:
            return await asyncio.wait_for(self._query_jev(plan_summary), timeout=self.timeout_s)
        except (asyncio.TimeoutError, urllib.error.URLError, OSError, ValueError):
            return self.local_fallback(plan_summary)

    async def _query_jev(self, plan_summary: dict) -> JevVerdict:
        payload = sanitize_payload(plan_summary)
        t0 = time.monotonic()
        # Blocking stdlib HTTP call, run off the event loop thread so it
        # doesn't stall other asyncio tasks; `asyncio.wait_for` above still
        # bounds how long the *caller* waits even though the underlying
        # thread may keep running past the timeout.
        body = await asyncio.get_running_loop().run_in_executor(None, self._post, payload)
        latency_ms = (time.monotonic() - t0) * 1000.0
        data: dict[str, Any] = json.loads(body)
        return JevVerdict(
            ok=bool(data.get("ok", True)),
            slow_factor=float(data.get("slow_factor", 1.0)),
            confidence=float(data.get("confidence", 1.0)),
            source="jev",
            latency_ms=latency_ms,
        )

    def _post(self, payload: dict) -> bytes:
        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()
