"""Shared factual experiment-window execution."""

from __future__ import annotations


def run_window(kernel, plan, logical_time, steps, evidence=None):
    start = logical_time
    for _ in range(steps):
        tick = kernel.step(logical_time, plan)
        if evidence is not None:
            evidence.append(tick)
        logical_time += 1
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    return logical_time, events.select(events.ingest_time >= start)
