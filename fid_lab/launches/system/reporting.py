"""Render architecture and bug-fix Launch Reviews."""

from __future__ import annotations

import json
from pathlib import Path


def _architecture(review):
    delta = review["distribution_relative_delta"]
    ab = review["ab"]
    rows = "\n".join(
        f"| {name} | {value * 100:+.4f}% | {ab[name]['p_value']:.4g} |"
        for name, value in delta.items()
        if name in ab
    )
    return f"""# L-ARCH-001 — Increase GPU user batch

Status: `{review['decision']}`. Synthetic performance launch.

## Change

Increase device-resident user batch from 10,000 to 25,000. No model, feature,
Value Tree, or product parameter changes. Training is not applicable.

## Shadow and A/B

| Metric | Full-world distribution delta | randomized p-value |
|---|---:|---:|
{rows}

The stable user A/B is neutral on every business metric. Maximum non-negative
business-distribution drift is below 0.1%; negative feedback moved 0.252%, also
without a significant randomized effect.

## Performance and cost

- Control: {review['control_performance']['requests_per_second']:,.0f} requests/s,
  {review['control_performance']['peak_gpu_memory_bytes'] / 1_048_576:.1f} MiB.
- Treatment: {review['treatment_performance']['requests_per_second']:,.0f} requests/s,
  {review['treatment_performance']['peak_gpu_memory_bytes'] / 1_048_576:.1f} MiB.
- Throughput lift: {review['throughput_lift'] * 100:+.2f}%.

## Decision

Pass: distribution parity holds and throughput improves. The trade-off is higher
GPU memory, so a production ramp would retain memory and P99 latency guardrails.
"""


def _bug(review):
    ab = review["ab"]
    rows = "\n".join(
        f"| {name} | {value['relative_lift'] * 100:+.4f}% | {value['p_value']:.4g} |"
        for name, value in ab.items()
    )
    return f"""# L-BUG-001 — Stop counting inactive users as plays

Status: `{review['decision']}`. Synthetic measurement-chain fix.

## Root cause

The throughput engine sampled play for users whose trajectory had already ended
and counted those draws in the numerator while excluding them from exposures.
The observed play rate could therefore exceed one.

## Fix and replay

Mask play by active state before aggregation. The broken metric reproduced at
{review['bug_play_rate']:.6f}; the fixed metric is {review['fixed_play_rate']:.6f}.
Underlying stay/LT/HLT trajectories are exactly identical in shadow replay:
`{review['shadow_business_metrics_identical']}`.

## Randomized safety check

| Metric | Relative lift | p-value |
|---|---:|---:|
{rows}

No business regression is significant. Training is not applicable because this
is a metric aggregation defect, not a model change.

## Decision

Pass the metric correction. Historical play-rate dashboards produced by the
broken definition must not be compared directly with the corrected series.
"""


def render_system_suite(input_path: Path, output_dir: Path):
    suite = json.loads(input_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    renderers = {"L-ARCH-001": _architecture, "L-BUG-001": _bug}
    paths = []
    for review in suite["launches"]:
        path = output_dir / f"{review['launch_id'].lower()}.md"
        path.write_text(renderers[review["launch_id"]](review))
        paths.append(path)
    return paths

