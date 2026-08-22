"""Render one auditable Markdown Launch Review per independent launch."""

from __future__ import annotations

import json
from pathlib import Path


def _percent(value: float) -> str:
    return f"{value * 100:+.4f}%"


def _training_result(mode: str) -> str:
    meanings = {
        "reuse_control_model_to_isolate_feature_effect": (
            "Reused the frozen control model to isolate feature-quality impact."
        ),
        "reuse_control_model_to_isolate_freshness_effect": (
            "Reused the frozen control model to isolate state-freshness impact."
        ),
        "no_weight_update_strategy_only": "No retraining; strategy-only change.",
        "no_weight_update_trigger_only": "No retraining; trigger-only product change.",
        "no_weight_update_value_tree_only": "No retraining; Value Tree-only change.",
        "no_weight_update_constraint_only": "No retraining; constraint-only change.",
        "reuse_control_model_chain_fix_only": (
            "Reused the frozen model and removed only the diagnosed chain defect."
        ),
    }
    return meanings.get(mode, mode)


def render_launch_review(launch: dict[str, object]) -> str:
    spec = launch["spec"]
    ab = launch["ab"]
    truth = launch["known_dgp_effect"]
    primary = spec["primary_metric"]
    primary_result = ab[primary]
    metrics = []
    for name in ("stay_per_exposure", "lt_rate", "hlt_rate", "negative_rate"):
        result = ab[name]
        metrics.append(
            f"| {name} | {_percent(result['relative_lift'])} | "
            f"{result['p_value']:.4g} | {_percent(truth[name]['relative_effect'])} |"
        )
    decision = launch["decision"]
    interpretation = (
        "The primary metric cleared the randomized gate without a significant HLT or "
        "negative-feedback regression."
        if decision == "pass_primary_metric"
        else "The evidence does not justify rollout. Preserve control and revise or stop."
    )
    return f"""# {spec['launch_id']} — {spec['title']}

Status: `{decision}`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `{spec['category']}`
- Owner: `{spec['owner']}`
- Hypothesis: {spec['hypothesis']}
- Change: {spec['change']}
- Product dependency: {spec['product_dependency']}
- Short-term value: {spec['short_term_value']}
- Long-term value: {spec['long_term_value']}

## Training and artifacts

{_training_result(launch['protocol']['training'])} Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
{chr(10).join(metrics)}

Primary metric `{primary}`: {_percent(primary_result['relative_lift'])},
p={primary_result['p_value']:.4g}. Absolute 95% confidence interval:
[{primary_result['confidence_interval'][0]:+.8f},
{primary_result['confidence_interval'][1]:+.8f}].

## Gate and review

{interpretation}

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: {launch['performance']['control']['requests_per_second']:,.0f} requests/s,
  {launch['performance']['control']['peak_gpu_memory_bytes'] / 1_048_576:.1f} MiB peak.
- Treatment: {launch['performance']['treatment']['requests_per_second']:,.0f} requests/s,
  {launch['performance']['treatment']['peak_gpu_memory_bytes'] / 1_048_576:.1f} MiB peak.

## Next action

{'Ramp only through the next guarded stage and continue monitoring.' if decision == 'pass_primary_metric' else 'Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.'}
"""


def render_suite(input_path: Path, output_dir: Path) -> list[Path]:
    suite = json.loads(input_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for launch in suite["launches"]:
        path = output_dir / f"{launch['spec']['launch_id'].lower()}.md"
        path.write_text(render_launch_review(launch))
        paths.append(path)
    rows = [
        f"| [{launch['spec']['launch_id']}]({launch['spec']['launch_id'].lower()}.md) "
        f"| {launch['spec']['category']} | {launch['decision']} | "
        f"{_percent(launch['ab'][launch['spec']['primary_metric']]['relative_lift'])} |"
        for launch in suite["launches"]
    ]
    index = output_dir / "README.md"
    index.write_text(
        "# Main Feed Independent Launch Reviews\n\n"
        "All launches use the same immutable evidence JSON, common-random shadow, "
        "stable user-level A/B, and multi-metric gate.\n\n"
        "| Launch | Category | Decision | Primary lift |\n"
        "|---|---|---|---:|\n"
        + "\n".join(rows)
        + "\n"
    )
    paths.append(index)
    return paths
