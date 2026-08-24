"""Launch-ledger projection for the Feed Posting V43 campaign."""

from __future__ import annotations


def build_feed_posting_v43_records(root, load, record):
    ladder_relative = (
        "reports/launches/2026-08-24-feed-posting-v43-esmm-ladder-400k.json"
    )
    ladder, evidence = load(root, ladder_relative)
    records = []
    for index, row in enumerate(ladder["launches"], 1):
        stage = "fine" if row["stage"] == "fine_incremental" else row["stage"]
        records.append(record(
            launch_id=f"L-FEED-POST-V43-LADDER-{index:03d}",
            surface="feed_posting", stage=stage,
            change_type=f"entire_space_{row['stage']}",
            control=row["control"], treatment=row["treatment"],
            decision=row["decision"], evidence=evidence,
            primary_metric="joint_publish_with_platform_lt_and_risk",
            evidence_boundary=ladder["evidence_boundary"],
        ))
    powered = (
        (
            "reports/launches/2026-08-24-feed-posting-v43-esmm-din-raw-ab-10m.json",
            "entire_space_objective_raw_blend",
        ),
        (
            "reports/launches/2026-08-24-feed-posting-v43-esmm-wide-deep-raw-ab-10m.json",
            "wide_deep_raw_blend",
        ),
        (
            "reports/launches/2026-08-24-feed-posting-v43-esmm-wide-deep-standardized-ab-10m.json",
            "wide_deep_standardized_residual",
        ),
    )
    for index, (relative, change_type) in enumerate(powered, 1):
        report, evidence = load(root, relative)
        records.append(record(
            launch_id=f"L-FEED-POST-V43-POWERED-{index:03d}",
            surface="feed_posting", stage="end_to_end",
            change_type=change_type,
            control=(
                f"{report['control']}_{report['control_blend_mode']}_"
                f"{report['control_blend']:.2f}"
            ),
            treatment=(
                f"{report['treatment']}_{report['treatment_blend_mode']}_"
                f"{report['treatment_blend']:.2f}"
            ),
            decision=report["decision"], evidence=evidence,
            primary_metric="creator_randomized_publish_and_platform_lt",
            evidence_boundary=report["evidence_boundary"],
        ))
    mediation_relative = (
        "reports/launches/2026-08-24-feed-posting-v43-cross-day-mediation.json"
    )
    mediation, evidence = load(root, mediation_relative)
    records.append(record(
        launch_id="L-FEED-POST-V43-MEDIATION-001",
        surface="feed_posting", stage="end_to_end",
        change_type="cross_day_supply_mediation",
        control=mediation["control"]["posting_policy"],
        treatment=mediation["treatment"]["posting_policy"],
        decision=(
            "pass_supply_primary_consumer_noninferior"
            if mediation["decision"] == "ecosystem_v4_pass"
            else "hold_cross_day_mediation"
        ),
        evidence=evidence,
        primary_metric="creator_posts_with_feed_lt_noninferiority",
        evidence_boundary=mediation["evidence_boundary"],
    ))
    return records
