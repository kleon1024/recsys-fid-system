"""Independent verifier for randomized external policy evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import ACCEPTANCE_THRESHOLDS


MIN_ESS_FRACTION = 0.05
MAX_IMPORTANCE_WEIGHT = 20.0
MAX_POLICY_VALUE_NORMALIZED_MAE = 0.25
MIN_IDENTIFIED_PAIRS = 2


def kendall_tau(observed, predicted):
    products = []
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            product = np.sign(observed[left] - observed[right]) * np.sign(
                predicted[left] - predicted[right]
            )
            if product:
                products.append(product)
    return float(np.mean(products)) if products else 0.0


def verify_policy_evidence(path: Path | None, manifest_sha256: str):
    unavailable = {
        "available": False,
        "reason": "user-disjoint randomized policy evidence was not supplied",
        "policy_order_pass": False,
    }
    if path is None or not path.exists():
        return unavailable
    evidence = json.loads(path.read_text())
    if evidence.get("schema") != "neural-scm-kuairand-policy-evidence-v1":
        return {**unavailable, "reason": "unsupported policy evidence schema"}
    if evidence.get("world_model_manifest_sha256") != manifest_sha256:
        return {**unavailable, "reason": "policy evidence is not bound to this artifact"}
    policies = evidence.get("policies", [])
    observed = np.asarray([row["observed_value"] for row in policies])
    predicted = np.asarray([row["predicted_value"] for row in policies])
    tau = kendall_tau(observed, predicted)
    value_normalized_mae = float(
        np.abs(observed - predicted).mean()
        / max(np.abs(observed).mean(), 1e-8)
    )
    support = bool(policies) and all(
        row["effective_sample_fraction"] >= MIN_ESS_FRACTION
        and row["maximum_importance_weight"] <= MAX_IMPORTANCE_WEIGHT
        for row in policies
    )
    identified = sum(
        bool(row.get("identified"))
        for row in evidence.get("pairwise_observed_differences", [])
    )
    pass_gate = (
        len(policies) >= 3
        and support
        and identified >= MIN_IDENTIFIED_PAIRS
        and tau >= ACCEPTANCE_THRESHOLDS["policy_kendall_tau"]
        and value_normalized_mae <= MAX_POLICY_VALUE_NORMALIZED_MAE
        and all(bool(value) for value in evidence.get("gates", {}).values())
    )
    return {
        "available": True,
        "schema": evidence["schema"],
        "policies": len(policies),
        "identified_policy_pairs": identified,
        "policy_kendall_tau": tau,
        "policy_value_normalized_mae": value_normalized_mae,
        "importance_support": support,
        "policy_order_pass": pass_gate,
        "evidence_decision": evidence.get("decision"),
        "evidence_boundary": evidence.get("evidence_boundary"),
    }
