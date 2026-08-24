"""Content-bound family partitions and resumable structural bridge builds."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import torch


BUILD_STATE_SCHEMA = "v4-structural-bridge-build-state-v1"


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _plan_payload(family_plan):
    return [
        {"family_id": family_id, "split": split, "rows": rows}
        for family_id, split, rows in family_plan
    ]


def load_or_create_build_state(output_dir, config, family_plan):
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "build-state.json"
    expected_config = asdict(config)
    expected_plan = _plan_payload(family_plan)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != BUILD_STATE_SCHEMA:
            raise ValueError("structural bridge build-state schema mismatch")
        if state.get("config") != expected_config:
            raise ValueError("structural bridge resume config mismatch")
        if state.get("family_plan") != expected_plan:
            raise ValueError("structural bridge resume family plan mismatch")
        return state
    if any(output_dir.iterdir()):
        raise ValueError("nonempty structural bridge directory lacks build state")
    state = {
        "schema": BUILD_STATE_SCHEMA,
        "config": expected_config,
        "family_plan": expected_plan,
        "completed": {},
    }
    _atomic_json(state_path, state)
    return state


def family_key(family_id, split):
    return f"{split}:family-{family_id}"


def _comparable_config(config):
    normalized = dict(config)
    normalized.pop("test_family_id", None)
    return normalized


def import_compatible_parts(output_dir, state, source_dir):
    source_path = source_dir / "build-state.json"
    if not source_path.exists():
        raise ValueError("reused structural build lacks build state")
    source = json.loads(source_path.read_text())
    if _comparable_config(source["config"]) != _comparable_config(state["config"]):
        raise ValueError("reused structural build config mismatch")
    expected = {
        family_key(row["family_id"], row["split"]): row["rows"]
        for row in state["family_plan"]
    }
    parts = output_dir / "parts"
    parts.mkdir(exist_ok=True)
    changed = False
    for key, record in source.get("completed", {}).items():
        if key not in expected or record["rows"] != expected[key]:
            continue
        source_part = source_dir / record["path"]
        if not source_part.exists() or _hash(source_part) != record["sha256"]:
            raise ValueError("reused structural family content mismatch")
        target_part = parts / source_part.name
        if not target_part.exists():
            try:
                os.link(source_part, target_part)
            except OSError:
                shutil.copy2(source_part, target_part)
        state["completed"][key] = {
            **record,
            "path": str(target_part.relative_to(output_dir)),
        }
        changed = True
    if changed:
        _atomic_json(output_dir / "build-state.json", state)


def write_family_part(
    output_dir, state, family_id, split, tensors, paired, family,
):
    parts = output_dir / "parts"
    parts.mkdir(exist_ok=True)
    key = family_key(family_id, split)
    path = parts / f"{split}-family-{family_id}.pt"
    torch.save({
        "tensors": tensors,
        "paired": paired,
        "family": family,
    }, path)
    state["completed"][key] = {
        "path": str(path.relative_to(output_dir)),
        "sha256": _hash(path),
        "rows": family["rows"],
    }
    _atomic_json(output_dir / "build-state.json", state)


def load_family_part(output_dir, state, family_id, split):
    record = state["completed"].get(family_key(family_id, split))
    if record is None:
        return None
    path = output_dir / record["path"]
    if not path.exists() or _hash(path) != record["sha256"]:
        raise ValueError("structural family partition content mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["family"]["rows"] != record["rows"]:
        raise ValueError("structural family partition row mismatch")
    return payload


def write_final_manifest(output_dir, manifest):
    _atomic_json(output_dir / "manifest.json", manifest)
