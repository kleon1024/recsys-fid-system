import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fid_lab.feed_loop.release import (
    apply_launch_decision,
    initial_release_state,
    release_state_from_manifest,
    write_release_manifest,
)


def artifact(name):
    return {
        "artifact_id": f"sha256:{name}",
        "model_name": name,
        "feature_names": [name],
    }


class ReleaseStateTest(unittest.TestCase):
    def test_hold_and_reject_keep_the_last_accepted_control(self):
        state = initial_release_state("basic", artifact("basic"), "base-models")
        state, held = apply_launch_decision(
            state,
            "basic__sequence",
            artifact("sequence"),
            "hold_unified_lt_uncertain",
            "F-LR-001",
        )
        state, rejected = apply_launch_decision(
            state,
            "basic__realtime",
            artifact("realtime"),
            "reject_unified_lt_negative",
            "F-LR-002",
        )

        self.assertFalse(held["promoted"])
        self.assertFalse(rejected["promoted"])
        self.assertEqual(state["active_key"], "basic")
        self.assertIsNone(state["rollback_key"])

    def test_pass_promotes_atomically_and_preserves_rollback(self):
        state = initial_release_state("basic", artifact("basic"), "base-models")
        state, promotion = apply_launch_decision(
            state,
            "basic__local_context",
            artifact("local"),
            "pass_unified_lt_nonnegative",
            "F-LR-003",
            "local-models",
        )

        self.assertTrue(promotion["promoted"])
        self.assertEqual(state["active_key"], "basic__local_context")
        self.assertEqual(state["rollback_key"], "basic")
        self.assertEqual(
            state["rollback_artifact"]["artifact_id"], "sha256:basic"
        )
        self.assertEqual(state["active_artifact_collection"], "local-models")
        self.assertEqual(state["rollback_artifact_collection"], "base-models")

    def test_release_manifest_binds_the_exact_report_bytes(self):
        state = initial_release_state("basic", artifact("basic"), "test-artifacts")
        report = {
            "release_state": state,
            "report_logical_key": "test-release-report",
            "artifact_collection": "test-artifacts",
            "production_readiness": "hold_synthetic_rates",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            release_path = root / "release.json"
            report_path.write_text(json.dumps(report) + "\n")
            manifest = write_release_manifest(report_path, report, release_path)
            saved = json.loads(release_path.read_text())

        self.assertEqual(saved, manifest)
        self.assertEqual(saved["active_control_key"], "basic")
        self.assertEqual(len(saved["source_report"]["sha256"]), 64)

    def test_release_continuity_fails_closed_on_wrong_campaign_base(self):
        release = {
            "active_control_key": "basic",
            "active_control_artifact": artifact("basic"),
            "rollback_key": None,
            "rollback_artifact": None,
            "promoted_by_launch": None,
        }
        with self.assertRaises(ValueError):
            release_state_from_manifest(
                release, "basic__realtime__local_context"
            )


if __name__ == "__main__":
    unittest.main()
