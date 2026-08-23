import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fid_lab.feed_loop.release import (
    apply_launch_decision,
    initial_release_state,
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
        state = initial_release_state("basic", artifact("basic"))
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
        state = initial_release_state("basic", artifact("basic"))
        state, promotion = apply_launch_decision(
            state,
            "basic__local_context",
            artifact("local"),
            "pass_unified_lt_nonnegative",
            "F-LR-003",
        )

        self.assertTrue(promotion["promoted"])
        self.assertEqual(state["active_key"], "basic__local_context")
        self.assertEqual(state["rollback_key"], "basic")
        self.assertEqual(
            state["rollback_artifact"]["artifact_id"], "sha256:basic"
        )

    def test_release_manifest_binds_the_exact_report_bytes(self):
        state = initial_release_state("basic", artifact("basic"))
        report = {
            "release_state": state,
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


if __name__ == "__main__":
    unittest.main()
