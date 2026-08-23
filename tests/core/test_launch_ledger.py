from __future__ import annotations

from pathlib import Path
import unittest

from fid_lab.launches.ledger import build_launch_ledger


class LaunchLedgerTest(unittest.TestCase):
    def test_ledger_validates_evidence_and_exposes_missing_business_cells(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ledger = build_launch_ledger(root)
        self.assertGreaterEqual(ledger["summary"]["records"], 20)
        self.assertGreater(ledger["summary"]["passed"], 0)
        self.assertEqual(ledger["coverage"]["main_feed"]["coarse"], "evidenced")
        self.assertEqual(ledger["coverage"]["main_feed"]["fine"], "evidenced")
        self.assertEqual(ledger["coverage"]["feed_posting"]["fine"], "missing")
        self.assertGreater(ledger["summary"]["missing_cells"], 0)
        for record in ledger["records"]:
            self.assertTrue((root / record["evidence"]["report"]).exists())
            self.assertEqual(len(record["evidence"]["report_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
