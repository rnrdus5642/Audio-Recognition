"""Reporting contract checks; no network, GPU, or real training files."""
import tempfile
from pathlib import Path
import unittest

from watch import Watcher, atomic_json, discover, evaluation_text, read_json


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="watch-test-", dir=Path(__file__).parent)
        self.run = Path(self.temp.name)
        self.config = {"run_dir": str(self.run), "thread_id": "test-thread"}
        self.sent = []

    def tearDown(self):
        self.temp.cleanup()

    def transport(self, config, item, path):
        self.sent.append(item["id"])
        return "fake-queue-receipt"

    def evaluation(self, name="monitor-one", **extra):
        data = {"status": "complete", "macro_cer": .25, "nonblank_fraction": .9,
                "identity": {"weights_id": name, "manifest_sha256": "fixed-monitor"},
                "groups": {f"age:{age}": {"cer": .25} for age in range(3, 7)}}
        data.update(extra)
        atomic_json(self.run / "evaluations" / (name + ".json"), data)
        return data

    def test_only_complete_evaluations(self):
        self.evaluation()
        self.evaluation("monitor-partial.partial")
        self.evaluation("monitor-incomplete", status="running")
        (self.run / "evaluations" / "broken.json").write_text('{"status":')
        results = discover(self.run, {})
        self.assertEqual(len(results), 1)
        self.assertIn("25.00%", results[0]["body"])
        self.assertNotIn("speakers", results[0]["body"])

    def test_restart_does_not_duplicate_accepted_report(self):
        self.evaluation()
        first = Watcher(self.config, self.transport)
        first.scan()
        first.scan()
        Watcher(self.config, self.transport).scan()
        self.assertEqual(len(self.sent), 1)

    def test_failed_delivery_survives_restart(self):
        self.evaluation()
        def fail(*args):
            raise RuntimeError("offline")
        first = Watcher(self.config, fail)
        first.scan()
        self.assertEqual(next(iter(first.state["reports"].values()))["status"], "pending")
        second = Watcher(self.config, self.transport)
        second.scan()  # Backoff must survive a watcher restart.
        self.assertFalse(self.sent)
        for value in second.state["reports"].values():
            value["retry_after"] = 0
        second.scan()
        self.assertEqual(len(self.sent), 1)

    def test_same_monitor_change_and_age_metrics(self):
        previous = self.evaluation("monitor-before", macro_cer=.30)
        current = self.evaluation("monitor-after", macro_cer=.25)
        body = evaluation_text(current, previous)
        self.assertIn("-5.00%p", body)
        self.assertIn("3세 25.00%", body)
        self.assertIn("정답률이 아닙니다", body)

    def test_ack_prevents_retry_after_ambiguous_delivery(self):
        self.evaluation()
        watcher = Watcher(self.config, self.transport)
        item = discover(self.run, {})[0]
        atomic_json(self.run / "reporting" / "acknowledged" / (item["id"] + ".json"), {"id": item["id"]})
        watcher.scan()
        self.assertFalse(self.sent)
        self.assertEqual(watcher.state["reports"][item["id"]]["status"], "acknowledged")

    def test_final_test_reported_with_export_once(self):
        score = self.evaluation("final-test")
        atomic_json(self.run / "result.json", {"status": "complete", "test": score, "model_sha256": "export"})
        atomic_json(self.run / "progress.json", {"status": "complete"})
        watcher = Watcher(self.config, self.transport)
        self.assertTrue(watcher.scan())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(read_json(self.run / "reporting" / "status.json")["status"], "complete")

    def test_diagnosis_reports_without_changing_progress(self):
        progress = {"status": "needs_diagnosis", "stage": "head_gate_failed", "total_updates": 1000}
        atomic_json(self.run / "progress.json", progress)
        Watcher(self.config, self.transport).scan()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(read_json(self.run / "progress.json"), progress)


if __name__ == "__main__":
    unittest.main(verbosity=2)
