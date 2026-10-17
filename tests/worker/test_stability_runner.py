import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.video.sampling import DeadlineExceeded, VideoError
from scripts.benchmark_worker import run_stability


class StabilityRunnerTests(unittest.TestCase):
    def test_expected_failures_continue_and_stdout_has_no_scene_text(self):
        self._run(unexpected=False)

    def test_unexpected_failure_does_not_hide_or_stop_later_tasks(self):
        self._run(unexpected=True)

    def _run(self, unexpected):
        cuda = Mock()
        cuda.memory_allocated.return_value = 1024
        cuda.memory_reserved.return_value = 2048
        observer = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), metrics=lambda: {"format_retries": 0})
        report = {"complete": True, "metrics": {"decode_sec": 0.1, "total_sec": 0.2},
                  "windows": [{"description": "SYNTHETIC_SCENE_CONTENT"}]}
        outcomes = [report, VideoError(), RuntimeError() if unexpected else report,
                    DeadlineExceeded(), DeadlineExceeded(), report]
        worker = SimpleNamespace(analyzer=SimpleNamespace(observer=observer), analyze=Mock(side_effect=outcomes))
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(stdout):
            rows = run_stability(worker, Path("synthetic.mov"), Path(directory), 30)
            self.assertEqual(len(rows), 6)
            self.assertTrue(rows[-1]["complete"])
            self.assertEqual(all(row["expected_outcome"] for row in rows), not unexpected)
            self.assertNotIn("SYNTHETIC_SCENE_CONTENT", (Path(directory) / "stability-metrics.json").read_text())
        self.assertEqual(worker.analyze.call_count, 6)
        self.assertNotIn("SYNTHETIC_SCENE_CONTENT", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
