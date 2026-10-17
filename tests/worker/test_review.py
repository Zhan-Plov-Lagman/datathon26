import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.inference.review import create_draft, render_draft, timestamp


def report():
    return {"schema_version": "observations.v1", "complete": True, "windows": [
        {"window_id": 0, "window_start_sec": 0, "window_end_sec": 2,
         "operation_context": "Synthetic test", "visibility_limits": ["Synthetic limit"],
         "observations": [{"start_sec": 0.125, "end_sec": 1.75,
                           "description": "![synthetic](https://example.com/image)"}]}]}


class ReviewTests(unittest.TestCase):
    def test_preserves_times_limits_and_unreviewed_status(self):
        draft = render_draft(report(), "test.MOV")
        self.assertIn("00:00.125–00:01.750", draft)
        self.assertIn("Synthetic limit", draft)
        self.assertIn("НЕ ПРОВЕРЕНО", draft)
        self.assertNotIn("![synthetic]", draft)
        self.assertEqual(timestamp(59.9999), "01:00.000")
        with self.assertRaises(ValueError):
            render_draft({"complete": False}, "test.MOV")

    def test_failure_then_success_and_no_content_in_console(self):
        analyzer = Mock()
        analyzer.analyze.side_effect = [RuntimeError("PRIVATE SYNTHETIC ERROR"), report()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                self.assertFalse(create_draft(analyzer, Path("test.mov"), output, 1, 30))
                self.assertTrue(create_draft(analyzer, Path("test.mov"), output, 2, 30))
            self.assertNotIn("PRIVATE", console.getvalue())
            self.assertNotIn("Synthetic", console.getvalue())
            self.assertFalse((output / "001.md").exists())
            saved = json.loads((output / "002.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["review_metadata"]["human_reviewed"])
            self.assertTrue((output / "002.md").is_file())


if __name__ == "__main__":
    unittest.main()
