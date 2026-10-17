import time
import unittest
from unittest.mock import Mock

from app.inference.qwen import QwenObserver
from app.inference.schema import ModelOutputError, parse_observations
from app.video.sampling import DeadlineExceeded


class FormatRetryTests(unittest.TestCase):
    def test_syntax_error_has_safe_reason(self):
        with self.assertRaises(ModelOutputError) as raised:
            parse_observations('{"private":', [0])
        self.assertEqual(raised.exception.reason, "invalid_json")
        self.assertNotIn("private", str(raised.exception))

    def test_single_short_retry_preserves_deadline(self):
        observer = object.__new__(QwenObserver)
        observer._observe_once = Mock(side_effect=[ModelOutputError("invalid_json"), {"observations": []}])
        deadline = time.monotonic() + 10
        result = observer.observe([], "synthetic", deadline)
        self.assertEqual(result["generation"]["attempts"], 2)
        self.assertEqual(observer._observe_once.call_args.kwargs, {"concise": True})
        self.assertEqual(observer._observe_once.call_args.args[-1], deadline)

    def test_bad_schema_and_repeated_syntax_error_stay_failed(self):
        observer = object.__new__(QwenObserver)
        observer._observe_once = Mock(side_effect=ModelOutputError("invalid_schema"))
        with self.assertRaises(ModelOutputError):
            observer.observe([])
        self.assertEqual(observer._observe_once.call_count, 1)
        observer._observe_once = Mock(side_effect=ModelOutputError("invalid_json"))
        with self.assertRaises(ModelOutputError):
            observer.observe([])
        self.assertEqual(observer._observe_once.call_count, 2)
        with self.assertRaises(DeadlineExceeded):
            observer.observe([], deadline=time.monotonic() - 1)
        self.assertEqual(observer._observe_once.call_count, 2)
