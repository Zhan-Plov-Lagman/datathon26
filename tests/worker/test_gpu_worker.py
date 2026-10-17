import threading
import unittest
from unittest.mock import Mock

from app.inference.worker import LocalGPUWorker, WorkerBusy


class WorkerTests(unittest.TestCase):
    def test_busy_rejected_and_failure_releases_worker(self):
        worker = LocalGPUWorker(object())
        entered, release = threading.Event(), threading.Event()
        errors = []

        def operation(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Synthetic test did not release worker")
            return {"complete": True}

        def first_request():
            try:
                worker.analyze("synthetic", "one")
            except Exception as exc:
                errors.append(exc)

        worker.analyzer.analyze = operation
        thread = threading.Thread(target=first_request)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaises(WorkerBusy):
                worker.refine("synthetic", "two", [])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        worker.analyzer.analyze = Mock(side_effect=ValueError("Synthetic failure"))
        with self.assertRaises(ValueError):
            worker.analyze("synthetic", "three")
        worker.analyzer.analyze = Mock(return_value={"complete": True})
        self.assertTrue(worker.analyze("synthetic", "four")["complete"])


if __name__ == "__main__":
    unittest.main()
