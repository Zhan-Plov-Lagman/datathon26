from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from app.evidence.frames import extract_evidence, extract_evidence_batch
from app.inference.pipeline import LocalVideoAnalyzer
from app.inference.schema import ModelOutputError, parse_observations
from app.video.sampling import DeadlineExceeded, VideoError, plan_windows, probe_video, sample_frames


class TimelineTests(unittest.TestCase):
    def test_windows_cover_tail_and_boundaries(self):
        windows = plan_windows(25, 12, 2, 6)
        self.assertEqual([(w.start_sec, w.end_sec) for w in windows], [(0, 12), (10, 22), (20, 25)])
        self.assertEqual(windows[-1].targets[-1], 25)
        self.assertEqual(len(plan_windows(0.2)), 1)
        with self.assertRaises(ValueError):
            plan_windows(10, overlap_sec=12)
        with self.assertRaises(ValueError):
            plan_windows(float("nan"))
        with self.assertRaises(ValueError):
            plan_windows(10, start_sec=8, end_sec=7)

    def test_model_indices_ground_to_actual_time(self):
        raw = json.dumps({"operation_context": "synthetic", "visibility_limits": [],
                          "observations": [{"start_frame": 0, "end_frame": 1, "description": "A color changes"}]})
        parsed = parse_observations(raw, [0.13, 0.91])
        self.assertEqual(parsed["observations"][0]["end_sec"], 0.91)
        self.assertEqual(parse_observations("```json\n" + raw + "\n```", [0.13, 0.91]), parsed)

    def test_invalid_json_indices_and_types_fail_without_echo(self):
        payload = {"operation_context": "synthetic", "visibility_limits": [],
                   "observations": [{"start_frame": 0, "end_frame": 99, "description": "private-content"}]}
        for end in (99, True, -1, "1"):
            payload["observations"][0]["end_frame"] = end
            with self.assertRaises(ModelOutputError) as error:
                parse_observations(json.dumps(payload), [0, 1])
            self.assertNotIn("private-content", str(error.exception))
        with self.assertRaises(ModelOutputError):
            parse_observations('{"incomplete":', [0])


class SyntheticVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "synthetic-vfr.mov"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc2=size=96x64:rate=10:duration=1", "-vf", r"setpts=PTS+gte(N\,4)*0.4/TB",
            "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", str(cls.video)
        ], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_nearest_actual_pts_not_nominal_frame_number(self):
        info = probe_video(self.video)
        result = sample_frames(self.video, [0, 0.32, 0.65, info.duration_sec], info)
        self.assertAlmostEqual(result[0].timestamp_sec, 0, places=5)
        self.assertAlmostEqual(result[0.32].timestamp_sec, 0.3, places=5)
        self.assertAlmostEqual(result[0.65].timestamp_sec, 0.8, places=5)
        self.assertLess(result[info.duration_sec].timestamp_sec, info.duration_sec)
        self.assertEqual(result[0].image.size, (96, 64))

    def test_source_jpeg_evidence_and_bounds(self):
        from PIL import Image
        frames = extract_evidence(self.video, [0.32, 0.65, 0.66])
        self.assertEqual(len(frames), 2)  # Two requests resolve to the same real frame.
        self.assertAlmostEqual(frames[1]["timestamp_sec"], 0.8, places=5)
        picture = Image.open(io.BytesIO(base64.b64decode(frames[0]["image_base64"], validate=True)))
        self.assertEqual(picture.format, "JPEG")
        self.assertEqual(picture.size, (96, 64))
        with self.assertRaises(ValueError):
            extract_evidence(self.video, [100])
        with self.assertRaises(ValueError):
            extract_evidence(self.video, [])

    def test_display_rotation_and_downscale(self):
        rotated = Path(self.temp.name) / "rotated.mov"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-display_rotation:v:0", "90", "-i", str(self.video),
                        "-c", "copy", str(rotated)],
                       check=True, capture_output=True, timeout=30)
        info = probe_video(rotated)
        self.assertEqual(info.rotation, 90)
        frame = sample_frames(rotated, [0], info)[0]
        self.assertEqual(frame.image.size, (64, 96))
        small = sample_frames(rotated, [0], info, max_long_edge=48)[0]
        self.assertEqual(small.image.size, (32, 48))

    def test_complete_observations_do_not_claim_a_verdict(self):
        class SyntheticObserver:
            def observe(self, frames, previous_context, deadline):
                return {"operation_context": "synthetic colors", "visibility_limits": [], "observations": []}

        analyzer = LocalVideoAnalyzer(SyntheticObserver(), window_sec=0.7, overlap_sec=0.2, frames_per_window=3)
        result = analyzer.analyze(self.video, "synthetic", timeout_sec=20)
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage_scope"], "full_video")
        self.assertNotIn("verdict", result)
        self.assertEqual(result["windows"][-1]["window_end_sec"], result["video"]["duration_sec"])
        refined = analyzer.analyze(self.video, "synthetic", start_sec=0.2, end_sec=0.8)
        self.assertEqual(refined["coverage_scope"], "interval")

    def test_decode_deadline_and_bad_input_fail(self):
        info = probe_video(self.video)
        with self.assertRaises(DeadlineExceeded):
            sample_frames(self.video, [0.5], info, deadline=time.monotonic() - 1)
        bad = Path(self.temp.name) / "invalid.mov"
        bad.write_bytes(b"not a video")
        with self.assertRaises(VideoError):
            probe_video(bad)

    def test_batch_evidence_decodes_once_and_keeps_event_mapping(self):
        with patch("app.evidence.frames.sample_frames", wraps=sample_frames) as sampling:
            result = extract_evidence_batch(self.video, {"a": [0.32, 0.65, 0.66], "b": [0.65]})
        self.assertEqual(sampling.call_count, 1)
        self.assertEqual(len(result["a"]), 2)
        self.assertEqual(result["a"][1], result["b"][0])
        self.assertEqual(result["b"][0]["timestamp_sec"], 0.8)
        with self.assertRaises(DeadlineExceeded):
            extract_evidence_batch(self.video, {"a": [0]}, deadline=time.monotonic() - 1)
        with self.assertRaises(ValueError):
            extract_evidence_batch(self.video, {"a": [0]}, deadline=float("nan"))

    def test_refinements_share_deadline_and_reuse_identical_interval(self):
        class Observer:
            calls = 0
            deadlines = []

            def observe(self, frames, previous_context, deadline):
                self.calls += 1
                self.deadlines.append(deadline)
                return {"operation_context": "synthetic", "observations": [], "visibility_limits": []}

        observer = Observer()
        analyzer = LocalVideoAnalyzer(observer)
        deadline = time.monotonic() + 30
        intervals = [{"candidate_id": "a", "start_sec": 0.1, "end_sec": 1.0},
                     {"candidate_id": "b", "start_sec": 0.1, "end_sec": 1.0},
                     {"candidate_id": "c", "start_sec": 0.3, "end_sec": 1.2}]
        result = analyzer.refine_intervals(self.video, "synthetic", intervals, deadline=deadline)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(observer.deadlines, [deadline, deadline])
        self.assertEqual([item["candidate_id"] for item in result["refinements"]], ["a", "b", "c"])
        self.assertEqual(result["refinements"][0]["report"]["coverage_scope"], "interval")
        self.assertNotIn("verdict", result)
        self.assertEqual(result["metrics"]["unique_intervals"], 2)
        before = observer.calls
        with self.assertRaises(ValueError):
            analyzer.refine_intervals(self.video, "synthetic", intervals + [
                {"candidate_id": "invalid", "start_sec": 0, "end_sec": 100}])
        self.assertEqual(observer.calls, before)
        with self.assertRaises(DeadlineExceeded):
            analyzer.analyze(self.video, "synthetic", deadline=time.monotonic() - 1)
        with self.assertRaises(ValueError):
            analyzer.analyze(self.video, "synthetic", deadline=float("nan"))


if __name__ == "__main__":
    unittest.main()
