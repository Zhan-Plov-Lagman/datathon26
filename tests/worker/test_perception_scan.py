from types import SimpleNamespace
from unittest.mock import patch
import unittest

from app.perception.scan import scan_video
from app.video.sampling import SampledFrame


class ScanTests(unittest.TestCase):
    def test_empty_detections_are_completed_search_not_skipped_stage(self):
        frames=[SampledFrame(0,None),SampledFrame(.5,None),SampledFrame(1.,None)]
        detector=lambda batch,deadline:[[] for _ in batch]
        with patch('app.perception.scan.probe_video',return_value=SimpleNamespace(duration_sec=1.1)), \
             patch('app.perception.scan.iter_batches',return_value=iter([frames])):
            result=scan_video('synthetic',detector,detector,job_id='synthetic')
        self.assertEqual(result['metrics']['people_frames'],3)
        self.assertEqual(result['metrics']['object_frames'],1)
        self.assertTrue(result['review_windows'])
        self.assertTrue(result['complete'])

    def test_missing_detector_outputs_fail_the_whole_scan(self):
        with patch('app.perception.scan.probe_video',return_value=SimpleNamespace(duration_sec=1.1)), \
             patch('app.perception.scan.iter_batches',return_value=iter([[SampledFrame(0,None)]])):
            with self.assertRaises(ValueError):
                scan_video('synthetic',lambda batch,deadline:[],lambda batch,deadline:[],job_id='synthetic')
