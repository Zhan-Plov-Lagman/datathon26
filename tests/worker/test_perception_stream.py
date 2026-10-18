import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from app.perception.stream import iter_batches
from app.perception.detectors import Detection, normalize_box
from app.video.sampling import DeadlineExceeded, probe_video, sample_frames


class StreamingTests(unittest.TestCase):
    def test_timestamps_batch_limit_and_source_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'synthetic.mov'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=size=96x64:rate=7:duration=2','-c:v','libx264',str(path)],
                check=True, capture_output=True)
            batches = list(iter_batches(path, fps=3, batch_size=2, max_long_edge=64))
            self.assertTrue(all(1 <= len(b) <= 2 for b in batches))
            frames = [f for batch in batches for f in batch]
            times = [f.timestamp_sec for f in frames]
            self.assertEqual(times, sorted(set(times)))
            self.assertEqual(times[0],0)
            self.assertAlmostEqual(times[-1],13/7,places=6)
            self.assertTrue(all(max(f.image.size)<=64 for f in frames))
            # Round-trip exact decoded PTS through the original evidence frame source.
            selected=sample_frames(path,times,probe_video(path))
            self.assertEqual([selected[t].timestamp_sec for t in times],times)
            with self.assertRaises(DeadlineExceeded):
                list(iter_batches(path,deadline=time.monotonic()-1))

    def test_invalid_configuration_and_boxes_fail(self):
        for kwargs in ({'fps':float('nan')},{'fps':0},{'batch_size':0},{'max_long_edge':4}):
            with self.assertRaises(ValueError):
                list(iter_batches('unused.mov', **kwargs))
        self.assertEqual(normalize_box((-10,10,110,40),(100,50)),(0,0.2,1,0.8))
        self.assertIsNone(normalize_box((120,10,150,20),(100,50)))
        with self.assertRaises(ValueError):
            Detection('person',float('nan'),(0,0,1,1))
        with self.assertRaises(ValueError):
            Detection('person',.5,(0,0,1,1),((0,0,float('inf')),))
