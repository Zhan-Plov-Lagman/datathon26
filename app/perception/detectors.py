"""Adapted detector calls from the two team projects; offline and common RGB/PTS contract.

Detector weights are loaded only from explicit local paths. No label lookups,
role assumptions, complete-video image lists or public verdicts live here.
"""
from dataclasses import dataclass
from pathlib import Path
import math

from app.video.sampling import check_deadline


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # normalized display coordinates
    keypoints: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self):
        if not self.label or not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('Invalid detection confidence')
        if len(self.box) != 4 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.box):
            raise ValueError('Invalid normalized box')
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError('Empty detection box')
        for point in self.keypoints:
            if len(point) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in point):
                raise ValueError('Invalid pose point')


def normalize_box(box, size):
    width, height = size
    values = [float(x) for x in box]
    if len(values) != 4 or any(not math.isfinite(x) for x in values):
        raise ValueError('Invalid model coordinates')
    # Detection regressors may extend outside an image. Intersection, not a role assertion.
    b = tuple(min(1., max(0., x / dim)) for x, dim in zip(values, (width, height, width, height)))
    return b if b[0] < b[2] and b[1] < b[3] else None


class RTDetr:
    def __init__(self, path, threshold=.3, device='cuda:0'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        path = Path(path).resolve(strict=True)
        self.torch, self.device, self.threshold = torch, device, threshold
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForObjectDetection.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device).eval()

    def __call__(self, frames, deadline=None):
        check_deadline(deadline)
        if not frames:
            return []
        images = [f.image for f in frames]
        with self.torch.inference_mode():
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            output = self.model(**inputs)
            results = self.processor.post_process_object_detection(output, threshold=self.threshold,
                target_sizes=[image.size[::-1] for image in images])
        check_deadline(deadline)
        batches = []
        for image, result in zip(images, results, strict=True):
            detections = []
            for score, label, box in zip(result['scores'], result['labels'], result['boxes'], strict=True):
                normalized = normalize_box(box.tolist(), image.size)
                if normalized is not None:
                    detections.append(Detection(self.model.config.id2label[int(label)], float(score), normalized))
            batches.append(detections)
        return batches


class YoloPose:
    def __init__(self, path, threshold=.3, device=0):
        import os
        os.environ['YOLO_OFFLINE'] = 'True'
        os.environ['YOLO_AUTOINSTALL'] = 'False'
        os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[2]/'.cache/yolo'))
        from ultralytics import YOLO
        from ultralytics import settings
        settings.update({'sync': False})
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Explicit local YOLO weights required')
        self.model = YOLO(str(path), task='pose')
        self.threshold, self.device = threshold, device

    def __call__(self, frames, deadline=None):
        check_deadline(deadline)
        if not frames:
            return []
        # PIL images are RGB; Ultralytics handles the conversion. Do not feed RGB ndarray as BGR.
        results = self.model.predict([f.image for f in frames], conf=self.threshold,
                                     device=self.device, verbose=False, save=False, stream=False)
        check_deadline(deadline)
        batches = []
        for frame, result in zip(frames, results, strict=True):
            detections = []
            for i, box in enumerate(result.boxes):
                normalized = normalize_box(box.xyxy[0].tolist(), frame.image.size)
                if normalized is None:
                    continue
                points = ()
                if result.keypoints is not None:
                    xy = result.keypoints.xyn[i].tolist()
                    conf = result.keypoints.conf[i].tolist()
                    if any(not math.isfinite(float(v)) for point in xy for v in point) or any(not math.isfinite(float(c)) for c in conf):
                        raise ValueError('Non-finite model keypoint')
                    points = tuple((min(1.,max(0.,float(x))), min(1.,max(0.,float(y))), float(c))
                                   for (x,y),c in zip(xy,conf,strict=True))
                detections.append(Detection('person', float(box.conf[0]), normalized, points))
            batches.append(detections)
        return batches


class OwlObjects:
    QUERIES = ('banknotes', 'a stack of paper money', 'a handbag', 'a coat', 'a mobile phone',
               'an office desk', 'a banknote counting machine', 'a metal safe', 'an open cabinet door')

    def __init__(self, path, queries=None, threshold=.1, device='cuda:0'):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        path = Path(path).resolve(strict=True)
        self.torch, self.device, self.threshold = torch, device, threshold
        self.queries = tuple(queries or self.QUERIES)
        self.processor = Owlv2Processor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = Owlv2ForObjectDetection.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device).eval()

    def __call__(self, frames, deadline=None):
        batches = []
        for frame in frames:
            check_deadline(deadline)
            with self.torch.inference_mode():
                inputs = self.processor(text=[list(self.queries)], images=frame.image, return_tensors='pt').to(self.device)
                output = self.model(**inputs)
                side = max(frame.image.size)  # OWLv2 pads to square before resize.
                result = self.processor.post_process_grounded_object_detection(output, threshold=self.threshold,
                    target_sizes=self.torch.tensor([[side,side]], device=self.device))[0]
            check_deadline(deadline)
            detections = []
            for score, label, box in zip(result['scores'], result['labels'], result['boxes'], strict=True):
                normalized = normalize_box(box.tolist(), frame.image.size)
                if normalized is not None:
                    detections.append(Detection(self.queries[int(label)], float(score), normalized))
            batches.append(detections)
        return batches
