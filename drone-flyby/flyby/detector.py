"""Detectors: turn one transmitted view into detections in source pixels.

* ``YoloDetector``   - the real one. Loads a trained .pt or an OpenVINO export.
* ``OracleDetector`` - a stand-in built from the helsinki annotations, so the
  tracker and camera logic can be tested end to end before a model exists. It
  only "recognises" objects that look at least ``recognise_px`` big in the view
  and reports smaller ones as unlabelled specks. Useless outside helsinki.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES, TRANSMITTED_VIEW_SIZE

EDGE_PX = 3   # a box this close to a view edge (inside the frame) may be cut off


@dataclass
class Detection:
    class_id: int            # index into OBJECT_CLASSES
    score: float
    box: np.ndarray          # x1, y1, x2, y2 in source pixels
    level: int               # resolution level it was seen at
    cut_off: bool            # touches a view edge that is not a frame edge

    @property
    def centre(self):
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def apparent_px(self):
        """Size of the object in the transmitted image, sqrt(w*h)."""
        factor = 4 / 2 ** self.level
        return float(np.sqrt(max(0.0, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])))
                     / factor)


def is_cut_off(box, region):
    x1, y1, x2, y2 = region
    factor = (x2 - x1) / TRANSMITTED_VIEW_SIZE[0]
    margin = EDGE_PX * factor
    return bool((box[0] - x1 < margin and x1 > 0) or (x2 - box[2] < margin and x2 < IMAGE_WIDTH)
                or (box[1] - y1 < margin and y1 > 0) or (y2 - box[3] < margin and y2 < IMAGE_HEIGHT))


class YoloDetector:
    def __init__(self, weights: str, confidence: float = 0.05):
        from ultralytics import YOLO
        self.model = YOLO(weights, task='detect')
        self.confidence = confidence
        # For OpenVINO exports, plain 'cpu' lets OpenVINO pick 'AUTO', which
        # tries the Iris Xe iGPU and hangs compiling there. Pin the CPU.
        self.device = 'intel:cpu' if 'openvino' in str(weights) else 'cpu'

    def warm_up(self):
        blank = np.zeros((TRANSMITTED_VIEW_SIZE[1], TRANSMITTED_VIEW_SIZE[0], 3), np.uint8)
        for _ in range(3):
            self(blank, (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0)

    def __call__(self, image, region, level, frame) -> List[Detection]:
        result = self.model.predict(image, imgsz=(544, 960), conf=self.confidence, iou=0.5,
                                    agnostic_nms=True, max_det=200, device=self.device,
                                    verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        factor = (region[2] - region[0]) / TRANSMITTED_VIEW_SIZE[0]
        offset = np.array([region[0], region[1], region[0], region[1]], float)
        source = boxes.xyxy.cpu().numpy() * factor + offset
        detections = []
        for box, class_id, score in zip(source, boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            detections.append(Detection(int(class_id), float(score), box, level,
                                        is_cut_off(box, region)))
        return detections


class OracleDetector:
    def __init__(self, recognise_px: float = 20, notice_px: float = 8, seed: int = 0):
        self.recognise_px = recognise_px
        self.notice_px = notice_px
        self.rng = np.random.default_rng(seed)

    def warm_up(self):
        pass

    def __call__(self, image, region, level, frame) -> List[Detection]:
        from utils import load_annotations
        factor = 4 / 2 ** level
        detections = []
        for a in load_annotations(frame):
            box = np.array(a['bbox'], float)
            inter = [max(box[0], region[0]), max(box[1], region[1]),
                     min(box[2], region[2]), min(box[3], region[3])]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if inter[2] <= inter[0] or inter[3] <= inter[1]:
                continue
            if (inter[2] - inter[0]) * (inter[3] - inter[1]) < 0.6 * area:
                continue
            apparent = np.sqrt(area) / factor
            jitter = self.rng.normal(0, factor * 0.5, 4)     # half a view pixel
            seen = np.array(inter) + jitter
            if apparent >= self.recognise_px:
                class_id, score = OBJECT_CLASSES.index(a['object_id']), 0.85
            elif apparent >= self.notice_px:
                class_id, score = int(self.rng.integers(len(OBJECT_CLASSES))), 0.1
            else:
                continue
            detections.append(Detection(class_id, score, seen, level, is_cut_off(seen, region)))
        return detections


def load_detector(name: str, weights: str = ''):
    if name == 'oracle':
        return OracleDetector()
    if not weights:
        candidates = sorted(Path(__file__).resolve().parents[1].glob('models/detector*'))
        if not candidates:
            raise FileNotFoundError('no detector weights in models/ (set FLYBY_WEIGHTS)')
        weights = str(candidates[0])
    return YoloDetector(weights)
