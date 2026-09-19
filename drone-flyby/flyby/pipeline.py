"""One request in, one response out, with state carried across the sequence."""

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from flyby.camera import load_policy
from flyby.detector import load_detector
from flyby.tracker import Tracker
from utils import clip_bbox_to_frame, decode_view, source_bbox_to_global

logger = logging.getLogger(__name__)

# A frame holds ~10-20 real objects; beyond this, extra boxes are mostly noise.
MAX_ANSWERS = 60


class Pipeline:
    def __init__(self):
        self.detector_name = os.environ.get('FLYBY_DETECTOR', 'yolo')
        self.camera_name = os.environ.get('FLYBY_CAMERA', 'look-and-zoom')
        self.detector = load_detector(self.detector_name, os.environ.get('FLYBY_WEIGHTS', ''))
        self.detector.warm_up()
        self.sequence_id = None
        self.lock = threading.Lock()
        # FLYBY_RECORD=<dir>: keep every received view (allowed for validation
        # runs) for retraining. Written on a worker thread, off the answer path.
        record = os.environ.get('FLYBY_RECORD', '')
        self.recorder = ThreadPoolExecutor(max_workers=1) if record else None
        self.record_dir = Path(record) if record else None
        logger.info('pipeline ready: detector=%s camera=%s', self.detector_name, self.camera_name)

    def _reset(self, sequence_id):
        self.sequence_id = sequence_id
        self.tracker = Tracker()
        self.policy = load_policy(self.camera_name)

    def predict(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        # The evaluator does not wait for our answer before sending the next
        # frame, so requests can overlap; the tracker is shared state.
        with self.lock:
            return self._predict(request)

    def _predict(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        if request.sequence_id != self.sequence_id:
            self._reset(request.sequence_id)
        if request.camera_command_feedback is not None:
            logger.warning('camera command ignored: %s', request.camera_command_feedback.reason)

        started = time.perf_counter()
        view = request.view
        if self.recorder is not None:
            self.recorder.submit(self._record, request)
        region = tuple(view.source_region_xyxy)
        annotations, command = [], None
        try:
            stale = self.tracker.frame is not None and request.frame < self.tracker.frame
            self.tracker.advance_to(request.frame)
            detections = self.detector(decode_view(view), region, view.resolution_level,
                                       request.frame)
            self.tracker.update(detections, region, view.resolution_level, request.frame)
            annotations = self._annotations(request)
            if stale:
                # A newer frame already moved the camera; a command computed
                # from this older view would be refused or undo that move.
                logger.info('late request for frame %s (at %s): no camera command',
                            request.frame, self.tracker.frame)
            else:
                command = self.policy.next_view(request, self.tracker, request.frame)
        except Exception:
            # An exception would lose the frame; an empty answer still scores it.
            logger.exception('pipeline failed on frame %s', request.frame)
            try:
                annotations = self._annotations(request)
            except Exception:
                annotations = []

        logger.debug('frame %s: %d tracks, %.0f ms', request.frame, len(self.tracker.tracks),
                     (time.perf_counter() - started) * 1000)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id,
            frame=request.frame,
            annotations=annotations,
            requested_view=RequestedViewDto(resolution_level=command[0], center_x=command[1],
                                            center_y=command[2]) if command else None,
        )

    def _record(self, request):
        try:
            folder = self.record_dir / request.sequence_id.replace(':', '_')
            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{request.frame_index:04d}_f{request.frame}'
            (folder / f'{stem}.png').write_bytes(base64.b64decode(request.view.image))
            meta = request.model_dump(exclude={'view': {'image'}})
            (folder / f'{stem}.json').write_text(json.dumps(meta))
        except Exception:
            logger.exception('recording frame %s failed', request.frame)

    def _annotations(self, request):
        out = []
        for name, box, confidence in self.tracker.annotations(request.frame):
            bbox = clip_bbox_to_frame(source_bbox_to_global(box, request.original_width,
                                                            request.original_height))
            if bbox is None:
                continue
            out.append(DroneFlybyPredictionDto(object_id=name, bbox=[round(c, 6) for c in bbox],
                                               confidence=round(confidence, 4)))
        out.sort(key=lambda a: -a.confidence)
        return out[:MAX_ANSWERS]
