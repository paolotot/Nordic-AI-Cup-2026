"""One request in, one response out, with state carried across the sequence."""

import logging
import os
import time

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


class Pipeline:
    def __init__(self):
        self.detector_name = os.environ.get('FLYBY_DETECTOR', 'yolo')
        self.camera_name = os.environ.get('FLYBY_CAMERA', 'look-and-zoom')
        self.detector = load_detector(self.detector_name, os.environ.get('FLYBY_WEIGHTS', ''))
        self.detector.warm_up()
        self.sequence_id = None
        logger.info('pipeline ready: detector=%s camera=%s', self.detector_name, self.camera_name)

    def _reset(self, sequence_id):
        self.sequence_id = sequence_id
        self.tracker = Tracker()
        self.policy = load_policy(self.camera_name)

    def predict(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        if request.sequence_id != self.sequence_id:
            self._reset(request.sequence_id)
        if request.camera_command_feedback is not None:
            logger.warning('camera command ignored: %s', request.camera_command_feedback.reason)

        started = time.perf_counter()
        view = request.view
        region = tuple(view.source_region_xyxy)
        annotations, command = [], None
        try:
            self.tracker.advance_to(request.frame)
            detections = self.detector(decode_view(view), region, view.resolution_level,
                                       request.frame)
            self.tracker.update(detections, region, view.resolution_level, request.frame)
            annotations = self._annotations(request)
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
        return out[:500]
