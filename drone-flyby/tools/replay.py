"""Replay helsinki through the pipeline in-process (no HTTP) and explain the score.

Same crops, camera rules and scorer as local_evaluator.py, every frame answered.
Prints the camera path, what each frame's answer got right, and AP per class.

    python tools/replay.py                       # FLYBY_* env vars pick detector/camera
    python tools/replay.py --detector oracle --camera tour
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--detector', default=None)
    parser.add_argument('--camera', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--quiet', action='store_true')
    arguments = parser.parse_args()
    for key, value in (('FLYBY_DETECTOR', arguments.detector), ('FLYBY_CAMERA', arguments.camera),
                       ('FLYBY_WEIGHTS', arguments.weights)):
        if value:
            os.environ[key] = value

    from dtos import DroneFlybyPredictRequestDto
    from flyby.pipeline import Pipeline
    from flyby.tracker import iou
    from local_evaluator import Camera, CameraRejection, build_request, render_view, score
    from utils import frame_numbers, global_bbox_to_source, load_annotations, load_frame

    pipeline = Pipeline()
    camera = Camera()
    predictions = {}
    for index, frame in enumerate(frame_numbers()):
        payload = build_request(frame, index, camera, render_view(load_frame(frame), camera), None)
        response = pipeline.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        boxes = [(a.object_id, global_bbox_to_source(a.bbox), a.confidence)
                 for a in response.annotations]
        predictions[frame] = [{'object_id': n, 'bbox': b, 'confidence': c} for n, b, c in boxes]

        truth = load_annotations(frame)
        hits, wrong_class, misplaced = 0, [], []
        for a in truth:
            best = max(((iou(np.array(b), np.array(a['bbox'])), n) for n, b, _ in boxes),
                       default=(0, None))
            same = [iou(np.array(b), np.array(a['bbox'])) for n, b, _ in boxes if n == a['object_id']]
            if same and max(same) >= 0.5:
                hits += 1
            elif best[0] >= 0.5:
                wrong_class.append(f"{a['object_id']}->{best[1]}")
            elif same:
                misplaced.append(f"{a['object_id']}({max(same):.2f})")
        if not arguments.quiet:
            view = payload['view']
            command = response.requested_view
            print(f"f{frame:2d} L{view['resolution_level']} ({view['center_x']:4d},{view['center_y']:4d})"
                  f" tracks {len(pipeline.tracker.tracks):2d} answers {len(boxes):2d}"
                  f" | correct {hits:2d}/{len(truth):2d}"
                  + (f" wrong-class {wrong_class}" if wrong_class else '')
                  + (f" misplaced {misplaced}" if misplaced else '')
                  + (f" -> L{command.resolution_level}" if command else ''))
        if response.requested_view is not None:
            v = response.requested_view
            try:
                camera.apply(v.resolution_level, v.center_x, v.center_y)
            except CameraRejection as exc:
                print(f'  camera refused: {exc}')

    total, per_class = score('helsinki', predictions)
    print('\nAP by class: ' + ', '.join(f'{k} {v:.2f}' for k, v in
                                         sorted(per_class.items(), key=lambda kv: -kv[1])))
    print(f'mAP@0.50: {total:.3f}')


if __name__ == '__main__':
    main()
