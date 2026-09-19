"""Score detectors on hand-labelled validation views (the only real test set we have).

Every 5th labelled view (sorted by name) is held out from training; the list is
written to data/recordings/validation/holdout.json the first time and reused,
and make_dataset.py leaves those views out. Scoring is COCO mAP@0.50 per view,
single-frame (no tracking, no camera), with detections centred inside "unsure"
areas dropped.

    python tools/eval_views.py models/kaggle_v3/yolo11s_960/best_openvino_model [more weights...]
    python tools/eval_views.py --all ...     # every labelled view (includes training views)
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES  # noqa: E402

RECORDINGS = ROOT / 'data' / 'recordings' / 'validation'
HOLDOUT = RECORDINGS / 'holdout.json'


def labelled_views():
    return sorted(str(p.relative_to(RECORDINGS)).replace('\\', '/').replace('.labels.json', '')
                  for p in RECORDINGS.glob('*/*.labels.json'))


def holdout():
    if HOLDOUT.exists():
        return set(json.loads(HOLDOUT.read_text()))
    keys = labelled_views()[::5]
    HOLDOUT.write_text(json.dumps(keys, indent=1))
    return set(keys)


def score(weights, keys):
    from faster_coco_eval import COCO, COCOeval_faster
    from ultralytics import YOLO
    device = 'intel:cpu' if 'openvino' in str(weights) else 'cpu'
    model = YOLO(str(weights), task='detect')
    images, annotations, detections = [], [], []
    for image_id, key in enumerate(keys, start=1):
        image = cv2.imread(str(RECORDINGS / f'{key}.png'))
        images.append({'id': image_id, 'width': 960, 'height': 540})
        for a in json.loads((RECORDINGS / f'{key}.labels.json').read_text()):
            x1, y1, x2, y2 = a['bbox']
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id,
                                'category_id': OBJECT_CLASSES.index(a['object_id']) + 1,
                                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1),
                                'iscrowd': 0})
        ignore_path = RECORDINGS / f'{key}.ignore.json'
        ignore = json.loads(ignore_path.read_text()) if ignore_path.exists() else []
        result = model.predict(image, imgsz=(544, 960), conf=0.01, device=device, verbose=False,
                               agnostic_nms=True, max_det=200)[0]
        for box, cls, conf in zip(result.boxes.xyxy.numpy(), result.boxes.cls.numpy(),
                                  result.boxes.conf.numpy()):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in ignore):
                continue
            detections.append({'image_id': image_id, 'category_id': int(cls) + 1,
                               'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]),
                                        float(box[3] - box[1])], 'score': float(conf)})
    present = sorted({a['category_id'] for a in annotations})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO({'images': images, 'annotations': annotations,
                   'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(OBJECT_CLASSES)]})
        ev = COCOeval_faster(gt, gt.loadRes(detections), 'bbox')
        ev.params.catIds = present
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
    per_class = {}
    for index, category in enumerate(present):
        p = ev.eval['precision'][0, :, index, 0, -1]
        p = p[p > -1]
        per_class[OBJECT_CLASSES[category - 1]] = float(p.mean()) if p.size else 0.0
    return float(np.mean(list(per_class.values()))), per_class, len(annotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('weights', nargs='+')
    parser.add_argument('--all', action='store_true')
    arguments = parser.parse_args()
    keys = labelled_views() if arguments.all else sorted(holdout())
    print(f'{len(keys)} views')
    for weights in arguments.weights:
        total, per_class, objects = score(weights, keys)
        print(f'\n{weights}\n  mAP@0.50 {total:.3f} over {len(per_class)} classes, {objects} objects')
        print('  ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])))


if __name__ == '__main__':
    main()
