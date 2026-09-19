"""Mark likely real objects in recorded validation views before they become backgrounds.

Recorded views are full of boats, cars and roofs we want the detector to learn
as background, but they also hold real (unlabelled) objects. Every spot the
current detector is confident about is written to <stem>.mask.json and painted
over by make_dataset.py, so those objects are not taught as "nothing here".
Hand labels (<stem>.labels.json, see tools/label_recordings.py) take priority.

    python tools/mask_recordings.py --weights models/kaggle_v2/yolo11n_960/best_openvino_model
"""

import argparse
import glob
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--recordings', default=str(ROOT / 'data' / 'recordings' / 'validation'))
    arguments = parser.parse_args()

    from ultralytics import YOLO
    device = 'intel:cpu' if 'openvino' in arguments.weights else 'cpu'
    model = YOLO(arguments.weights, task='detect')
    total, masked = 0, 0
    for path in sorted(glob.glob(str(Path(arguments.recordings) / '*' / '*_f*.json'))):
        if path.endswith(('.mask.json', '.labels.json')):
            continue
        image = cv2.imread(path.replace('.json', '.png'))
        result = model.predict(image, imgsz=(544, 960), conf=arguments.confidence,
                               device=device, verbose=False, agnostic_nms=True)[0]
        boxes = [[round(float(c), 1) for c in box] for box in result.boxes.xyxy.numpy()]
        Path(path.replace('.json', '.mask.json')).write_text(json.dumps(boxes))
        total += 1
        masked += len(boxes)
    print(f'{total} views, {masked} confident spots masked')


if __name__ == '__main__':
    main()
