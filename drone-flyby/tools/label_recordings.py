"""Hand-label real objects in recorded validation views.

A labelled view is treated as complete by make_dataset.py: everything in it
that is not labelled is taught as background (boats, cars, roofs). So label
every real object in a view you touch, or leave the view alone.

    # 1. draw numbered detector candidates + a grid on some views
    python tools/label_recordings.py render --level 2 --every 4
    #    -> outputs/label/<sequence>_<stem>.jpg  and outputs/label/candidates.json

    # 2. write labels from a spec file
    python tools/label_recordings.py save spec.json

spec.json maps "<sequence>/<stem>" to
    {"keep": {"<candidate index>": "<class>"}, "add": [["<class>", x1, y1, x2, y2]],
     "ignore": [<candidate index>, ...], "ignore_boxes": [[x1, y1, x2, y2], ...]}
with boxes in the 960x540 view's pixels. Ignored spots are painted over (not
an object, not background): use them wherever the class is uncertain. An
entry with nothing kept or added marks a view as containing no objects.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES  # noqa: E402

RECORDINGS = ROOT / 'data' / 'recordings' / 'validation'
OUT = ROOT / 'outputs' / 'label'
WEIGHTS = ROOT / 'models' / 'kaggle_v2' / 'yolo11n_960' / 'best_openvino_model'


def views(level, every, sequence=None):
    """Recorded views at a level, thinned to every n-th, from the largest run."""
    runs = sorted((p for p in RECORDINGS.glob('*') if p.is_dir()),
                  key=lambda p: -len(list(p.glob('*.png'))))
    run = next((r for r in runs if r.name.startswith(sequence)), None) if sequence else runs[0]
    picked = []
    for meta in sorted(glob.glob(str(run / '*_f*.json'))):
        if meta.endswith(('.mask.json', '.labels.json', '.ignore.json')):
            continue
        if json.load(open(meta))['view']['resolution_level'] == level:
            picked.append(Path(meta))
    return picked[::every]


def render(arguments):
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS), task='detect')
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = {}
    for meta in views(arguments.level, arguments.every, arguments.sequence):
        key = f'{meta.parent.name}/{meta.stem}'
        image = cv2.imread(str(meta.with_suffix('.png')))
        result = model.predict(image, imgsz=(544, 960), conf=arguments.confidence,
                               device='intel:cpu', verbose=False, agnostic_nms=True)[0]
        canvas = image.copy()
        for x in range(0, 960, 100):
            cv2.line(canvas, (x, 0), (x, 539), (255, 255, 255), 1)
            cv2.putText(canvas, str(x), (x + 2, 10), 0, 0.3, (255, 255, 255), 1)
        for y in range(0, 540, 100):
            cv2.line(canvas, (0, y), (959, y), (255, 255, 255), 1)
            cv2.putText(canvas, str(y), (2, y + 10), 0, 0.3, (255, 255, 255), 1)
        candidates = []
        for index, (box, cls, conf) in enumerate(zip(result.boxes.xyxy.numpy(),
                                                     result.boxes.cls.numpy(),
                                                     result.boxes.conf.numpy())):
            x1, y1, x2, y2 = (int(round(c)) for c in box)
            candidates.append({'box': [x1, y1, x2, y2], 'yolo': OBJECT_CLASSES[int(cls)],
                               'conf': round(float(conf), 2)})
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(canvas, str(index), (x1, max(9, y1 - 2)), 0, 0.4, (0, 0, 0), 3)
            cv2.putText(canvas, str(index), (x1, max(9, y1 - 2)), 0, 0.4, (0, 255, 255), 1)
        catalogue[key] = candidates
        cv2.imwrite(str(OUT / f"{key.replace('/', '_')}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    path = OUT / f'candidates_L{arguments.level}.json'
    path.write_text(json.dumps(catalogue, indent=1))
    print(f'{len(catalogue)} views rendered to {OUT}; candidates in {path.name}')


def save(arguments):
    spec = json.loads(Path(arguments.spec).read_text())
    catalogues = {}
    for path in OUT.glob('candidates_L*.json'):
        catalogues.update(json.loads(path.read_text()))
    written = 0
    for key, entry in spec.items():
        labels = []
        for index, name in entry.get('keep', {}).items():
            labels.append({'object_id': name, 'bbox': catalogues[key][int(index)]['box']})
        for name, x1, y1, x2, y2 in entry.get('add', []):
            labels.append({'object_id': name, 'bbox': [x1, y1, x2, y2]})
        for label in labels:
            if label['object_id'] not in OBJECT_CLASSES:
                raise SystemExit(f'{key}: unknown class {label["object_id"]}')
        # Unsure spots: painted over, so taught neither as object nor as background.
        ignore = [catalogues[key][int(index)]['box'] for index in entry.get('ignore', [])]
        ignore += [list(box) for box in entry.get('ignore_boxes', [])]
        target = RECORDINGS / (key + '.labels.json')
        target.write_text(json.dumps(labels))
        (RECORDINGS / (key + '.ignore.json')).write_text(json.dumps(ignore))
        written += 1
    print(f'wrote labels for {written} views')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('render')
    r.add_argument('--level', type=int, default=2)
    r.add_argument('--every', type=int, default=4)
    r.add_argument('--sequence', default=None)
    r.add_argument('--confidence', type=float, default=0.05)
    s = sub.add_parser('save')
    s.add_argument('spec')
    arguments = parser.parse_args()
    render(arguments) if arguments.command == 'render' else save(arguments)


if __name__ == '__main__':
    main()
