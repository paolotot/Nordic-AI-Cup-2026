"""Show every class as it looks at resolution levels 0, 1 and 2.

For each class, take the frame where its box is largest (least clipped), cut a
fixed window around it and downsample the window the way the evaluator does
(INTER_AREA by 4, 2 and 1). Every tile is then blown up to the same display
size with nearest-neighbour, so you see the real pixels each level delivers.

    python tools/object_sheet.py        # writes outputs/object_sheet.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES  # noqa: E402
from utils import DEFAULT_SCENE, frame_numbers, load_annotations, load_frame  # noqa: E402

WINDOW = 256    # source pixels around the object
TILE = 256      # display size of each tile
LABEL_WIDTH = 170
OUTPUT = Path(__file__).resolve().parents[1] / 'outputs' / 'object_sheet.png'


def best_sighting(scene):
    """For each class, the (frame, bbox) with the largest box area."""
    best = {}
    for frame in frame_numbers(scene):
        for annotation in load_annotations(frame, scene):
            x1, y1, x2, y2 = annotation['bbox']
            area = (x2 - x1) * (y2 - y1)
            name = annotation['object_id']
            if name not in best or area > best[name][2]:
                best[name] = (frame, annotation['bbox'], area)
    return best


def window_around(image, bbox):
    height, width = image.shape[:2]
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    x1 = int(np.clip(cx - WINDOW // 2, 0, width - WINDOW))
    y1 = int(np.clip(cy - WINDOW // 2, 0, height - WINDOW))
    return image[y1:y1 + WINDOW, x1:x1 + WINDOW]


def as_level(window, factor):
    small = cv2.resize(window, (WINDOW // factor, WINDOW // factor),
                       interpolation=cv2.INTER_AREA) if factor > 1 else window
    return cv2.resize(small, (TILE, TILE), interpolation=cv2.INTER_NEAREST)


def main():
    scene = DEFAULT_SCENE
    best = best_sighting(scene)
    rows = []
    header = np.full((30, LABEL_WIDTH + 3 * TILE, 3), 255, np.uint8)
    for i, text in enumerate(('Level 0 (1/4)', 'Level 1 (1/2)', 'Level 2 (1:1)')):
        cv2.putText(header, text, (LABEL_WIDTH + i * TILE + 60, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    rows.append(header)

    for name in OBJECT_CLASSES:
        if name not in best:
            continue
        frame, bbox, _ = best[name]
        window = window_around(load_frame(frame, scene), bbox)
        label = np.full((TILE, LABEL_WIDTH, 3), 255, np.uint8)
        size = f'{bbox[2] - bbox[0]}x{bbox[3] - bbox[1]}px'
        cv2.putText(label, name, (8, TILE // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(label, f'{size} f{frame}', (8, TILE // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        tiles = [as_level(window, factor) for factor in (4, 2, 1)]
        rows.append(np.hstack([label] + tiles))

    OUTPUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT), np.vstack(rows))
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
