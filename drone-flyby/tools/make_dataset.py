"""Build a YOLO training set from the helsinki frames plus copy-paste.

Each sample is a random camera view (level 0, 1 or 2) of a real 4K frame:
* the real objects already in that view keep their real labels;
* a handful of cutouts (tools/make_cutouts.py) are pasted onto empty ground,
  randomly rotated, flipped, rescaled and relit, classes balanced;
* the view is then downsampled to 960x540 exactly as the evaluator does it.

Train views come from frames 0-19 and val views from frames 20-24. The two
overlap in ground (the drone only moves ~65 px per frame), so val is a sanity
check; the real test is a validation run on the competition server.

    python tools/make_dataset.py                       # data/synth, 3000 + 300
    python tools/make_dataset.py --train 200 --val 50  # quick
"""

import argparse
import functools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES, TRANSMITTED_VIEW_SIZE  # noqa: E402
from utils import (  # noqa: E402
    center_bounds_for_level,
    load_annotations,
    load_frame,
    source_region_for_view,
)

CUTOUTS = ROOT / 'data' / 'cutouts'
OUT = ROOT / 'data' / 'synth'
LEVEL_WEIGHTS = {0: 0.3, 1: 0.35, 2: 0.35}
PASTES_PER_VIEW = {0: (6, 14), 1: (3, 8), 2: (1, 4)}
MIN_VISIBLE = 0.5      # label a real object if this much of it is in view
MIN_LABEL_PX = 3       # drop labels smaller than this in the transmitted image


def load_cutouts():
    cutouts = {}
    for name in OBJECT_CLASSES:
        files = sorted((CUTOUTS / name).glob('*.png'))
        cutouts[name] = [(cv2.imread(str(f), cv2.IMREAD_UNCHANGED), 'patch' in f.stem)
                         for f in files]
    missing = [n for n, v in cutouts.items() if not v]
    if missing:
        raise SystemExit(f'no cutouts for {missing}; run tools/make_cutouts.py first')
    return cutouts


def augment(rgba, is_patch, rng):
    """Rotate/flip/scale/relight one cutout. Patches only turn by 90 degrees so
    their ground stays a clean rectangle."""
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if is_patch:
        rgba = np.rot90(rgba, rng.integers(4))
    else:
        h, w = rgba.shape[:2]
        angle = rng.uniform(0, 360)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        rgba = cv2.warpAffine(np.ascontiguousarray(rgba), m, (nw, nh),
                              flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    scale = rng.uniform(0.8, 1.25)
    h, w = rgba.shape[:2]
    rgba = cv2.resize(np.ascontiguousarray(rgba),
                      (max(2, round(w * scale)), max(2, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    colour = rgba[:, :, :3].astype(np.float32)
    colour = colour * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    colour *= rng.uniform(0.93, 1.07, size=3)          # slight tint
    rgba[:, :, :3] = np.clip(colour, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(rgba[:, :, 3] > 40)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def overlaps(box, boxes, margin=8):
    return any(box[0] < b[2] + margin and b[0] < box[2] + margin and
               box[1] < b[3] + margin and b[1] < box[3] + margin for b in boxes)


@functools.lru_cache(maxsize=None)
def cached_frame(frame):
    return load_frame(frame)


def make_view(frame, level, cutouts, rng):
    image = cached_frame(frame)
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    cx, cy = int(rng.integers(min_x, max_x + 1)), int(rng.integers(min_y, max_y + 1))
    x1, y1, x2, y2 = source_region_for_view(level, cx, cy)
    view = image[y1:y2, x1:x2].copy()

    labels, occupied = [], []
    for a in load_annotations(frame):
        b = a['bbox']
        occupied.append((b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1))
        inter = (max(b[0], x1), max(b[1], y1), min(b[2], x2), min(b[3], y2))
        if inter[2] <= inter[0] or inter[3] <= inter[1]:
            continue
        visible = (inter[2] - inter[0]) * (inter[3] - inter[1]) / ((b[2] - b[0]) * (b[3] - b[1]))
        if visible >= MIN_VISIBLE:
            labels.append((a['object_id'], (inter[0] - x1, inter[1] - y1,
                                            inter[2] - x1, inter[3] - y1)))

    height, width = view.shape[:2]
    low, high = PASTES_PER_VIEW[level]
    for _ in range(int(rng.integers(low, high + 1))):
        name = OBJECT_CLASSES[rng.integers(len(OBJECT_CLASSES))]
        rgba, is_patch = cutouts[name][rng.integers(len(cutouts[name]))]
        rgba = augment(rgba, is_patch, rng)
        h, w = rgba.shape[:2]
        if w >= width or h >= height:
            continue
        for _attempt in range(20):
            px, py = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
            box = (px, py, px + w, py + h)
            if not overlaps(box, occupied):
                break
        else:
            continue
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255
        region = view[py:py + h, px:px + w].astype(np.float32)
        view[py:py + h, px:px + w] = (rgba[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
        occupied.append(box)
        labels.append((name, box))

    if (width, height) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    factor = width / TRANSMITTED_VIEW_SIZE[0]
    lines = []
    for name, (bx1, by1, bx2, by2) in labels:
        w, h = (bx2 - bx1) / factor, (by2 - by1) / factor
        if max(w, h) < MIN_LABEL_PX:
            continue
        cxn = (bx1 + bx2) / 2 / factor / TRANSMITTED_VIEW_SIZE[0]
        cyn = (by1 + by2) / 2 / factor / TRANSMITTED_VIEW_SIZE[1]
        lines.append(f'{OBJECT_CLASSES.index(name)} {cxn:.6f} {cyn:.6f} '
                     f'{w / TRANSMITTED_VIEW_SIZE[0]:.6f} {h / TRANSMITTED_VIEW_SIZE[1]:.6f}')
    return view, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--val', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    cutouts = load_cutouts()
    levels, weights = list(LEVEL_WEIGHTS), list(LEVEL_WEIGHTS.values())
    splits = {'train': (arguments.train, range(0, 20)), 'val': (arguments.val, range(20, 25))}
    for split, (count, frames) in splits.items():
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            frame = int(rng.choice(list(frames)))
            level = int(rng.choice(levels, p=weights))
            view, lines = make_view(frame, level, cutouts, rng)
            stem = f'{split}_{i:05d}_f{frame:02d}_L{level}'
            cv2.imwrite(str(OUT / 'images' / split / f'{stem}.jpg'), view,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
            if (i + 1) % 500 == 0:
                print(f'{split}: {i + 1}/{count}', flush=True)

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(OBJECT_CLASSES))
    (OUT / 'data.yaml').write_text(
        f'path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
