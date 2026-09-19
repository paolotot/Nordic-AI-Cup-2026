"""Build a YOLO training set by copy-paste, from two kinds of background.

1. helsinki views: a random camera view (level 0, 1 or 2) of a supplied 4K
   frame. Real objects keep their real labels; cutouts are pasted at source
   scale and the view is downsampled exactly as the evaluator does it.
2. recorded validation views (data/recordings/validation, if present): the
   960x540 images the evaluator actually sent us. Harbours, cars and roofs we
   must learn to ignore. Spots the old model was confident about (.mask.json)
   are painted over unless the view has hand labels (.labels.json), which then
   become real labels. Cutouts are scaled down to the view's level first.

Pasted cutouts are rotated, flipped, rescaled, strongly relit and (mostly)
given a cast shadow from a per-image sun direction, because the validation
scene renders shadows and helsinki does not.

Train/val: helsinki frames 0-19 / 20-24; recordings of the full validation run
/ the earlier partial run. Both overlap their train side in ground, so val is
a sanity check; the real test is a validation run.

    python tools/make_dataset.py                             # data/synth
    python tools/make_dataset.py --train 200 --rec-train 200 --val 50 --rec-val 50
"""

import argparse
import functools
import glob
import json
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
RECORDINGS = ROOT / 'data' / 'recordings' / 'validation'
OUT = ROOT / 'data' / 'synth'
LEVEL_WEIGHTS = {0: 0.3, 1: 0.35, 2: 0.35}
PASTES_PER_VIEW = {0: (6, 14), 1: (3, 8), 2: (1, 4)}
DOWNSAMPLE = {0: 4, 1: 2, 2: 1}
MIN_VISIBLE = 0.5      # label a real object if this much of it is in view
MIN_LABEL_PX = 3       # drop labels smaller than this in the transmitted image
NO_SHADOW_SHARE = 0.25
REAL_SHARE = 0.7           # pastes of a class with real validation crops that use one
LABELLED_SHARE = 0.5       # recording-based images built on a hand-labelled view


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
    """Rotate/flip/scale/relight one cutout at source scale. Patches only turn
    by 90 degrees so their ground stays a clean rectangle."""
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if is_patch:
        rgba = np.rot90(rgba, rng.integers(4)).copy()
        # Fade the patch's own (helsinki) ground out toward its edges, so no
        # hard square gives the paste away.
        h, w = rgba.shape[:2]
        fade_y = np.clip(np.minimum(np.arange(h), np.arange(h)[::-1]) / max(1, 0.2 * h), 0, 1)
        fade_x = np.clip(np.minimum(np.arange(w), np.arange(w)[::-1]) / max(1, 0.2 * w), 0, 1)
        rgba[:, :, 3] = (rgba[:, :, 3] * np.outer(fade_y, fade_x)).astype(np.uint8)
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
    colour = rgba[:, :, :3].astype(np.float32) / 255
    colour = colour ** rng.uniform(0.7, 1.4)                      # gamma
    colour = colour * rng.uniform(0.6, 1.3) + rng.uniform(-0.08, 0.08)
    colour *= rng.uniform(0.9, 1.1, size=3)                       # tint
    grey = colour.mean(axis=2, keepdims=True)
    colour = grey + (colour - grey) * rng.uniform(0.6, 1.3)       # saturation
    rgba[:, :, :3] = np.clip(colour * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        rgba[:, :, :3] = cv2.GaussianBlur(rgba[:, :, :3], (3, 3), 0)
    ys, xs = np.nonzero(rgba[:, :, 3] > 40)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste(view, rgba, px, py, shadow):
    """Alpha-blend rgba at (px, py), after darkening the ground under its
    shadow. shadow = (dx, dy, strength) in view pixels, or None."""
    height, width = view.shape[:2]
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3].astype(np.float32) / 255
    if shadow is not None:
        dx, dy, strength = shadow
        k = max(1, int(0.12 * max(h, w))) | 1
        soft = cv2.GaussianBlur(alpha, (k, k), 0) if k > 1 else alpha
        sx, sy = px + int(round(dx)), py + int(round(dy))
        x1, y1, x2, y2 = max(0, sx), max(0, sy), min(width, sx + w), min(height, sy + h)
        if x2 > x1 and y2 > y1:
            s = soft[y1 - sy:y2 - sy, x1 - sx:x2 - sx, None]
            region = view[y1:y2, x1:x2].astype(np.float32)
            view[y1:y2, x1:x2] = (region * (1 - strength * s)).astype(np.uint8)
    a = alpha[:, :, None]
    region = view[py:py + h, px:px + w].astype(np.float32)
    view[py:py + h, px:px + w] = (rgba[:, :, :3] * a + region * (1 - a)).astype(np.uint8)


def overlaps(box, boxes, margin=8):
    return any(box[0] < b[2] + margin and b[0] < box[2] + margin and
               box[1] < b[3] + margin and b[1] < box[3] + margin for b in boxes)


def sun(rng):
    """Per-image shadow direction, length (x object size) and darkness."""
    if rng.random() < NO_SHADOW_SHARE:
        return None
    return rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.6), rng.uniform(0.3, 0.65)


def paste_many(view, cutouts, level, rng, occupied, labels, downsample, real=None):
    """Paste a level-appropriate number of cutouts, scaled down by `downsample`.
    Classes with real validation crops use one of those most of the time."""
    height, width = view.shape[:2]
    light = sun(rng)
    low, high = PASTES_PER_VIEW[level]
    for _ in range(int(rng.integers(low, high + 1))):
        name = OBJECT_CLASSES[rng.integers(len(OBJECT_CLASSES))]
        is_real = bool(real and real.get(name)) and rng.random() < REAL_SHARE
        if is_real:
            crop, scale = real[name][rng.integers(len(real[name]))]
            rgba, is_patch = real_variant(crop, scale, downsample, rng), True
        else:
            rgba, is_patch = cutouts[name][rng.integers(len(cutouts[name]))]
            rgba = augment(rgba, is_patch, rng)
            if downsample > 1:
                h, w = rgba.shape[:2]
                rgba = cv2.resize(np.ascontiguousarray(rgba),
                                  (max(2, round(w / downsample)), max(2, round(h / downsample))),
                                  interpolation=cv2.INTER_AREA)
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
        shadow = None
        if light is not None and not is_real:          # real crops carry their own shadow
            angle, length, strength = light
            reach = length * max(h, w)
            shadow = (np.cos(angle) * reach, np.sin(angle) * reach,
                      strength * (0.5 if is_patch else 1.0))
        paste(view, rgba, px, py, shadow)
        occupied.append(box)
        labels.append((name, box))


def yolo_lines(labels, factor):
    lines = []
    for name, (bx1, by1, bx2, by2) in labels:
        w, h = (bx2 - bx1) / factor, (by2 - by1) / factor
        if max(w, h) < MIN_LABEL_PX:
            continue
        cxn = (bx1 + bx2) / 2 / factor / TRANSMITTED_VIEW_SIZE[0]
        cyn = (by1 + by2) / 2 / factor / TRANSMITTED_VIEW_SIZE[1]
        lines.append(f'{OBJECT_CLASSES.index(name)} {cxn:.6f} {cyn:.6f} '
                     f'{w / TRANSMITTED_VIEW_SIZE[0]:.6f} {h / TRANSMITTED_VIEW_SIZE[1]:.6f}')
    return lines


@functools.lru_cache(maxsize=None)
def cached_frame(frame):
    return load_frame(frame)


def helsinki_view(frame, level, cutouts, rng, real=None):
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

    paste_many(view, cutouts, level, rng, occupied, labels, downsample=1, real=real)
    height, width = view.shape[:2]
    if (width, height) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view, yolo_lines(labels, width / TRANSMITTED_VIEW_SIZE[0])


def load_holdout():
    """Labelled views kept out of training for tools/eval_views.py."""
    path = RECORDINGS / 'holdout.json'
    return set(json.loads(path.read_text())) if path.exists() else set()


def load_real_crops(records):
    """Real validation objects cut from hand-labelled views: {class: [(rgba, scale)]}.

    `scale` is the view's downsample factor, so a crop can be pasted at the
    right size into a view of any level. Box edges fade out over a few pixels.
    """
    crops = {}
    for png, level, labels, _ in records:
        if not labels:
            continue
        view = cv2.imread(png)
        height, width = view.shape[:2]
        for a in labels:
            x1, y1, x2, y2 = (int(round(c)) for c in a['bbox'])
            pad = 3
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = view[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            fade_y = np.clip(np.minimum(np.arange(h), np.arange(h)[::-1]) / pad, 0, 1)
            fade_x = np.clip(np.minimum(np.arange(w), np.arange(w)[::-1]) / pad, 0, 1)
            alpha = (np.outer(fade_y, fade_x) * 255).astype(np.uint8)
            crops.setdefault(a['object_id'], []).append((np.dstack([crop, alpha]), DOWNSAMPLE[level]))
    return crops


def real_variant(rgba, scale, downsample, rng):
    """Flip/turn/relight a real crop and size it for a view downsampled by `downsample`."""
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if rng.random() < 0.5:
        rgba = rgba[::-1]
    rgba = np.ascontiguousarray(np.rot90(rgba, rng.integers(4)))
    factor = scale / downsample * rng.uniform(0.85, 1.2)
    h, w = rgba.shape[:2]
    rgba = cv2.resize(rgba, (max(2, round(w * factor)), max(2, round(h * factor))),
                      interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    colour = rgba[:, :, :3].astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
    rgba[:, :, :3] = np.clip(colour, 0, 255).astype(np.uint8)
    return rgba


def load_recordings(sequences, exclude=frozenset()):
    """[(png path, level, hand labels or None, mask boxes)] for the given sequences."""
    out = []
    for sequence in sequences:
        for meta_path in sorted(glob.glob(str(sequence / '*_f*.json'))):
            if meta_path.endswith(('.mask.json', '.labels.json', '.ignore.json')):
                continue
            if f'{sequence.name}/{Path(meta_path).stem}' in exclude:
                continue
            level = json.load(open(meta_path))['view']['resolution_level']
            labels_path = Path(meta_path.replace('.json', '.labels.json'))
            mask_path = Path(meta_path.replace('.json', '.mask.json'))
            ignore_path = Path(meta_path.replace('.json', '.ignore.json'))
            labels = json.loads(labels_path.read_text()) if labels_path.exists() else None
            if labels is not None:
                # Hand-labelled: only the spots marked unsure get painted over.
                masks = json.loads(ignore_path.read_text()) if ignore_path.exists() else []
            else:
                masks = json.loads(mask_path.read_text()) if mask_path.exists() else []
            out.append((meta_path.replace('.json', '.png'), level, labels, masks))
    return out


@functools.lru_cache(maxsize=None)
def cached_background(png, masks_key):
    """The recorded view with confident-but-unlabelled spots painted over."""
    view = cv2.imread(png)
    if masks_key:
        mask = np.zeros(view.shape[:2], np.uint8)
        for x1, y1, x2, y2 in json.loads(masks_key):
            pad = 0.15 * max(x2 - x1, y2 - y1) + 2
            cv2.rectangle(mask, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(y2 + pad)), 255, -1)
        view = cv2.inpaint(view, mask, 5, cv2.INPAINT_TELEA)
    return view


def recorded_view(record, cutouts, rng, real=None):
    png, level, hand_labels, masks = record
    labels, occupied = [], []
    view = cached_background(png, json.dumps(masks) if masks else '').copy()
    occupied = [tuple(m) for m in masks]
    for a in hand_labels or []:
        box = tuple(a['bbox'])
        labels.append((a['object_id'], box))
        occupied.append(box)
    if rng.random() < 0.5:                       # flips are free looking straight down
        view, labels, occupied = flip(view, labels, occupied, horizontal=True)
    if rng.random() < 0.5:
        view, labels, occupied = flip(view, labels, occupied, horizontal=False)
    paste_many(view, cutouts, level, rng, occupied, labels, downsample=DOWNSAMPLE[level], real=real)
    return view, yolo_lines(labels, 1.0)


def flip(view, labels, occupied, horizontal):
    height, width = view.shape[:2]
    if horizontal:
        mirror = lambda b: (width - b[2], b[1], width - b[0], b[3])  # noqa: E731
        view = np.ascontiguousarray(view[:, ::-1])
    else:
        mirror = lambda b: (b[0], height - b[3], b[2], height - b[1])  # noqa: E731
        view = np.ascontiguousarray(view[::-1])
    return view, [(n, mirror(b)) for n, b in labels], [mirror(b) for b in occupied]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--train', type=int, default=1500, help='helsinki-based train images')
    parser.add_argument('--val', type=int, default=300)
    parser.add_argument('--rec-train', type=int, default=4500, help='recording-based train images')
    parser.add_argument('--rec-val', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    cutouts = load_cutouts()
    levels, weights = list(LEVEL_WEIGHTS), list(LEVEL_WEIGHTS.values())

    # The largest recorded run trains; the others validate. Held-out labelled
    # views (tools/eval_views.py) are used for nothing here.
    sequences = sorted((p for p in RECORDINGS.glob('*') if p.is_dir()),
                       key=lambda p: -len(list(p.glob('*.png'))))
    holdout = load_holdout()
    recordings = {'train': load_recordings(sequences[:1], holdout),
                  'val': load_recordings(sequences[1:], holdout)}
    if not recordings['val']:
        recordings['val'] = recordings['train'][::10]
    labelled = {split: [r for r in recs if r[2] is not None] for split, recs in recordings.items()}
    real = load_real_crops(labelled['train'])
    print({k: len(v) for k, v in recordings.items()}, 'recorded views;',
          len(labelled['train']), 'hand-labelled in train;', len(holdout), 'held out;',
          sum(len(v) for v in real.values()), 'real crops over', len(real), 'classes', flush=True)

    plan = {'train': (arguments.train, range(0, 20), arguments.rec_train),
            'val': (arguments.val, range(20, 25), arguments.rec_val)}
    for split, (count, frames, rec_count) in plan.items():
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)
        jobs = [('h', i) for i in range(count)]
        if recordings[split]:
            jobs += [('r', i) for i in range(rec_count)]
        for n, (kind, i) in enumerate(jobs):
            if kind == 'h':
                frame = int(rng.choice(list(frames)))
                level = int(rng.choice(levels, p=weights))
                view, lines = helsinki_view(frame, level, cutouts, rng, real)
                stem = f'{split}_h{i:05d}_f{frame:02d}_L{level}'
            else:
                pool = (labelled[split] if labelled[split] and rng.random() < LABELLED_SHARE
                        else recordings[split])
                record = pool[rng.integers(len(pool))]
                view, lines = recorded_view(record, cutouts, rng, real)
                stem = f'{split}_r{i:05d}_{Path(record[0]).stem}_L{record[1]}'
            cv2.imwrite(str(OUT / 'images' / split / f'{stem}.jpg'), view,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
            if (n + 1) % 1000 == 0:
                print(f'{split}: {n + 1}/{len(jobs)}', flush=True)

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(OBJECT_CLASSES))
    (OUT / 'data.yaml').write_text(
        f'path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
