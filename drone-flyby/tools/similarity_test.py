"""Does "looks like one of our 16 objects" separate real objects from boats and cars?

Builds a reference gallery from data/cutouts (every cutout, 8 rotations x 2
flips, degraded to each zoom level's pixel size), embeds it with DINOv2-small,
then embeds every YOLO candidate (conf >= 0.1) in the recorded validation views
and reports its best match and cosine similarity. Output: contact sheets of
candidates sorted by similarity, so a person can judge where real objects land.

    python tools/similarity_test.py
"""

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES  # noqa: E402

SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
GREY = (118, 124, 110)          # neutral ground colour behind cut-out objects
OUT = ROOT / 'outputs'


def to_tensor(crops_bgr):
    batch = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC),
                           cv2.COLOR_BGR2RGB) / 255.0
        batch.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return torch.tensor(np.array(batch), dtype=torch.float32)


def embed(model, crops):
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            out.append(torch.nn.functional.normalize(model(to_tensor(crops[i:i + 64])), dim=1))
    return torch.cat(out).numpy()


def square_pad(image, pad_fraction=0.15, fill=GREY):
    h, w = image.shape[:2]
    side = int(max(h, w) * (1 + 2 * pad_fraction))
    canvas = np.full((side, side, 3), fill, np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = image
    return canvas


def gallery_views(rgba):
    """Rotations and flips of one cutout, composited on neutral ground."""
    alpha = rgba[:, :, 3:4] / 255.0
    flat = (rgba[:, :, :3] * alpha + np.array(GREY) * (1 - alpha)).astype(np.uint8)
    views = []
    for flip in (False, True):
        base = flat[:, ::-1] if flip else flat
        base = square_pad(np.ascontiguousarray(base), 0.25)
        side = base.shape[0]
        for angle in range(0, 360, 45):
            m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
            views.append(cv2.warpAffine(base, m, (side, side), borderValue=GREY))
    return views


def degrade(image, factor):
    if factor == 1:
        return image
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    return small


def main():
    from ultralytics import YOLO
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).eval()
    detector = YOLO(str(ROOT / 'models/kaggle_v2/yolo11n_960/best_openvino_model'), task='detect')

    # Gallery per zoom level.
    gallery = {}
    for level, factor in ((0, 4), (1, 2), (2, 1)):
        crops, labels = [], []
        for name in OBJECT_CLASSES:
            for path in sorted((ROOT / 'data/cutouts' / name).glob('*.png')):
                for view in gallery_views(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)):
                    crops.append(degrade(view, factor))
                    labels.append(name)
        gallery[level] = (embed(model, crops), np.array(labels))
        print(f'gallery L{level}: {len(crops)} reference views', flush=True)

    # Candidates from the recorded validation views.
    records = []
    for path in sorted(glob.glob(str(ROOT / 'data/recordings/validation/*/*.json'))):
        meta = json.load(open(path))
        level = meta['view']['resolution_level']
        image = cv2.imread(path.replace('.json', '.png'))
        result = detector.predict(image, imgsz=(544, 960), conf=0.1, device='intel:cpu',
                                  verbose=False, agnostic_nms=True)[0]
        for box, cls, conf in zip(result.boxes.xyxy.numpy(), result.boxes.cls.numpy(),
                                  result.boxes.conf.numpy()):
            x1, y1, x2, y2 = box.astype(int)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            pad = int(max(x2 - x1, y2 - y1) * 0.15) + 1
            crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            records.append({'crop': crop, 'level': level, 'yolo': OBJECT_CLASSES[int(cls)],
                            'conf': float(conf), 'frame': meta['frame']})
    print(f'{len(records)} candidates', flush=True)

    for level in (0, 1, 2):
        items = [r for r in records if r['level'] == level]
        if not items:
            continue
        vectors = embed(model, [square_pad(r['crop'], 0.0) for r in items])
        features, labels = gallery[level]
        sims = vectors @ features.T
        for r, row in zip(items, sims):
            best = int(row.argmax())
            r['match'], r['sim'] = labels[best], float(row[best])

    sims = np.array([r['sim'] for r in records])
    print('similarity percentiles 10/50/90/99:', np.percentile(sims, [10, 50, 90, 99]).round(3))

    # Contact sheets: highest, middle and lowest similarity.
    ordered = sorted(records, key=lambda r: -r['sim'])
    for name, chunk in (('top', ordered[:60]), ('middle', ordered[len(ordered) // 2 - 30:len(ordered) // 2 + 30]),
                        ('bottom', ordered[-60:])):
        tiles = []
        for r in chunk:
            tile = cv2.resize(square_pad(r['crop'], 0.0), (150, 150), interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(tile, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, f"{r['match']} {r['sim']:.2f}", (3, 164), 0, 0.4, (0, 0, 180), 1)
            cv2.putText(tile, f"yolo {r['yolo']} L{r['level']}", (3, 179), 0, 0.35, (60, 60, 60), 1)
            tiles.append(tile)
        while len(tiles) % 10:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
        cv2.imwrite(str(OUT / f'similarity_{name}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote outputs/similarity_{top,middle,bottom}.jpg')


if __name__ == '__main__':
    main()
