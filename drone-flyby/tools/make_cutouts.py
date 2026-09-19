"""Cut every object out of the 4K frames as RGBA images, for copy-paste.

The annotations are boxes only, so the object is separated from the ground
with GrabCut seeded by the box: outside the box is certain background, the box
interior is "probably object". Where GrabCut fails (thin or tiny objects) the
box itself is used as a soft-edged patch. The objects are rendered models pasted onto
aerial photos, which makes this split fairly clean.

One cutout per class every few frames (different lighting/scale/position),
skipping sightings clipped by the frame edge.

    python tools/make_cutouts.py        # data/cutouts/<class>/f<frame>.png
                                        # + outputs/cutouts_sheet.png to check
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

OUT = ROOT / 'data' / 'cutouts'
EVERY = 2          # frames between cutouts of the same object
PAD = 12           # context pixels around the box given to GrabCut
# GrabCut results covering less/more of the box than this are treated as
# failures (lost the object / kept the ground) and replaced by a box patch.
MIN_FILL, MAX_FILL = 0.2, 0.9
# Classes GrabCut gets wrong even when the fill looks plausible.
PATCH_CLASSES = {'helicopter', 'large_tower', 'medium_launcher', 'small_launcher'}


def cutout(image, box):
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
    cx2, cy2 = min(IMAGE_WIDTH, x2 + PAD), min(IMAGE_HEIGHT, y2 + PAD)
    crop = image[cy1:cy2, cx1:cx2].copy()
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    mask[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Keep the largest blob plus anything touching it after a small closing;
    # thin parts (rotor blades, lattice) survive the closing.
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = stats[:, cv2.CC_STAT_AREA] >= 0.05 * stats[largest, cv2.CC_STAT_AREA]
        keep[0] = False
        alpha = np.where(keep[labels], 255, 0).astype(np.uint8)
    box_alpha = alpha[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]
    fill = (box_alpha > 0).mean()
    if not MIN_FILL <= fill <= MAX_FILL:
        return None
    ys, xs = np.nonzero(alpha)
    tight = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return np.dstack([crop, alpha])[tight]


def patch(image, box):
    """The box itself with softened edges, for objects GrabCut cannot isolate
    (thin rotor blades, lattice towers, specks). Carries a little ground."""
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    alpha = np.full(crop.shape[:2], 255, np.uint8)
    alpha[[0, -1], :] = 128
    alpha[:, [0, -1]] = 128
    return np.dstack([crop, alpha])


def main():
    last_saved = {}
    saved = {}
    for frame in frame_numbers():
        image = load_frame(frame)
        for a in load_annotations(frame):
            name = a['object_id']
            x1, y1, x2, y2 = a['bbox']
            if x1 <= 1 or y1 <= 1 or x2 >= IMAGE_WIDTH - 2 or y2 >= IMAGE_HEIGHT - 2:
                continue
            if frame - last_saved.get(name, -EVERY) < EVERY:
                continue
            rgba = None if name in PATCH_CLASSES else cutout(image, (x1, y1, x2, y2))
            kind = 'cut'
            if rgba is None:
                rgba, kind = patch(image, (x1, y1, x2, y2)), 'patch'
            folder = OUT / name
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f'f{frame:02d}_{kind}.png'), rgba)
            last_saved[name] = frame
            saved.setdefault(name, []).append(rgba)

    # Contact sheet: each cutout on magenta so holes and leftovers show.
    rows = []
    for name in sorted(saved):
        tiles = []
        for rgba in saved[name][:8]:
            tile = np.full((200, 200, 3), (255, 0, 255), np.uint8)
            h, w = rgba.shape[:2]
            scale = min(190 / w, 190 / h)
            small = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_NEAREST)
            sh, sw = small.shape[:2]
            oy, ox = (200 - sh) // 2, (200 - sw) // 2
            a = small[:, :, 3:4] / 255.0
            region = tile[oy:oy + sh, ox:ox + sw]
            tile[oy:oy + sh, ox:ox + sw] = (small[:, :, :3] * a + region * (1 - a)).astype(np.uint8)
            tiles.append(tile)
        tiles += [np.full((200, 200, 3), 255, np.uint8)] * (8 - len(tiles))
        label = np.full((200, 160, 3), 255, np.uint8)
        cv2.putText(label, name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(label, f'{len(saved[name])} cutouts', (5, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        rows.append(np.hstack([label] + tiles))
    sheet = ROOT / 'outputs' / 'cutouts_sheet.png'
    sheet.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(sheet), np.vstack(rows))
    print({k: len(v) for k, v in saved.items()})
    print(f'wrote {OUT} and {sheet}')


if __name__ == '__main__':
    main()
