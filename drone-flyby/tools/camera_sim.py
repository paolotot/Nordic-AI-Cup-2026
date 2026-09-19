"""Compare camera strategies before there is a detector.

The detector is replaced by the ground truth, filtered by what the camera could
plausibly show: an object is *recognised* when it is mostly inside the current
view and looks at least ``T`` pixels big in the transmitted image
(sqrt(w*h) / downsample factor). Objects smaller than that but above ``0.4*T``
are only *noticed*: the camera sees a blob but cannot name it.

Every recognised object is remembered and moved along with the ground. The
remembered box picks up a tracking error of ``drift`` pixels per frame since
it was last seen (a per-object velocity error), which is what makes re-looking
at small objects matter.

Camera commands go through the evaluator's own Camera class, so illegal moves
are refused exactly as they would be for real, and scoring uses the same COCO
mAP@0.50.

Worlds:
* ``helsinki`` - the 25 supplied frames, with their real annotations.
* synthetic   - 250-frame flights with the motion fitted on helsinki (slide
  plus ~1% zoom per frame, speed wandering 46-58 px), objects entering at the
  top at a rate that matches helsinki's density (~11 on screen at once).

    python tools/camera_sim.py                 # the full comparison table
    python tools/camera_sim.py --seeds 10      # more synthetic flights
"""

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import (  # noqa: E402
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
)
from local_evaluator import Camera, CameraRejection  # noqa: E402
from utils import center_bounds_for_level, frame_numbers, load_annotations  # noqa: E402

DOWNSAMPLE = {0: 4, 1: 2, 2: 1}
NOTICE_FRACTION = 0.4
MIN_VISIBLE_FRACTION = 0.6

# Motion fitted on helsinki: x' = cx + SX (x - cx), y' = SY y + b_t
SX, SY, CX = 1.0071, 1.0133, 1918.0


# --------------------------------------------------------------------------- #
# Worlds: gt[t] = list of (object key, class, clipped box)
# --------------------------------------------------------------------------- #

World = Dict[int, List[Tuple[str, str, Tuple[float, float, float, float]]]]


def helsinki_world() -> World:
    return {
        frame: [(a['object_id'], a['object_id'], tuple(map(float, a['bbox'])))
                for a in load_annotations(frame)]
        for frame in frame_numbers()
    }


def class_sizes() -> Dict[str, List[Tuple[float, float]]]:
    """Unclipped box sizes per class seen in helsinki."""
    sizes: Dict[str, List[Tuple[float, float]]] = {}
    for frame in frame_numbers():
        for a in load_annotations(frame):
            x1, y1, x2, y2 = a['bbox']
            if x1 > 2 and y1 > 2 and x2 < IMAGE_WIDTH - 3 and y2 < IMAGE_HEIGHT - 3:
                sizes.setdefault(a['object_id'], []).append((x2 - x1, y2 - y1))
    return sizes


def synthetic_world(seed: int, frames: int = 250, rate: float = 0.35,
                    burn_in: int = 45) -> World:
    rng = np.random.default_rng(seed)
    sizes = class_sizes()
    classes = sorted(sizes)
    # centre x, centre y, w, h, class, key
    objects: List[list] = []
    world: World = {}
    offset = 51.0
    next_key = 0
    for t in range(-burn_in, frames):
        offset = float(np.clip(offset + rng.normal(0, 0.8), 46, 58))
        for o in objects:
            o[0] = CX + SX * (o[0] - CX)
            o[1] = SY * o[1] + offset
            o[2] *= SX
            o[3] *= SY
        for _ in range(rng.poisson(rate)):
            name = classes[rng.integers(len(classes))]
            w, h = sizes[name][rng.integers(len(sizes[name]))]
            # Sizes in helsinki are measured mid-flight; start a bit smaller.
            w, h = w * 0.85, h * 0.85
            if rng.random() < 0.5:
                w, h = h, w
            objects.append([rng.uniform(0, IMAGE_WIDTH), -h / 2 + rng.uniform(0, offset),
                            w, h, name, f'{name}#{next_key}'])
            next_key += 1
        objects = [o for o in objects
                   if o[1] - o[3] / 2 < IMAGE_HEIGHT and -o[2] < o[0] < IMAGE_WIDTH + o[2]]
        if t < 0:
            continue
        world[t] = []
        for cx, cy, w, h, name, key in objects:
            box = (max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                   min(float(IMAGE_WIDTH), cx + w / 2), min(float(IMAGE_HEIGHT), cy + h / 2))
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                world[t].append((key, name, box))
    return world


# --------------------------------------------------------------------------- #
# Perception and memory
# --------------------------------------------------------------------------- #

def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def look(world_frame, region, level, threshold):
    """What the camera shows: (recognised keys, noticed-only keys)."""
    recognised, noticed = [], []
    for key, _, box in world_frame:
        if area(intersect(box, region)) < MIN_VISIBLE_FRACTION * area(box):
            continue
        apparent = math.sqrt(area(box)) / DOWNSAMPLE[level]
        if apparent >= threshold:
            recognised.append(key)
        elif apparent >= NOTICE_FRACTION * threshold:
            noticed.append(key)
    return recognised, noticed


@dataclass
class Track:
    last_seen: int
    velocity_error: Tuple[float, float]


@dataclass
class Memory:
    drift: float
    rng: np.random.Generator
    tracks: Dict[str, Track] = field(default_factory=dict)
    candidates: Dict[str, int] = field(default_factory=dict)  # noticed, unnamed

    def update(self, t, recognised, noticed):
        for key in recognised:
            self.tracks[key] = Track(t, tuple(self.rng.normal(0, self.drift, 2)))
            self.candidates.pop(key, None)
        for key in noticed:
            if key not in self.tracks:
                self.candidates[key] = t

    def predictions(self, t, world_frame):
        out = []
        for key, name, box in world_frame:
            track = self.tracks.get(key)
            if track is None:
                continue
            age = t - track.last_seen
            dx, dy = track.velocity_error[0] * age, track.velocity_error[1] * age
            moved = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
            clipped = intersect(moved, (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT))
            if area(clipped) > 0:
                out.append({'object_id': name, 'bbox': clipped,
                            'confidence': 0.95 * 0.97 ** age + 0.04})
        return out


# --------------------------------------------------------------------------- #
# Camera strategies
# --------------------------------------------------------------------------- #

def legal_step(camera: Camera, level: int, x: float, y: float) -> Tuple[int, int, int]:
    """The closest legal command toward (level, x, y) from where the camera is."""
    if level == 0:
        if 0 in ALLOWED_RESOLUTION_LEVELS[camera.resolution_level]:
            return (0, *FULL_FRAME_CENTER)
        level = 1
    if level not in ALLOWED_RESOLUTION_LEVELS[camera.resolution_level]:
        level = 1
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    limit = MAXIMUM_CENTER_DELTA_PIXELS[camera.resolution_level] - 1
    here = np.array([camera.center_x, camera.center_y], float)
    # Walk from the nearest in-bounds point toward the target; both ends are
    # in bounds, so every point between them is too. Take the furthest one
    # within the distance limit.
    start = np.array([np.clip(here[0], min_x, max_x), np.clip(here[1], min_y, max_y)])
    goal = np.array([np.clip(x, min_x, max_x), np.clip(y, min_y, max_y)])
    low, high = 0.0, 1.0
    if np.hypot(*(goal - here)) > limit:
        for _ in range(30):
            mid = (low + high) / 2
            if np.hypot(*(start + mid * (goal - start) - here)) <= limit:
                low = mid
            else:
                high = mid
        goal = start + low * (goal - start)
    return level, int(round(goal[0])), int(round(goal[1]))


class Strategy:
    name = 'strategy'

    def next_view(self, t, camera, memory, world_frame):
        raise NotImplementedError


class AlwaysFull(Strategy):
    name = 'A  always L0'

    def next_view(self, t, camera, memory, world_frame):
        return (0, *FULL_FRAME_CENTER)


class Tour(Strategy):
    """Cycle through a fixed list of (level, x, y) waypoints."""

    def __init__(self, name, waypoints):
        self.name = name
        self.waypoints = waypoints
        self.index = 0

    def next_view(self, t, camera, memory, world_frame):
        level, x, y = self.waypoints[self.index]
        command = legal_step(camera, level, x, y)
        if command == (level, x, y) or (camera.resolution_level, camera.center_x,
                                        camera.center_y) == (level, x, y):
            self.index = (self.index + 1) % len(self.waypoints)
        return command


def band_tour(level, y, count):
    min_x, max_x, _, _ = center_bounds_for_level(level)
    xs = list(np.linspace(min_x, max_x, count).round().astype(int))
    ping_pong = xs + xs[-2:0:-1]
    return [(level, int(x), y) for x in ping_pong]


class LookAndZoom(Strategy):
    """L0 to find blobs; zoom to L1 (and L2 if needed) on the best cluster."""

    def __init__(self, name, allow_level2):
        self.name = name
        self.allow_level2 = allow_level2
        self.target: Optional[Tuple[float, float]] = None

    def next_view(self, t, camera, memory, world_frame):
        positions = {key: box for key, _, box in world_frame}
        pending = [positions[k] for k in memory.candidates if k in positions]
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + 60) for b in pending]
        if camera.resolution_level == 0:
            if not centres:
                return (0, *FULL_FRAME_CENTER)
            best = max(centres, key=lambda c: sum(
                abs(c[0] - o[0]) < 900 and abs(c[1] - o[1]) < 500 for o in centres))
            return legal_step(camera, 1, *best)
        if camera.resolution_level == 1 and self.allow_level2 and centres:
            half_w, half_h = 960, 540
            inside = [c for c in centres
                      if abs(c[0] - camera.center_x) < half_w
                      and abs(c[1] - camera.center_y) < half_h]
            if inside:
                return legal_step(camera, 2, *inside[0])
        return legal_step(camera, 0, 0, 0)


def strategies():
    return [
        AlwaysFull(),
        Tour('B1 L1 top band, 3 stops', band_tour(1, 600, 3)),
        Tour('B2 L1 full tour, 6 stops', [(1, 960, 540), (1, 1920, 540), (1, 2880, 540),
                                          (1, 2880, 1620), (1, 1920, 1620), (1, 960, 1620)]),
        Tour('B3 L2 top band, 6 stops', band_tour(2, 330, 6)),
        Tour('B4 L1 band + L0 every 4th', [(1, 960, 600), (1, 1920, 600), (1, 2880, 600),
                                           (0, 1920, 1080)]),
        LookAndZoom('C1 look L0, zoom L1', allow_level2=False),
        LookAndZoom('C2 look L0, zoom L1/L2', allow_level2=True),
    ]


# --------------------------------------------------------------------------- #
# Running and scoring
# --------------------------------------------------------------------------- #

def run(world: World, strategy: Strategy, threshold: float, drift: float,
        answer_every: int, seed: int):
    camera = Camera()
    memory = Memory(drift, np.random.default_rng(seed))
    predictions = {}
    refused = 0
    for t in sorted(world):
        if t % answer_every:
            continue  # a frame we were too slow for: no answer, camera frozen
        recognised, noticed = look(world[t], camera.source_region,
                                   camera.resolution_level, threshold)
        memory.update(t, recognised, noticed)
        predictions[t] = memory.predictions(t, world[t])
        command = strategy.next_view(t, camera, memory, world[t])
        try:
            camera.apply(*command)
        except CameraRejection:
            refused += 1
    return coco_map50(world, predictions), refused


def coco_map50(world: World, predictions) -> float:
    from faster_coco_eval import COCO, COCOeval_faster
    import contextlib
    import io

    frames = sorted(world)
    ids = {f: i for i, f in enumerate(frames, start=1)}
    cats = {n: i for i, n in enumerate(OBJECT_CLASSES, start=1)}
    anns = []
    for f in frames:
        for _, name, b in world[f]:
            anns.append({'id': len(anns) + 1, 'image_id': ids[f], 'category_id': cats[name],
                         'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         'area': area(b), 'iscrowd': 0})
    present = sorted({a['category_id'] for a in anns})
    dets = [{'image_id': ids[f], 'category_id': cats[d['object_id']],
             'bbox': [d['bbox'][0], d['bbox'][1], d['bbox'][2] - d['bbox'][0],
                      d['bbox'][3] - d['bbox'][1]], 'score': d['confidence']}
            for f, ds in predictions.items() for d in ds]
    if not dets:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO({'images': [{'id': i, 'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT}
                              for i in ids.values()],
                   'categories': [{'id': i, 'name': n} for n, i in cats.items()],
                   'annotations': anns})
        ev = COCOeval_faster(gt, gt.loadRes(dets), 'bbox')
        ev.params.imgIds = list(ids.values())
        ev.params.catIds = present
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision']
    aps = []
    for c in range(len(present)):
        p = precision[0, :, c, 0, -1]
        p = p[p > -1]
        aps.append(float(p.mean()) if p.size else 0.0)
    return sum(aps) / len(aps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[12, 20, 28])
    parser.add_argument('--drifts', type=float, nargs='+', default=[0.3, 1.0])
    arguments = parser.parse_args()

    worlds = [('helsinki', helsinki_world())] + [
        (f'synthetic{s}', synthetic_world(s)) for s in range(arguments.seeds)]
    density = np.mean([len(v) for _, w in worlds[1:] for v in w.values()])
    print(f'synthetic worlds: {arguments.seeds} x 250 frames, '
          f'{density:.1f} objects on screen on average (helsinki: '
          f'{np.mean([len(v) for v in worlds[0][1].values()]):.1f})\n')

    names = [s.name for s in strategies()]
    for answer_every, drift in itertools.product((1, 2), arguments.drifts):
        if answer_every == 2 and drift != arguments.drifts[0]:
            continue
        print(f'== answering {"every frame" if answer_every == 1 else "every 2nd frame (slow server)"}'
              f', tracking drift {drift} px/frame ==')
        header = f'{"strategy":30s}' + ''.join(
            f'  T={t:<4g} syn  hel' for t in arguments.thresholds)
        print(header)
        for i, name in enumerate(names):
            row = f'{name:30s}'
            for threshold in arguments.thresholds:
                scores = {}
                for world_name, world in worlds:
                    strategy = strategies()[i]
                    score, refused = run(world, strategy, threshold, drift, answer_every, seed=1)
                    assert refused == 0, f'{name} made {refused} illegal moves'
                    scores[world_name] = score
                synthetic = np.mean([v for k, v in scores.items() if k != 'helsinki'])
                row += f'       {synthetic:.2f} {scores["helsinki"]:.2f}'
            print(row)
        print()
    print('T = how many pixels an object must span in the transmitted image to be '
          'recognised.\nsyn = mean over synthetic 250-frame flights, hel = the 25 '
          'helsinki frames.')


if __name__ == '__main__':
    main()
