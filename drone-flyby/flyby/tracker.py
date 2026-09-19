"""Remember every object seen so far and carry it along with the ground.

The response has to cover the whole source frame, but the camera only shows
part of it. Each track is an object we believe is on the ground: its box is
moved every frame by the ground motion model, corrected whenever the object is
seen again, and its class is decided by a confidence-weighted vote across all
sightings (sightings at higher zoom count more).
"""

import itertools
from dataclasses import dataclass, field
from typing import List

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES
from flyby.detector import Detection
from flyby.motion import GroundMotion

LEVEL_WEIGHT = {0: 0.6, 1: 0.85, 2: 1.0}
LOOK_PX = 24          # an object is "properly seen" once it spanned this many view pixels
MISS_VISIBLE_PX = 12  # only count a miss if the object should have been this visible
MERGE_IOU = 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    votes: np.ndarray = field(default_factory=lambda: np.zeros(len(OBJECT_CLASSES)))
    hits: int = 0
    misses: int = 0
    last_seen: int = 0
    last_centre: np.ndarray = None      # where it was last observed uncut
    last_centre_frame: int = -1
    best_apparent: float = 0.0          # largest view size it was seen at
    best_level: int = -1                # highest zoom level it was seen at

    @property
    def fully_zoomed(self):
        """Seen big enough to name, or already at the closest zoom there is."""
        return self.best_apparent >= LOOK_PX or self.best_level == 2

    @property
    def centre(self):
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def strength(self):
        return float(self.votes.max())

    @property
    def purity(self):
        total = self.votes.sum()
        return float(self.votes.max() / total) if total > 0 else 0.0

    def look_priority(self, frame):
        """How much a closer look at this object would be worth."""
        priority = 0.0
        if not self.fully_zoomed:
            priority += 1.0                 # never seen big enough to name
        if (self.strength < 0.8 or self.purity < 0.6) and self.best_level < 2:
            priority += 0.7                 # unsure what it is, and a closer look exists
        if frame - self.last_seen > 12 and (self.box[2] - self.box[0]) < 80:
            priority += 0.3                 # small and stale: position may drift
        return priority


class Tracker:
    def __init__(self):
        self.motion = GroundMotion()
        self.tracks: List[Track] = []
        self.ids = itertools.count()
        self.frame = None

    def advance_to(self, frame):
        if self.frame is not None and frame > self.frame:
            steps = frame - self.frame
            for track in self.tracks:
                track.box = self.motion.advance(track.box, steps)
            self.tracks = [t for t in self.tracks
                           if t.box[1] < IMAGE_HEIGHT and t.box[3] > 0
                           and t.box[0] < IMAGE_WIDTH and t.box[2] > 0]
        self.frame = frame

    def update(self, detections: List[Detection], region, level, frame):
        pairs = []
        for d_index, detection in enumerate(detections):
            for t_index, track in enumerate(self.tracks):
                overlap = iou(detection.box, track.box)
                size = np.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1])
                distance = np.hypot(*(detection.centre - track.centre))
                gate = max(0.6 * size, 12 * 4 / 2 ** level)
                if overlap > 0.2 or distance < gate:
                    pairs.append((overlap - distance / (gate * 10), d_index, t_index))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _, d_index, t_index in pairs:
            if d_index in used_d or t_index in used_t:
                continue
            used_d.add(d_index)
            used_t.add(t_index)
            self._apply(self.tracks[t_index], detections[d_index], frame)

        for d_index, detection in enumerate(detections):
            if d_index not in used_d:
                track = Track(next(self.ids), detection.box.copy())
                self._apply(track, detection, frame)
                self.tracks.append(track)

        factor = 4 / 2 ** level
        survivors = []
        for t_index, track in enumerate(self.tracks):
            if t_index not in used_t:
                inside = (track.box[0] >= region[0] and track.box[1] >= region[1]
                          and track.box[2] <= region[2] and track.box[3] <= region[3])
                apparent = np.sqrt((track.box[2] - track.box[0]) * (track.box[3] - track.box[1])) / factor
                if inside and apparent >= MISS_VISIBLE_PX and track.last_seen != frame:
                    track.misses += 1
            if track.misses >= max(2, track.hits // 2):
                continue
            survivors.append(track)
        self.tracks = self._merge(survivors)

    def _apply(self, track: Track, detection: Detection, frame):
        if not detection.cut_off:
            centre = detection.centre
            if track.last_centre is not None and track.last_centre_frame < frame:
                self.motion.observe(frame, track.last_centre, centre,
                                    frame - track.last_centre_frame)
            track.last_centre, track.last_centre_frame = centre, frame
            track.box = detection.box.copy()
        track.votes[detection.class_id] += detection.score * LEVEL_WEIGHT[detection.level]
        track.best_apparent = max(track.best_apparent, detection.apparent_px)
        track.best_level = max(track.best_level, detection.level)
        track.hits += 1
        track.misses = 0
        track.last_seen = frame

    @staticmethod
    def _merge(tracks):
        tracks = sorted(tracks, key=lambda t: -t.strength)
        kept: List[Track] = []
        for track in tracks:
            for other in kept:
                if iou(track.box, other.box) > MERGE_IOU:
                    other.votes += track.votes
                    other.hits += track.hits
                    other.best_apparent = max(other.best_apparent, track.best_apparent)
                    other.best_level = max(other.best_level, track.best_level)
                    break
            else:
                kept.append(track)
        return kept

    def annotations(self, frame):
        """(class name, source box, confidence) for every live track."""
        out = []
        for track in self.tracks:
            if track.votes.sum() <= 0:
                continue
            age = frame - track.last_seen
            confidence = ((1 - np.exp(-track.strength / 0.8)) * (0.5 + 0.5 * track.purity)
                          * 0.985 ** age)
            out.append((OBJECT_CLASSES[int(track.votes.argmax())], track.box,
                        float(np.clip(confidence, 0.001, 0.999))))
        return out
