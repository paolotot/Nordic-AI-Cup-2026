"""Where to point the camera next.

``LookAndZoom`` (strategy C2 from tools/camera_sim.py, the simulator's best):
look at the full frame, then zoom to Level 1 on the cluster of objects most
worth a closer look, on to Level 2 if they are still too small to name, and
back out. ``Tour`` (B2, the simple fallback) cycles through six Level-1 views.

Every command is built from the request's own camera_constraints, so it is
legal by construction.
"""

import math
from typing import Optional, Tuple

import numpy as np

from dtos import FULL_FRAME_CENTER, SOURCE_REGION_SIZES
from flyby.tracker import LOOK_PX, Tracker

Command = Tuple[int, int, int]


def legal(constraints, here: Command, level, x, y) -> Optional[Command]:
    """Closest legal command toward (level, x, y), or None if that level is out."""
    if level not in constraints.allowed_resolution_levels:
        return None
    if level == 0:
        return (0, *FULL_FRAME_CENTER)
    bounds = constraints.bounds_for_level(level)
    if bounds is None:
        return None
    lo = np.array([bounds.minimum_center_x, bounds.minimum_center_y], float)
    hi = np.array([bounds.maximum_center_x, bounds.maximum_center_y], float)
    current = np.array(here[1:], float)
    start = np.clip(current, lo, hi)
    goal = np.clip(np.array([x, y], float), lo, hi)
    # The evaluator refuses only distance > limit, so the limit itself is legal
    # (a corner Level-2 view is exactly 551 px from the nearest Level-1 centre).
    limit = constraints.maximum_center_delta
    if np.hypot(*(start - current)) > limit:
        return None
    if np.hypot(*(goal - current)) > limit:
        # Furthest point on start->goal within the limit (binary search).
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = (low + high) / 2
            if np.hypot(*(start + mid * (goal - start) - current)) <= limit:
                low = mid
            else:
                high = mid
        goal = start + low * (goal - start)
    # Integer centres: of the four floor/ceil roundings, take the one closest
    # to the goal that is still in bounds and within the limit.
    options = [(int(ox), int(oy)) for ox in {math.floor(goal[0]), math.ceil(goal[0])}
               for oy in {math.floor(goal[1]), math.ceil(goal[1])}
               if lo[0] <= ox <= hi[0] and lo[1] <= oy <= hi[1]
               and math.hypot(ox - current[0], oy - current[1]) <= limit]
    if not options:
        return None
    best = min(options, key=lambda o: math.hypot(o[0] - goal[0], o[1] - goal[1]))
    return (level, *best)


class LookAndZoom:
    name = 'look-and-zoom'

    def __init__(self, max_frames_zoomed: int = 6):
        self.max_frames_zoomed = max_frames_zoomed
        self.frames_zoomed = 0

    def next_view(self, request, tracker: Tracker, frame) -> Optional[Command]:
        view, constraints = request.view, request.camera_constraints
        here = (view.resolution_level, view.center_x, view.center_y)
        self.frames_zoomed = 0 if here[0] == 0 else self.frames_zoomed + 1

        targets = []
        for track in tracker.tracks:
            priority = track.look_priority(frame)
            if priority <= 0:
                continue
            centre = track.centre + np.array(tracker.motion.velocity(*track.centre))
            size = math.sqrt(max(1.0, (track.box[2] - track.box[0]) * (track.box[3] - track.box[1])))
            # Worth Level 2 only if Level 1 would still show it too small.
            needs_level2 = not track.fully_zoomed and size / 2 < LOOK_PX
            targets.append((centre, priority, needs_level2))

        if here[0] == 0:
            if not targets:
                return legal(constraints, here, 0, 0, 0)
            centre = self._best_centre(targets, 1)
            return legal(constraints, here, 1, *centre)

        too_long = self.frames_zoomed >= self.max_frames_zoomed
        width, height = SOURCE_REGION_SIZES[here[0]]
        if here[0] == 1:
            inside = [t for t in targets
                      if abs(t[0][0] - here[1]) < width / 2 and abs(t[0][1] - here[2]) < height / 2
                      and t[2]]
            if inside and not too_long:
                return legal(constraints, here, 2, *self._best_centre(inside, 2))
            return legal(constraints, here, 0, 0, 0)

        # Level 2: hop to another nearby target that needs full detail, else back out.
        reach = constraints.maximum_center_delta
        nearby = [t for t in targets
                  if t[2] and np.hypot(t[0][0] - here[1], t[0][1] - here[2]) < reach]
        if nearby and not too_long:
            return legal(constraints, here, 2, *self._best_centre(nearby, 2))
        return legal(constraints, here, 1, here[1], here[2])

    @staticmethod
    def _best_centre(targets, level):
        """The target position whose window (with a margin) covers the most priority."""
        width, height = SOURCE_REGION_SIZES[level]
        half_w, half_h = width / 2 * 0.85, height / 2 * 0.85
        best, best_score = targets[0][0], -1.0
        for candidate, _, _ in targets:
            covered = [t for t in targets
                       if abs(t[0][0] - candidate[0]) < half_w and abs(t[0][1] - candidate[1]) < half_h]
            score = sum(t[1] for t in covered)
            if score > best_score:
                # Centre on the covered group rather than on one member.
                best = np.mean([t[0] for t in covered], axis=0)
                best_score = score
        return best


class Tour:
    name = 'tour'
    WAYPOINTS = [(1, 960, 540), (1, 1920, 540), (1, 2880, 540),
                 (1, 2880, 1620), (1, 1920, 1620), (1, 960, 1620)]

    def __init__(self):
        self.index = 0

    def next_view(self, request, tracker, frame) -> Optional[Command]:
        view = request.view
        here = (view.resolution_level, view.center_x, view.center_y)
        if here == self.WAYPOINTS[self.index]:
            self.index = (self.index + 1) % len(self.WAYPOINTS)
        return legal(request.camera_constraints, here, *self.WAYPOINTS[self.index])


def load_policy(name: str):
    return {'look-and-zoom': LookAndZoom, 'tour': Tour}[name]()
