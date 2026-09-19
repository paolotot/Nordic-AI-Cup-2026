"""How the ground moves between two consecutive source frames.

Fitted on helsinki, one frame of flight is a slide plus a slight zoom:

    dx = ax + cx * (x - 1920) / 1920
    dy = ay + cy * (y - 1080) / 1080

with ax ~ 0, cx ~ 13.6, ay ~ 65.6, cy ~ 14.4 source pixels (the zoom also grows
boxes by ~0.7% wide / ~1.3% tall per frame). Residuals of that fit are under
1 px, but the speed wandered from 64 to 70 px/frame over 25 frames, so the
parameters are re-estimated online from re-sighted objects, pulled toward the
helsinki prior when there is little evidence.
"""

from collections import deque

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH

HALF_W, HALF_H = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
PRIOR_X = np.array([0.0, 13.6])     # (a, c) for dx
PRIOR_Y = np.array([65.6, 14.4])    # (a, c) for dy
# Prior strength, in "observations worth". Offsets wander over a flight, so
# their prior is weak; the zoom terms are geometry and change slowly.
PRIOR_WEIGHT = np.array([2.0, 20.0])
HISTORY = 60                        # recent observations used for the fit
FORGET = 0.97                       # weight decay per frame of observation age


class GroundMotion:
    def __init__(self):
        self.x = PRIOR_X.copy()
        self.y = PRIOR_Y.copy()
        # (frame, x, y, dx per frame, dy per frame)
        self.observations = deque(maxlen=HISTORY)

    def velocity(self, x, y):
        """Per-frame displacement of ground at source point (x, y)."""
        return (self.x[0] + self.x[1] * (x - HALF_W) / HALF_W,
                self.y[0] + self.y[1] * (y - HALF_H) / HALF_H)

    def advance(self, box, frames):
        """Move a source-pixel box forward by a number of frames."""
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        grow_x, grow_y = 1 + self.x[1] / HALF_W, 1 + self.y[1] / HALF_H
        for _ in range(frames):
            dx, dy = self.velocity(cx, cy)
            cx, cy = cx + dx, cy + dy
            w, h = w * grow_x, h * grow_y
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def observe(self, frame, before, after, frames):
        """Record an object's centre moving from ``before`` to ``after`` (source
        pixels) over ``frames`` frames, then refit."""
        if frames <= 0:
            return
        mid = (np.asarray(before) + np.asarray(after)) / 2
        step = (np.asarray(after) - np.asarray(before)) / frames
        self.observations.append((frame, mid[0], mid[1], step[0], step[1]))
        self._refit(frame)

    def _refit(self, now):
        data = np.array(self.observations)
        weights = FORGET ** (now - data[:, 0])
        for axis, position, target, prior, half in (
                ('x', data[:, 1], data[:, 3], PRIOR_X, HALF_W),
                ('y', data[:, 2], data[:, 4], PRIOR_Y, HALF_H)):
            design = np.stack([np.ones_like(position), (position - half) / half], axis=1)
            # Weighted least squares with a ridge pull toward the prior.
            lhs = design.T @ (design * weights[:, None]) + np.diag(PRIOR_WEIGHT)
            rhs = design.T @ (target * weights) + PRIOR_WEIGHT * prior
            fit = np.linalg.solve(lhs, rhs)
            setattr(self, axis, fit)
