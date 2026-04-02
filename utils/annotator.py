"""
utils/annotator.py — Draw bounding boxes, IDs, speed labels, and movement trails.
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List

from utils.tracker import Track

# Colour palette — one distinct colour per track ID
_PALETTE = [
    (0, 229, 255),   # cyan
    (124, 77, 255),  # violet
    (0, 230, 118),   # green
    (255, 171, 0),   # amber
    (255, 82, 82),   # red
    (64, 196, 255),  # sky
    (255, 214, 0),   # yellow
    (0, 176, 255),   # blue
    (105, 240, 174), # mint
    (255, 138, 101), # orange
]


def _color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


class Annotator:
    """Draws all visual elements onto a frame."""

    TRAIL_LEN = 40   # frames kept in trail history

    def __init__(self, show_trails: bool = True, show_speed: bool = True):
        self.show_trails = show_trails
        self.show_speed  = show_speed
        self._trails: Dict[int, list] = defaultdict(list)

    def draw(self, frame: np.ndarray, tracks: List[Track], speeds: Dict[int, float]) -> np.ndarray:
        """
        Annotate frame with:
          - Coloured bounding boxes
          - Track ID label
          - Speed label (km/h)
          - Trajectory trail
        """
        # Update trails
        for t in tracks:
            cx = int((t.tlbr[0] + t.tlbr[2]) / 2)
            cy = int(t.tlbr[3])               # bottom-centre
            self._trails[t.track_id].append((cx, cy))
            if len(self._trails[t.track_id]) > self.TRAIL_LEN:
                self._trails[t.track_id].pop(0)

        # Draw trails first (under boxes)
        if self.show_trails:
            for t in tracks:
                pts = self._trails[t.track_id]
                color = _color(t.track_id)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(frame, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)

        # Draw bounding boxes and labels
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            color = _color(t.track_id)

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label background + text
            speed_val = speeds.get(t.track_id, 0.0)
            label = f"#{t.track_id}"
            if self.show_speed and speed_val > 0.5:
                label += f"  {speed_val:.1f} km/h"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # Pill background
            lx, ly = x1, max(y1 - th - 8, 0)
            cv2.rectangle(frame, (lx, ly), (lx + tw + 8, ly + th + 6), color, -1)
            cv2.putText(frame, label, (lx + 4, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Corner watermark
        cv2.putText(frame, "CricketTrack AI", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame
