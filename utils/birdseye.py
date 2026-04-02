"""
utils/birdseye.py — Simple homographic top-down projection of player trajectories.

Without a full camera calibration rig, we apply a fixed perspective warp
using approximate cricket field keypoints. Users can override via config.
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


# Default top-view canvas size (metres × scale)
TOP_VIEW_W = 800
TOP_VIEW_H = 600
SCALE      = 10   # pixels per metre in top-view


class BirdsEyeProjector:
    """
    Accumulates track centres in screen space and plots them
    on an approximate top-down view using a default homography.
    """

    def __init__(self, src_width: int, src_height: int):
        self.src_w = src_width
        self.src_h = src_height
        self._positions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._H = self._default_homography()

    def _default_homography(self) -> np.ndarray:
        """
        Approximate perspective transform from broadcast view to top-down.
        Source points: four corners of a typical cricket pitch in frame.
        Destination: orthographic top-view.
        These are rough defaults — improve with camera calibration.
        """
        W, H = self.src_w, self.src_h
        src = np.float32([
            [W * 0.3, H * 0.55],   # near-left crease
            [W * 0.7, H * 0.55],   # near-right crease
            [W * 0.6, H * 0.30],   # far-right crease
            [W * 0.4, H * 0.30],   # far-left crease
        ])
        dst = np.float32([
            [150, 500],
            [650, 500],
            [650, 100],
            [150, 100],
        ])
        H_mat, _ = cv2.findHomography(src, dst)
        return H_mat

    def update(self, track_id: int, cx: int, cy: int):
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self._H)
        wx, wy = int(warped[0, 0, 0]), int(warped[0, 0, 1])
        self._positions[track_id].append((wx, wy))

    def save(self, path: str):
        canvas = np.zeros((TOP_VIEW_H, TOP_VIEW_W, 3), dtype=np.uint8)

        # Draw pitch outline (approximate)
        cv2.rectangle(canvas, (150, 100), (650, 500), (60, 80, 60), -1)
        cv2.rectangle(canvas, (150, 100), (650, 500), (100, 140, 100), 2)
        cv2.line(canvas, (150, 300), (650, 300), (150, 200, 150), 1)  # pitch centre

        colours = [
            (0, 229, 255), (124, 77, 255), (0, 230, 118),
            (255, 171, 0), (255, 82, 82),  (64, 196, 255),
        ]

        for i, (tid, pts) in enumerate(self._positions.items()):
            col = colours[tid % len(colours)]
            for j in range(1, len(pts)):
                p1 = pts[j - 1]
                p2 = pts[j]
                if all(0 <= v < (TOP_VIEW_W if k % 2 == 0 else TOP_VIEW_H) for k, v in enumerate([*p1, *p2])):
                    cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
            if pts:
                lx, ly = pts[-1]
                if 0 <= lx < TOP_VIEW_W and 0 <= ly < TOP_VIEW_H:
                    cv2.circle(canvas, (lx, ly), 5, col, -1)
                    cv2.putText(canvas, f"#{tid}", (lx + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        cv2.putText(canvas, "Bird's-Eye Trajectory View", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "(Approximate — calibrate src points for accuracy)", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imwrite(path, canvas)
        return path
