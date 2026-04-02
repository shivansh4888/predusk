"""
utils/speed.py — Per-player speed estimation using bounding-box centre displacement.

Speed (km/h) = pixels_displaced_per_frame × pixels_per_meter_inv × fps × 3.6
"""

import numpy as np
from collections import defaultdict
from typing import Dict


class SpeedEstimator:
    """
    Estimates instantaneous and rolling-average speed for each track.

    Assumptions / Limitations:
      - Camera is assumed relatively static (pan/zoom not compensated).
      - pixels_per_meter is a user-provided calibration constant.
      - Speed is a rough estimate; more accurate with camera calibration.
    """

    SMOOTH_WINDOW = 10   # frames for rolling average
    MAX_PLAUSIBLE_KMH = 45.0  # clip implausible jumps (tracking artifacts)

    def __init__(self, pixels_per_meter: float = 50.0, fps: float = 25.0):
        self.ppm = pixels_per_meter          # pixels ≈ 1 metre
        self.fps = fps

        self._prev_centres: Dict[int, np.ndarray] = {}
        self._speed_history: Dict[int, list]       = defaultdict(list)
        self._current_speeds: Dict[int, float]     = {}

        # For final stats
        self._max_speeds: Dict[int, float]   = defaultdict(float)
        self._all_speeds: Dict[int, list]    = defaultdict(list)
        self._frame_counts: Dict[int, int]   = defaultdict(int)
        self._distances: Dict[int, float]    = defaultdict(float)

    def _centre(self, tlbr: np.ndarray) -> np.ndarray:
        return np.array([(tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2])

    def update(self, track_id: int, tlbr: np.ndarray, frame_idx: int):
        centre = self._centre(tlbr)
        self._frame_counts[track_id] += 1

        if track_id in self._prev_centres:
            px_dist  = np.linalg.norm(centre - self._prev_centres[track_id])
            metres   = px_dist / self.ppm
            kmh      = min(metres * self.fps * 3.6, self.MAX_PLAUSIBLE_KMH)

            self._speed_history[track_id].append(kmh)
            if len(self._speed_history[track_id]) > self.SMOOTH_WINDOW:
                self._speed_history[track_id].pop(0)

            smooth = float(np.mean(self._speed_history[track_id]))
            self._current_speeds[track_id] = smooth

            self._all_speeds[track_id].append(smooth)
            self._distances[track_id] += metres
            if smooth > self._max_speeds[track_id]:
                self._max_speeds[track_id] = smooth
        else:
            self._current_speeds[track_id] = 0.0

        self._prev_centres[track_id] = centre

    def get_speeds(self) -> Dict[int, float]:
        """Returns {track_id: current_speed_kmh}."""
        return dict(self._current_speeds)

    def get_all_stats(self) -> Dict[int, dict]:
        """Aggregate stats per player for the full video."""
        result = {}
        for tid in self._frame_counts:
            speeds = self._all_speeds[tid]
            result[tid] = {
                "max_speed":    round(self._max_speeds[tid], 1),
                "avg_speed":    round(float(np.mean(speeds)), 1) if speeds else 0.0,
                "distance":     round(self._distances[tid], 1),
                "frames":       self._frame_counts[tid],
            }
        return result
