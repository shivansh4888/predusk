"""
utils/tracker.py — ByteTrack multi-object tracker via supervision library
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from utils.detector import Detection


@dataclass
class Track:
    """Active track with assigned persistent ID."""
    track_id: int
    tlbr: np.ndarray        # [x1, y1, x2, y2]
    confidence: float


class ByteTrackerWrapper:
    """
    Wraps supervision's ByteTrack implementation.
    supervision >= 0.21 ships ByteTrack natively.
    """

    def __init__(self, track_buffer: int = 30, frame_rate: int = 25):
        import supervision as sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate,
        )

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        import supervision as sv

        if not detections:
            return []

        xyxy  = np.array([d.xyxy       for d in detections], dtype=float)
        confs = np.array([d.confidence for d in detections], dtype=float)
        cls   = np.array([d.class_id   for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=cls,
        )

        tracked = self.tracker.update_with_detections(sv_dets)

        results = []
        if len(tracked) > 0:
            for i in range(len(tracked)):
                results.append(Track(
                    track_id=int(tracked.tracker_id[i]),
                    tlbr=tracked.xyxy[i].astype(float),
                    confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                ))

        return results


def build_tracker(track_buffer: int = 30, frame_rate: int = 25) -> ByteTrackerWrapper:
    return ByteTrackerWrapper(track_buffer=track_buffer, frame_rate=frame_rate)


def update_tracker(tracker: ByteTrackerWrapper, detections: List[Detection], frame: np.ndarray) -> List[Track]:
    return tracker.update(detections, frame)
