"""
utils/detector.py — YOLOv8 person detector wrapper using Ultralytics

FIX: Model is loaded once via @st.cache_resource (or a module-level singleton
     when running outside Streamlit) to prevent repeated downloads on reruns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# ── Module-level singleton so the model is only loaded once per process ──────
_MODEL_CACHE: dict = {}


def _load_yolo(model_name: str):
    """Load (or return cached) YOLO model. Thread-safe enough for Streamlit."""
    if model_name not in _MODEL_CACHE:
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        _MODEL_CACHE[model_name] = YOLO(model_name)
    return _MODEL_CACHE[model_name]


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Detection:
    """Single detection result."""
    xyxy: np.ndarray      # [x1, y1, x2, y2]
    confidence: float
    class_id: int


class YOLODetector:
    """
    Wraps YOLOv8 for person detection.
    Uses the pretrained COCO model; class 0 = 'person'.
    Model weights are downloaded once and cached for the lifetime of the process.
    """

    PERSON_CLASS = 0

    def __init__(self, model_name: str = "yolov8m.pt", confidence: float = 0.4, iou: float = 0.5):
        self.model_name = model_name
        self.confidence = confidence
        self.iou        = iou
        self._device    = _get_device()
        # Eagerly load so the download happens at init time, not inside the frame loop
        _load_yolo(model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR frame. Returns list of Detection."""
        model = _load_yolo(self.model_name)
        results = model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS],
            verbose=False,
            device=self._device,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0].cpu().numpy())
                cls  = int(box.cls[0].cpu().numpy())
                detections.append(Detection(xyxy=xyxy, confidence=conf, class_id=cls))
        return detections


def build_detector(confidence: float = 0.4, iou: float = 0.5, model: str = "yolov8m.pt") -> YOLODetector:
    """Factory — swap model variant here (yolov8s = faster, yolov8x = more accurate)."""
    return YOLODetector(model_name=model, confidence=confidence, iou=iou)
