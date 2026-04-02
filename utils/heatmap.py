"""
utils/heatmap.py — Accumulate player positions and render movement heatmap.
"""

import cv2
import numpy as np
from pathlib import Path


class HeatmapGenerator:
    """
    Accumulates foot-point positions across all frames and renders
    a Gaussian-blurred heatmap overlaid on a blank canvas.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._canvas = np.zeros((height, width), dtype=np.float32)

    def update(self, cx: int, cy: int, weight: float = 1.0):
        """Add a player foot position."""
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self._canvas[cy, cx] += weight

    def save(self, path: str):
        """Render heatmap as PNG."""
        blurred = cv2.GaussianBlur(self._canvas, (0, 0), sigmaX=25, sigmaY=25)

        # Normalise
        mn, mx = blurred.min(), blurred.max()
        if mx > mn:
            norm = ((blurred - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(blurred, dtype=np.uint8)

        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Overlay on dark background
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        alpha_mask = (norm.astype(float) / 255.0)
        for c in range(3):
            bg[:, :, c] = (coloured[:, :, c] * alpha_mask + bg[:, :, c] * (1 - alpha_mask)).astype(np.uint8)

        # Labels
        cv2.putText(bg, "Player Movement Heatmap", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(path, bg)
        return path
