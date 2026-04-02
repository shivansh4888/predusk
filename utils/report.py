"""
utils/report.py — Auto-generate the technical report in Markdown.
"""

import datetime


def generate_report(path: str, stats: dict, cfg: dict, speed_data: dict, video_source: str):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    speed_rows = ""
    for tid, s in sorted(speed_data.items()):
        speed_rows += f"| {tid} | {s['max_speed']} | {s['avg_speed']} | {s['distance']} | {s['frames']} |\n"

    report = f"""# CricketTrack AI — Technical Report
*Generated: {now}*

---

## 1. Objective
Multi-object detection and persistent ID tracking of players in sports footage,
with per-player speed estimation.

## 2. Video Source
```
{video_source}
```

## 3. Pipeline Summary

| Stage | Tool / Model |
|-------|-------------|
| Detection | YOLOv8-medium (COCO pretrained, class: person) |
| Tracking | ByteTrack (via `supervision`) |
| Speed Estimation | Optical-flow-free, bounding-box displacement × calibration constant |
| Annotation | OpenCV + custom Annotator |
| Optional | Heatmap (Gaussian blur), Bird's-Eye (homography) |

## 4. Model & Tracker Choices

**YOLOv8-medium** was selected because:
- State-of-the-art person detection accuracy in crowded scenes
- Real-time inference speed on CPU (≈ 8–15 FPS on modern CPU)
- Easily swappable to `yolov8s` (faster) or `yolov8x` (more accurate)
- No fine-tuning needed for general person detection

**ByteTrack** was selected because:
- Handles short occlusions without losing IDs (uses both high- and low-confidence detections)
- No re-ID appearance features needed (works with bounding boxes only)
- Proven on sports tracking benchmarks (MOT17, DanceTrack)

## 5. ID Consistency

ByteTrack assigns IDs by solving a two-step matching problem:
1. High-confidence detections matched to existing tracks via IoU (Kalman-predicted positions)
2. Low-confidence detections matched to unmatched tracks

A `track_buffer` of **{cfg['track_buffer']} frames** allows tracks to survive brief occlusions
or missed detections before being terminated.

## 6. Speed Estimation

```
speed_kmh = (pixel_displacement / pixels_per_meter) × fps × 3.6
```

- **pixels_per_meter calibration:** {cfg['pixels_per_meter']} px/m (user-set)
- Speed is smoothed over a 10-frame rolling window
- Capped at 45 km/h to discard tracking artefacts
- For a cricket broadcast: measure the 22-yard (20.12 m) pitch in pixels and divide

## 7. Pipeline Settings Used

| Parameter | Value |
|-----------|-------|
| Detection Confidence | {cfg['confidence']} |
| IOU Threshold (NMS) | {cfg['iou_thresh']} |
| Track Buffer (frames) | {cfg['track_buffer']} |
| Frame Skip | every {cfg['frame_skip']} frame(s) |
| Output FPS | {cfg['fps_out']} |
| Pixels per Metre | {cfg['pixels_per_meter']} |

## 8. Results

| Metric | Value |
|--------|-------|
| Total Frames | {stats.get('total_frames', '-')} |
| Frames Processed | {stats.get('processed_frames', '-')} |
| Unique Players Tracked | {stats.get('unique_ids', '-')} |
| Average Speed (km/h) | {stats.get('avg_speed', '-')} |
| Peak Speed (km/h) | {stats.get('peak_speed', '-')} |

### Per-Player Speed Stats

| Player ID | Max Speed (km/h) | Avg Speed (km/h) | Distance (m) | Frames Tracked |
|-----------|-----------------|-----------------|-------------|---------------|
{speed_rows}

## 9. Challenges Faced

- **Occlusion**: Players overlapping during close fielding. ByteTrack handles short occlusions
  but longer occlusions may cause ID switches.
- **Similar appearance**: Players wearing the same kit colour. ByteTrack relies only on
  position/IoU (no appearance features), so densely packed players with same kit can swap IDs.
- **Camera motion**: Panning/zooming artificially inflates speed estimates. A more robust
  solution would use optical flow to subtract camera motion before computing player velocity.
- **Scale changes**: Distant players produce small bounding boxes — confidence drops and
  tracks can terminate. Reduce confidence threshold or use a tiled detection approach.
- **Speed calibration**: Pixels-per-metre is user-supplied and only approximate without
  full camera intrinsic/extrinsic calibration.

## 10. Failure Cases

- IDs may switch when two players cross paths (common in cricket run-taking).
- Very fast bowler run-up with motion blur can cause missed detections for 1–3 frames.
- Boundary fielders who exit the frame and return receive new IDs.

## 11. Possible Improvements

1. **Re-ID appearance features** (e.g., OSNet, BoT-SORT) to recover IDs after long occlusions
2. **Optical flow camera motion compensation** for more accurate speed
3. **Camera calibration** (checkerboard or known pitch dimensions) for metric-accurate speed
4. **Fine-tuned YOLO on cricket footage** to reduce false positives (stumps, umpires, crowd)
5. **Kalman smoother** post-processing on trajectories for cleaner trails
6. **Team clustering** via jersey colour (HSV histogram per track) to separate batting/fielding

---
*CricketTrack AI — Assignment Submission*
"""

    with open(path, "w") as f:
        f.write(report)
