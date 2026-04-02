"""
pipeline.py — Core orchestration: download → detect → track → annotate → report

Uses imageio-ffmpeg (bundled binary) instead of system ffmpeg so it works
on Render's native Python runtime without any apt-get / Docker required.
"""

import os
import cv2
import subprocess
import tempfile
import numpy as np
from pathlib import Path

from utils.downloader import download_video
from utils.detector   import build_detector
from utils.tracker    import build_tracker, update_tracker
from utils.annotator  import Annotator
from utils.speed      import SpeedEstimator
from utils.heatmap    import HeatmapGenerator
from utils.birdseye   import BirdsEyeProjector
from utils.report     import generate_report


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _ffmpeg_bin() -> str:
    """Return path to the bundled ffmpeg binary from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"   # fall back to system ffmpeg if available


def _run_ffmpeg(*args):
    """Run ffmpeg with the bundled binary. Returns (returncode, stderr)."""
    cmd = [_ffmpeg_bin(), "-y"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stderr


def run_pipeline(
    video_url,
    uploaded_file,
    cfg: dict,
    progress_bar=None,
    status_text=None,
) -> dict:

    def update_ui(pct, msg):
        if progress_bar:  progress_bar.progress(pct)
        if status_text:   status_text.markdown(f"**{msg}**")

    # ── 1. Acquire video ──────────────────────────────────────────────────────
    update_ui(0.02, "📥 Acquiring video...")

    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_file.read())
        tmp.flush()
        video_path = tmp.name
    elif video_url:
        video_path = download_video(video_url, str(OUTPUT_DIR))
    else:
        raise ValueError("No video source provided.")

    # ── 2. Open video — transcode if codec unsupported (e.g. AV1) ─────────────
    def try_open(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        ok, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap if ok else None

    cap = try_open(video_path)

    if cap is None:
        update_ui(0.04, "⚙️ Transcoding to H.264 (AV1/VP9 not supported by OpenCV)...")
        transcoded = str(OUTPUT_DIR / "input_h264.mp4")
        rc, err = _run_ffmpeg(
            "-i", video_path,
            "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
            "-acodec", "aac", "-movflags", "+faststart",
            transcoded
        )
        if rc != 0 or not os.path.exists(transcoded):
            raise RuntimeError(
                f"Codec not supported and transcode failed.\nffmpeg stderr:\n{err}\n\n"
                "Try uploading an MP4 file directly instead of a URL."
            )
        video_path = transcoded
        cap = try_open(video_path)
        if cap is None:
            raise RuntimeError("Cannot open video even after transcoding.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_out  = str(OUTPUT_DIR / "tracked_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(raw_out, fourcc, cfg["fps_out"], (width, height))

    # ── 3. Build detector + tracker ───────────────────────────────────────────
    update_ui(0.05, "🧠 Loading YOLOv8 model...")
    detector = build_detector(confidence=cfg["confidence"], iou=cfg["iou_thresh"])

    update_ui(0.08, "🔄 Initialising ByteTrack...")
    tracker  = build_tracker(track_buffer=cfg["track_buffer"], frame_rate=int(src_fps))

    # ── 4. Helpers ────────────────────────────────────────────────────────────
    annotator   = Annotator(show_trails=cfg["show_trails"], show_speed=cfg["show_speed"])
    speed_est   = SpeedEstimator(pixels_per_meter=cfg["pixels_per_meter"], fps=src_fps)
    heatmap_gen = HeatmapGenerator(width, height) if cfg["show_heatmap"]   else None
    birdseye    = BirdsEyeProjector(width, height) if cfg["show_birdseye"] else None

    # ── 5. Frame loop ─────────────────────────────────────────────────────────
    frame_idx     = 0
    processed     = 0
    all_track_ids = set()
    update_ui(0.10, "🎬 Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % cfg["frame_skip"] != 0:
            writer.write(frame)
            continue

        processed += 1
        detections = detector.detect(frame)
        tracks     = update_tracker(tracker, detections, frame)

        for t in tracks:
            all_track_ids.add(t.track_id)
            speed_est.update(t.track_id, t.tlbr, frame_idx)
            cx = int((t.tlbr[0] + t.tlbr[2]) / 2)
            cy = int((t.tlbr[1] + t.tlbr[3]) / 2)
            if heatmap_gen: heatmap_gen.update(cx, cy)
            if birdseye:    birdseye.update(t.track_id, cx, cy)

        speeds    = speed_est.get_speeds()
        annotated = annotator.draw(frame.copy(), tracks, speeds)
        writer.write(annotated)

        pct = 0.10 + 0.85 * (frame_idx / max(total_frames, 1))
        if frame_idx % 30 == 0:
            update_ui(min(pct, 0.95), f"🎬 Frame {frame_idx}/{total_frames} — {len(all_track_ids)} unique IDs")

    cap.release()
    writer.release()

    # ── 6. Re-encode to H.264 for browser playback ────────────────────────────
    update_ui(0.96, "🔧 Re-encoding for browser...")
    final_video = str(OUTPUT_DIR / "tracked_output.mp4")
    rc, _ = _run_ffmpeg(
        "-i", raw_out,
        "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
        "-acodec", "aac", "-movflags", "+faststart",
        final_video
    )
    if rc != 0 or not os.path.exists(final_video):
        final_video = raw_out   # use raw mp4v as fallback

    # ── 7. Optional outputs ───────────────────────────────────────────────────
    heatmap_path  = None
    birdseye_path = None
    if heatmap_gen:
        heatmap_path = str(OUTPUT_DIR / "heatmap.png")
        heatmap_gen.save(heatmap_path)
    if birdseye:
        birdseye_path = str(OUTPUT_DIR / "birdseye.png")
        birdseye.save(birdseye_path)

    # ── 8. Stats ──────────────────────────────────────────────────────────────
    speed_data  = speed_est.get_all_stats()
    avg_speeds  = [v["avg_speed"] for v in speed_data.values() if v["avg_speed"] > 0]
    peak_speeds = [v["max_speed"] for v in speed_data.values() if v["max_speed"] > 0]

    stats = {
        "total_frames":     total_frames,
        "processed_frames": processed,
        "unique_ids":       len(all_track_ids),
        "avg_speed":        round(np.mean(avg_speeds), 1)  if avg_speeds  else 0,
        "peak_speed":       round(max(peak_speeds), 1)     if peak_speeds else 0,
    }

    # ── 9. Report ─────────────────────────────────────────────────────────────
    update_ui(0.98, "📄 Generating report...")
    report_path = str(OUTPUT_DIR / "technical_report.md")
    generate_report(report_path, stats, cfg, speed_data, video_url or "uploaded file")

    update_ui(1.0, "✅ Done!")

    return {
        "output_video":  final_video,
        "heatmap_path":  heatmap_path,
        "birdseye_path": birdseye_path,
        "speed_data":    speed_data,
        "report_path":   report_path,
        "stats":         stats,
    }
