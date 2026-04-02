"""
Cricket Player Detection, Tracking & Speed Estimation
Streamlit App - Main Entry Point
"""

import streamlit as st
import os
import tempfile
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="CricketTrack AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #0a0e1a; color: #e8eaf6; }

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #78909c;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #00e5ff;
}
.metric-label {
    font-size: 0.8rem;
    color: #546e7a;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-running { background: #1a3a2a; color: #00e676; border: 1px solid #00e676; }
.badge-done    { background: #1a2a3a; color: #00e5ff; border: 1px solid #00e5ff; }
.badge-error   { background: #3a1a1a; color: #ff5252; border: 1px solid #ff5252; }

div[data-testid="stSidebarContent"] {
    background: #0d1220;
    border-right: 1px solid #1e2d45;
}
.stButton > button {
    background: linear-gradient(135deg, #7c4dff, #00e5ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
    padding: 0.75rem;
    font-size: 1rem;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


def render_header():
    st.markdown('<div class="hero-title">🏏 CricketTrack AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multi-Object Detection · Persistent ID Tracking · Speed Estimation</div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")
        st.markdown("---")

        confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05,
                               help="YOLO detection threshold. Lower = more detections, more false positives.")
        iou_thresh = st.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.5, 0.05,
                               help="Non-max suppression IoU. Higher = fewer merged boxes.")
        track_buffer = st.slider("Track Buffer (frames)", 10, 100, 30, 5,
                                 help="How many frames a track survives without re-detection.")
        frame_skip = st.slider("Process Every N Frames", 1, 5, 1, 1,
                               help="1 = every frame. 2 = every other frame (faster, less smooth).")
        fps_out = st.slider("Output Video FPS", 10, 30, 20, 1)
        pixels_per_meter = st.number_input("Pixels per Meter (calibration)", 10.0, 500.0, 50.0, 5.0,
                                           help="Rough calibration: how many pixels equal 1 metre in the video. Cricket pitch = 20.12m ≈ adjust based on pitch length in pixels.")

        st.markdown("---")
        st.markdown("### 🎨 Visualization")
        show_trails = st.checkbox("Show Movement Trails", value=True)
        show_speed  = st.checkbox("Show Speed Labels", value=True)
        show_heatmap = st.checkbox("Generate Heatmap", value=False)
        show_birdseye = st.checkbox("Bird's-Eye Projection", value=False)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Stack:**  
        `YOLOv8` · `ByteTrack` · `OpenCV`  
        `Supervision` · `Streamlit`

        **Assignment:** Multi-Object Detection  
        & Persistent ID Tracking  
        """)

    return {
        "confidence": confidence,
        "iou_thresh": iou_thresh,
        "track_buffer": track_buffer,
        "frame_skip": frame_skip,
        "fps_out": fps_out,
        "pixels_per_meter": pixels_per_meter,
        "show_trails": show_trails,
        "show_speed": show_speed,
        "show_heatmap": show_heatmap,
        "show_birdseye": show_birdseye,
    }


def render_input_section():
    st.markdown("## 📥 Video Input")
    col1, col2 = st.columns([3, 1])

    with col1:
        video_url = st.text_input(
            "Paste a public video URL (YouTube, direct MP4, etc.)",
            placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
            help="YouTube URLs are downloaded via yt-dlp. Direct MP4 links are streamed."
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Or upload a file", type=["mp4", "avi", "mov", "mkv"])

    return video_url.strip() if video_url else None, uploaded


def render_metrics(stats: dict):
    st.markdown("### 📊 Pipeline Results")
    cols = st.columns(5)
    items = [
        ("Total Frames",       stats.get("total_frames", "-"),   "frames"),
        ("Frames Processed",   stats.get("processed_frames", "-"), "frames"),
        ("Unique Players",     stats.get("unique_ids", "-"),     "tracked"),
        ("Avg Speed (km/h)",   stats.get("avg_speed", "-"),      "km/h"),
        ("Peak Speed (km/h)",  stats.get("peak_speed", "-"),     "km/h"),
    ]
    for col, (label, val, unit) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    render_header()
    cfg = render_sidebar()

    video_url, uploaded_file = render_input_section()

    st.markdown("---")
    run_btn = st.button("🚀 Run Tracking Pipeline", use_container_width=True)

    if run_btn:
        if not video_url and not uploaded_file:
            st.error("Please provide a video URL or upload a file.")
            return

        from pipeline import run_pipeline
        import json

        with st.spinner("Setting up pipeline..."):
            progress_bar = st.progress(0)
            status_text  = st.empty()

        try:
            result = run_pipeline(
                video_url=video_url,
                uploaded_file=uploaded_file,
                cfg=cfg,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            st.success("✅ Processing complete!")
            render_metrics(result["stats"])

            st.markdown("---")
            st.markdown("## 🎬 Annotated Output Video")
            if os.path.exists(result["output_video"]):
                with open(result["output_video"], "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f, file_name="tracked_output.mp4", mime="video/mp4")
                st.video(result["output_video"])
            else:
                st.warning("Output video not found. Check logs.")

            if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
                st.markdown("## 🔥 Movement Heatmap")
                st.image(result["heatmap_path"], use_container_width=True)

            if result.get("birdseye_path") and os.path.exists(result["birdseye_path"]):
                st.markdown("## 🗺️ Bird's-Eye Projection")
                st.image(result["birdseye_path"], use_container_width=True)

            if result.get("speed_data"):
                st.markdown("## 🏃 Player Speed Statistics")
                import pandas as pd
                df = pd.DataFrame(result["speed_data"]).T
                df.index.name = "Player ID"
                df.columns = ["Max Speed (km/h)", "Avg Speed (km/h)", "Distance (m)", "Frames Tracked"]
                st.dataframe(df.style.highlight_max(subset=["Max Speed (km/h)"], color="#1a3a2a"), use_container_width=True)

            if result.get("report_path") and os.path.exists(result["report_path"]):
                st.markdown("## 📄 Technical Report")
                with open(result["report_path"]) as f:
                    st.markdown(f.read())

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

    else:
        # Landing info
        st.markdown("""
        ## How It Works

        1. **Paste a public video URL** (YouTube or direct MP4) or upload a file
        2. **Tune settings** in the sidebar (confidence, track buffer, speed calibration)
        3. **Click Run** — the pipeline will:
           - Download / read the video
           - Detect all players with **YOLOv8**
           - Track them across frames with **ByteTrack** (persistent IDs)
           - Estimate **speed in km/h** per player
           - Render an **annotated output video** with bounding boxes, IDs, and speeds
           - Optionally generate **heatmaps** and **bird's-eye view**

        ### Supported Video Sources
        - YouTube links (`youtube.com/watch?v=...`, `youtu.be/...`)
        - Direct `.mp4` / `.avi` / `.mov` links
        - Local file upload

        ### Speed Calibration
        Set **Pixels per Meter** in the sidebar to match your video. For a cricket broadcast:
        - Typical 22-yard pitch occupies ~X pixels — measure it and divide by 20.12.
        """)


if __name__ == "__main__":
    main()
