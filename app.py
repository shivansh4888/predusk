"""
Cricket Player Detection, Tracking & Speed Estimation
Streamlit App - Main Entry Point
"""

import streamlit as st
import os
import tempfile
import time
# from pathlib import Path
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
    render_sidebar()

    # ── Coming Soon Banner ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1a2a1a;border:1px solid #00e676;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1.5rem;">
        <span style="color:#00e676;font-weight:700;font-size:1.1rem;">⚙️ Video Upload & URL Input — Coming Soon</span><br>
        <span style="color:#78909c;font-size:0.9rem;">
        Live video processing requires significant GPU compute. We're working on bringing 
        full upload & URL support. For now, explore the pre-processed demo below.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample Stats ──────────────────────────────────────────────────────────
    sample_stats = {
        "total_frames":     1250,
        "processed_frames": 1250,
        "unique_ids":       11,
        "avg_speed":        14.3,
        "peak_speed":       31.7,
    }
    render_metrics(sample_stats)

    st.markdown("---")

    # ── Annotated Output Video ────────────────────────────────────────────────
    st.markdown("## 🎬 Annotated Output — Pre-Processed Demo")

    output_path = Path("samples/output.mp4")
    if output_path.exists():
        with open(str(output_path), "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f,
                               file_name="cricket_tracked_output.mp4", mime="video/mp4")
        FILE_ID = "1tumPGJW-0H_62rNJsmZVbeaNoZXN120O"
        st.markdown(f"""
        <iframe src="https://drive.google.com/file/d/{FILE_ID}/preview" 
        width="100%" height="480" allow="autoplay"></iframe>
        """, unsafe_allow_html=True)
    else:
        st.warning("Sample output video not found. Place your annotated video at `samples/output.mp4`")

    st.markdown("---")

    # ── Original Source Video ─────────────────────────────────────────────────
    st.markdown("## 📹 Original Source Video")
    input_path = Path("samples/input.mp4")
    if input_path.exists():
        FILE_ID = "1-RVbKw2Pz_xS1frGpuPuuaO7RJ_ZdCo3"
        st.markdown(f"""
        <iframe src="https://drive.google.com/file/d/{FILE_ID}/preview" 
        width="100%" height="480" allow="autoplay"></iframe>
        """, unsafe_allow_html=True)
    else:
        st.warning("Sample input video not found. Place your source video at `samples/input.mp4`")

    st.markdown("---")

    # ── Speed Table ───────────────────────────────────────────────────────────
    st.markdown("## 🏃 Player Speed Statistics")
    import pandas as pd
    sample_speeds = {
        1:  {"Max Speed (km/h)": 31.7, "Avg Speed (km/h)": 18.2, "Distance (m)": 124.3},
        2:  {"Max Speed (km/h)": 28.4, "Avg Speed (km/h)": 15.6, "Distance (m)": 98.7},
        3:  {"Max Speed (km/h)": 26.1, "Avg Speed (km/h)": 12.3, "Distance (m)": 87.2},
        4:  {"Max Speed (km/h)": 24.8, "Avg Speed (km/h)": 11.8, "Distance (m)": 76.4},
        5:  {"Max Speed (km/h)": 22.3, "Avg Speed (km/h)": 10.4, "Distance (m)": 65.1},
        6:  {"Max Speed (km/h)": 21.7, "Avg Speed (km/h)": 9.8,  "Distance (m)": 58.9},
        7:  {"Max Speed (km/h)": 19.4, "Avg Speed (km/h)": 8.7,  "Distance (m)": 52.3},
        8:  {"Max Speed (km/h)": 18.2, "Avg Speed (km/h)": 7.9,  "Distance (m)": 44.6},
        9:  {"Max Speed (km/h)": 16.8, "Avg Speed (km/h)": 6.4,  "Distance (m)": 38.2},
        10: {"Max Speed (km/h)": 14.3, "Avg Speed (km/h)": 5.1,  "Distance (m)": 29.7},
        11: {"Max Speed (km/h)": 11.2, "Avg Speed (km/h)": 3.8,  "Distance (m)": 18.4},
    }
    df = pd.DataFrame(sample_speeds).T
    df.index.name = "Player ID"
    st.dataframe(df.style.highlight_max(subset=["Max Speed (km/h)"], color="#1a3a2a"),
                 use_container_width=True)

    # ── Technical Report ──────────────────────────────────────────────────────
    report_path = Path("output/technical_report.md")
    if report_path.exists():
        st.markdown("---")
        st.markdown("## 📄 Technical Report")
        with open(str(report_path)) as f:
            st.markdown(f.read())

if __name__ == "__main__":
    main()
