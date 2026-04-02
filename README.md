# 🏏 CricketTrack AI
**Multi-Object Detection · Persistent ID Tracking · Speed Estimation**

> Assignment: Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
> Stack: YOLOv8 · ByteTrack · OpenCV · Streamlit

---

## 🖼️ What It Does

| Feature | Detail |
|---------|--------|
| **Person Detection** | YOLOv8-medium, COCO pretrained (`person` class) |
| **Tracking** | ByteTrack via `supervision` — persistent IDs across full video |
| **Speed Estimation** | km/h per player from bounding-box displacement + calibration |
| **Annotated Video** | Coloured boxes, ID labels, speed overlays, movement trails |
| **Heatmap** | Gaussian-blurred movement density map |
| **Bird's-Eye View** | Top-down trajectory projection via homography |
| **Technical Report** | Auto-generated Markdown with all stats and analysis |

---

## ⚡ Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cricket-tracker-ai.git
cd cricket-tracker-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `torch` will install the CPU version by default. For GPU support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Deploy to Render

### One-click deploy
1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**

The `render.yaml` sets:
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- 5 GB persistent disk at `/app/output`

### Manual Render setup
| Field | Value |
|-------|-------|
| Environment | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |

---

## 🐳 Docker Deployment

```bash
docker build -t cricket-tracker .
docker run -p 8501:8501 cricket-tracker
```

---

## 📖 How to Use

1. **Paste a public video URL** — YouTube links and direct `.mp4` URLs are supported
2. **Or upload** a video file (MP4, AVI, MOV, MKV — up to 500 MB)
3. **Tune settings** in the sidebar:
   - Detection confidence
   - Track buffer (occlusion tolerance)
   - Frame skip (speed vs. smoothness trade-off)
   - Pixels per metre (speed calibration)
4. Click **🚀 Run Tracking Pipeline**
5. Download or preview the annotated video

---

## ⚙️ Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| Detection Confidence | 0.40 | YOLO detection threshold |
| IOU Threshold | 0.50 | Non-max suppression IoU |
| Track Buffer | 30 frames | Frames a lost track survives |
| Frame Skip | 1 | Process every Nth frame |
| Output FPS | 20 | Output video frame rate |
| Pixels per Metre | 50 | Speed calibration constant |
| Show Trails | ✅ | Draw trajectory history |
| Show Speed | ✅ | Overlay km/h on each player |
| Generate Heatmap | ❌ | Density heatmap image |
| Bird's-Eye View | ❌ | Top-down trajectory projection |

### Speed Calibration
For a cricket broadcast:
1. Pause on a frame with the full pitch visible
2. Measure the 22-yard pitch in pixels
3. `pixels_per_meter = pitch_pixels / 20.12`

---

## 🧠 Technical Architecture

```
Input URL / File
      │
      ▼
  Downloader (yt-dlp / urllib)
      │
      ▼
  OpenCV VideoCapture
      │ frame-by-frame
      ▼
  YOLOv8 Detector ──► [x1,y1,x2,y2, conf, class=person]
      │
      ▼
  ByteTrack ──────────► [track_id, tlbr]
      │
      ├──► SpeedEstimator (px displacement → km/h)
      ├──► HeatmapGenerator (optional)
      ├──► BirdsEyeProjector (optional)
      │
      ▼
  Annotator (boxes + IDs + speed + trails)
      │
      ▼
  VideoWriter → FFmpeg H.264 re-encode
      │
      ▼
  Streamlit UI (download + preview + stats + report)
```

---

## 🏗️ Project Structure

```
cricket-tracker-ai/
├── app.py                  # Streamlit UI entry point
├── pipeline.py             # Main orchestration pipeline
├── utils/
│   ├── downloader.py       # Video acquisition (yt-dlp, urllib)
│   ├── detector.py         # YOLOv8 wrapper
│   ├── tracker.py          # ByteTrack wrapper
│   ├── annotator.py        # OpenCV annotation (boxes, IDs, trails, speed)
│   ├── speed.py            # Per-player speed estimation
│   ├── heatmap.py          # Movement density heatmap
│   ├── birdseye.py         # Top-down homographic projection
│   └── report.py           # Auto-generate Markdown technical report
├── output/                 # Generated files (gitignored)
├── .streamlit/
│   └── config.toml         # Streamlit theme + server config
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## 📊 Sample Output

**Annotated Video:**  
Each player gets a unique coloured bounding box, persistent ID, real-time speed (km/h), and a fading movement trail.

**Speed Table:**
| Player | Max Speed (km/h) | Avg Speed (km/h) | Distance (m) |
|--------|-----------------|-----------------|-------------|
| #1 | 28.3 | 14.1 | 87.2 |
| #2 | 32.7 | 18.4 | 112.6 |

---

## ⚠️ Known Limitations

- **Camera motion** inflates speed estimates — no optical flow compensation
- **ID switches** can occur when two players cross paths at close proximity
- **Speed requires calibration** — defaults are approximate
- **YouTube download** may fail for age-restricted or region-blocked videos
- **No appearance re-ID** — long occlusions (5+ seconds) may cause new IDs

---

## 🔭 Possible Improvements

- BoT-SORT / StrongSORT for appearance-based re-ID
- Camera motion compensation with optical flow
- Full camera calibration for metric-accurate speed
- Fine-tuned YOLO on cricket-specific dataset
- Team clustering by jersey colour
- Count-over-time chart per player

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection |
| `supervision` | ByteTrack, detection utilities |
| `opencv-python-headless` | Video I/O, annotation |
| `yt-dlp` | YouTube video download |
| `streamlit` | Web UI |
| `torch` | YOLO backend |
| `numpy` | Numerical ops |
| `pandas` | Stats display |

---

## 📄 Author
Shivansh Singh