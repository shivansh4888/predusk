"""
utils/downloader.py — Download video from URL.

Uses imageio-ffmpeg bundled binary for any post-processing,
so no system ffmpeg / apt-get is required (works on Render native Python).
"""

import os
import re
import subprocess
import urllib.request
from pathlib import Path


def _ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def is_youtube_url(url: str) -> bool:
    patterns = [
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)",
        r"(https?://)?(www\.)?youtube\.com/shorts/",
    ]
    return any(re.search(p, url) for p in patterns)


def download_video(url: str, out_dir: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "input_video.mp4")

    if is_youtube_url(url):
        return _download_youtube(url, out_path)
    else:
        return _download_direct(url, out_path)


def _download_youtube(url: str, out_path: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    # Prefer H.264 (avc1) explicitly — avoids AV1/VP9 which OpenCV can't decode
    FORMAT = (
        "bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]"
        "/bestvideo[vcodec^=avc1][height<=720]+bestaudio"
        "/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
        "/best[height<=720][ext=mp4]"
        "/best[height<=720]"
        "/best"
    )

    ydl_opts = {
        "format": FORMAT,
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        # Force H.264 output via bundled ffmpeg
        "ffmpeg_location": str(Path(_ffmpeg_bin()).parent),
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
        "postprocessor_args": {
            "ffmpeg": [
                "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                "-acodec", "aac", "-movflags", "+faststart",
            ],
        },
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_path):
        candidates = sorted(
            Path(out_path).parent.glob("input_video*"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError("yt-dlp download failed — no output file found.")

    return out_path


def _download_direct(url: str, out_path: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CricketTracker/1.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

    if os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Downloaded file is empty: {url}")

    return out_path
