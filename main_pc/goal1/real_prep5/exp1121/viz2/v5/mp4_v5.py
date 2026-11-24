"""mp4_v5.py - viz2 v5 (win2 2026-11-24): keep an MP4 near the size limit by quality only. shrink(path): if the file is over limit_mb
(12 MB, i.e. well over the 10 MB target) it is re-encoded (H.264, same size, same frame rate) with crf 23, 26, 29, 32 in turn until it
is <= target_mb (10 MB); the last try is kept if none fits. Returns the crf in use (20 = the composers' first encode, unchanged)."""
import os, subprocess

CRF0 = 20


def shrink(path, target_mb=10.0, limit_mb=12.0, crfs=(23, 26, 29, 32)):
    if os.path.getsize(path) / 1e6 <= limit_mb:
        return CRF0
    src = path + ".crf20.mp4"; os.replace(path, src); used = CRF0
    try:
        for c in crfs:
            subprocess.run(["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(c), "-preset", "medium",
                            "-movflags", "+faststart", path], check=True)
            used = c
            if os.path.getsize(path) / 1e6 <= target_mb:
                break
    finally:
        if os.path.exists(path):
            os.remove(src)
        else:
            os.replace(src, path); used = CRF0
    return used
