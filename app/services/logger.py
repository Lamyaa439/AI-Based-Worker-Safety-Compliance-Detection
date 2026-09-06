"""
Logging Service
Responsible only for: saving the violation log to a CSV file.
Knows nothing about the model or the UI — it just "receives a list of
violations and logs them".
"""

import csv
import os
from datetime import datetime

from app.config import LOG_FILE


def log_violations(violations: list, image_name: str):
    """
    Logs each violation to a CSV file along with the timestamp and image name.
    Creates the file with a header row on first use if it doesn't exist yet.
    """
    if not violations:
        return

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image", "violation_type", "confidence"])
        for v in violations:
            writer.writerow([
                datetime.now().isoformat(),
                image_name,
                v["type"],
                v["confidence"],
            ])


def log_video_violations(violation_counts: dict, video_name: str):
    """
    Logs an aggregated summary of violations found across an entire video.
    One row per violation type, with the count of frames it appeared in
    (not per-instance, since a video can have thousands of frames).
    """
    if not violation_counts:
        return

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image", "violation_type", "confidence"])
        for violation_type, frame_count in violation_counts.items():
            writer.writerow([
                datetime.now().isoformat(),
                video_name,
                f"{violation_type} (in {frame_count} frames)",
                "-",
            ])
