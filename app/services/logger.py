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
from app.services.detector import format_timestamp


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


def log_video_violations(violation_events: list, video_name: str):
    """
    Logs each violation event (with its start/end timestamp) found in a
    video to the CSV file — one row per continuous violation occurrence.
    Example row: "no hardhat", "01:15 - 01:20"
    """
    if not violation_events:
        return

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image", "violation_type", "confidence"])
        for event in violation_events:
            time_range = f"{format_timestamp(event['start_seconds'])} - {format_timestamp(event['end_seconds'])}"
            writer.writerow([
                datetime.now().isoformat(),
                video_name,
                f"{event['type']} ({time_range})",
                "-",
            ])