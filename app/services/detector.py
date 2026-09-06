"""
Detection Service
Responsible only for: loading the YOLO model, running it on an image or video,
and returning a structured result.
Knows nothing about CSV, PDF, or the UI — it just "detects and returns the result".
"""

import cv2
from ultralytics import YOLO
from app.config import MODEL_PATH, VIOLATION_PREFIXES


class DetectionResult:
    """
    A simple object that holds a single-image detection result in a structured way,
    so other layers can use it without needing to understand YOLO's internal details.
    """

    def __init__(self, annotated_image, detected_classes, confidences):
        self.annotated_image = annotated_image      # image with boxes drawn (numpy array)
        self.detected_classes = detected_classes     # list of detected class names
        self.confidences = confidences                # corresponding confidence scores

    @property
    def total_persons(self):
        return self.detected_classes.count("person")

    @property
    def violations(self):
        """Returns violations only (classes starting with 'no ')"""
        return [
            {"type": cls, "confidence": conf}
            for cls, conf in zip(self.detected_classes, self.confidences)
            if cls.startswith(VIOLATION_PREFIXES)
        ]

    @property
    def is_compliant(self):
        return len(self.violations) == 0 and len(self.detected_classes) > 0


class VideoDetectionResult:
    """
    Holds the aggregated result of processing an entire video:
    the annotated output video file, a list of violation "events" with
    start/end timestamps, and one representative frame (the one with the
    most simultaneous violations) used later for the PDF report thumbnail.
    """

    def __init__(self, output_video_path, violation_events, total_frames_processed,
                 max_persons_in_a_frame, representative_frame):
        self.output_video_path = output_video_path
        # violation_events: list of dicts, e.g.
        # {"type": "no hardhat", "start_seconds": 75.0, "end_seconds": 80.0}
        self.violation_events = violation_events
        self.total_frames_processed = total_frames_processed
        self.max_persons_in_a_frame = max_persons_in_a_frame
        self.representative_frame = representative_frame        # numpy array (annotated)

    @property
    def total_violations(self):
        return len(self.violation_events)

    @property
    def is_compliant(self):
        return self.total_violations == 0


def format_timestamp(seconds: float) -> str:
    """Formats a number of seconds as MM:SS, e.g. 75.0 -> '01:15'."""
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


class SafetyDetector:
    """
    Main detection class — loads the model once on creation,
    then reuses it to run inference on any incoming image or video.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image) -> DetectionResult:
        """Runs detection on a single image."""
        results = self.model(image)
        result = results[0]

        annotated_image = result.plot()

        detected_classes = [self.model.names[int(box.cls[0])] for box in result.boxes]
        confidences = [float(box.conf[0]) for box in result.boxes]

        return DetectionResult(annotated_image, detected_classes, confidences)

    def detect_video(self, video_path: str, output_path: str, frame_skip: int = 3) -> VideoDetectionResult:
        """
        Runs detection on a video, frame by frame, and writes an annotated
        output video to `output_path`.

        Instead of just counting how many frames contained each violation,
        this tracks *when* each violation started and ended, producing a
        timeline of events like:
            "no hardhat" from 01:15 to 01:20
            "no hardhat" from 03:05 to 03:10

        How it works: for every analyzed frame, we know the current playback
        time (frame_index / fps). For each violation type, we keep an "open"
        event running as long as that violation keeps appearing in
        consecutive analyzed frames. The moment it stops appearing, we close
        the event (end_seconds = last time it was seen) and record it.

        `frame_skip`: process every Nth frame instead of every single frame,
        to keep processing time reasonable on a CPU. Skipped frames are still
        written to the output video using the last computed annotation, so
        the output video plays at normal speed without stutter.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Tracks violations currently "in progress": {class_name: {"start": t, "last_seen": t}}
        active_violations = {}
        violation_events = []  # finished events go here: {"type", "start_seconds", "end_seconds"}

        total_frames_processed = 0
        max_persons_in_a_frame = 0
        representative_frame = None
        max_violations_seen = -1

        frame_index = 0
        last_annotated_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                current_time = frame_index / fps

                results = self.model(frame, verbose=False)
                result = results[0]
                last_annotated_frame = result.plot()

                detected_classes = [self.model.names[int(box.cls[0])] for box in result.boxes]
                persons_here = detected_classes.count("person")
                max_persons_in_a_frame = max(max_persons_in_a_frame, persons_here)

                present_violation_types = {c for c in detected_classes if c.startswith(("no ", "no-"))}

                # Close any active event whose violation type disappeared this frame
                for v_type in list(active_violations.keys()):
                    if v_type not in present_violation_types:
                        event = active_violations.pop(v_type)
                        violation_events.append({
                            "type": v_type,
                            "start_seconds": event["start"],
                            "end_seconds": event["last_seen"],
                        })

                # Start new events or extend ongoing ones
                for v_type in present_violation_types:
                    if v_type in active_violations:
                        active_violations[v_type]["last_seen"] = current_time
                    else:
                        active_violations[v_type] = {"start": current_time, "last_seen": current_time}

                if len(present_violation_types) > max_violations_seen:
                    max_violations_seen = len(present_violation_types)
                    representative_frame = last_annotated_frame

                total_frames_processed += 1

            # write the last computed annotation on every frame, so output stays smooth
            writer.write(last_annotated_frame if last_annotated_frame is not None else frame)
            frame_index += 1

        cap.release()
        writer.release()

        # Close any violations that were still active when the video ended
        for v_type, event in active_violations.items():
            violation_events.append({
                "type": v_type,
                "start_seconds": event["start"],
                "end_seconds": event["last_seen"],
            })

        # Sort events by start time so the report/log reads chronologically
        violation_events.sort(key=lambda e: e["start_seconds"])

        if representative_frame is None:
            representative_frame = last_annotated_frame

        return VideoDetectionResult(
            output_video_path=output_path,
            violation_events=violation_events,
            total_frames_processed=total_frames_processed,
            max_persons_in_a_frame=max_persons_in_a_frame,
            representative_frame=representative_frame,
        )

    def save_annotated_image(self, annotated_image, path: str):
        cv2.imwrite(path, annotated_image)