"""
Detection Service
Responsible only for: loading the YOLO model, running it on an image or video,
and returning a structured result.
Knows nothing about CSV, PDF, or the UI - it just "detects and returns the result".
"""

import cv2
from ultralytics import YOLO
from app.config import MODEL_PATH, VIOLATION_PREFIXES


class DetectionResult:
    """
    A simple object that holds the detection result in a structured way,
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
    the annotated output video file, per-class violation counts across all
    processed frames, and one representative frame (the one with the most
    violations) used later for the PDF report thumbnail.
    """
 
    def __init__(self, output_video_path, violation_counts, total_frames_processed,
                 max_persons_in_a_frame, representative_frame):
        self.output_video_path = output_video_path
        self.violation_counts = violation_counts              # dict: {"no hardhat": 12, ...}
        self.total_frames_processed = total_frames_processed
        self.max_persons_in_a_frame = max_persons_in_a_frame
        self.representative_frame = representative_frame        # numpy array (annotated)
 
    @property
    def total_violations(self):
        return sum(self.violation_counts.values())
 
    @property
    def is_compliant(self):
        return self.total_violations == 0
    

class SafetyDetector:
    """
    Main detection class - loads the model once on creation,
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
 
        violation_counts = {}
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
                results = self.model(frame, verbose=False)
                result = results[0]
                last_annotated_frame = result.plot()
 
                detected_classes = [self.model.names[int(box.cls[0])] for box in result.boxes]
                persons_here = detected_classes.count("person")
                max_persons_in_a_frame = max(max_persons_in_a_frame, persons_here)
 
                frame_violations = [c for c in detected_classes if c.startswith(("no ", "no-"))]
                for v in frame_violations:
                    violation_counts[v] = violation_counts.get(v, 0) + 1
 
                if len(frame_violations) > max_violations_seen:
                    max_violations_seen = len(frame_violations)
                    representative_frame = last_annotated_frame
 
                total_frames_processed += 1
 
            # write the last computed annotation on every frame, so output stays smooth
            writer.write(last_annotated_frame if last_annotated_frame is not None else frame)
            frame_index += 1
 
        cap.release()
        writer.release()
 
        if representative_frame is None:
            representative_frame = last_annotated_frame
 
        return VideoDetectionResult(
            output_video_path=output_path,
            violation_counts=violation_counts,
            total_frames_processed=total_frames_processed,
            max_persons_in_a_frame=max_persons_in_a_frame,
            representative_frame=representative_frame,
        )
 
    def save_annotated_image(self, annotated_image, path: str):
        cv2.imwrite(path, annotated_image)

