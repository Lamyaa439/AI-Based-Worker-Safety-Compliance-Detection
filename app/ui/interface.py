"""
UI Layer
Responsible only for: designing the Gradio interface and calling the services.
Contains no technical logic (detection, logging, reporting) — it just "wires"
the services together.
"""

import os
from datetime import datetime

import gradio as gr

from app.config import REPORTS_DIR
from app.services.detector import SafetyDetector, format_timestamp
from app.services.logger import log_violations, log_video_violations
from app.services.report import generate_pdf_report, generate_video_pdf_report

# Model is loaded once when the app starts (not on every upload)
detector = SafetyDetector()


def process_image(image):
    """
    Main wiring function for images: calls the three services in order
    and returns the final result to the Gradio interface.
    """
    if image is None:
        return None, "Please upload an image first.", None

    # 1. Detect
    result = detector.detect(image)

    # 2. Save the annotated image (needed to embed it in the PDF)
    annotated_path = os.path.join(REPORTS_DIR, "_temp_annotated.jpg")
    detector.save_annotated_image(result.annotated_image, annotated_path)

    # 3. Build the text summary
    if not result.detected_classes:
        summary = "No workers detected in the image."
    elif result.violations:
        violation_names = ", ".join(sorted({v["type"] for v in result.violations}))
        summary = (
            f"⚠️ Violations detected: {violation_names}\n"
            f"Workers detected: {result.total_persons}"
        )
    else:
        summary = f"✅ All {result.total_persons} detected worker(s) are compliant with safety standards."

    # 4. Log violations (if any)
    image_name = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_violations(result.violations, image_name)

    # 5. Generate PDF report
    pdf_path = generate_pdf_report(annotated_path, result.total_persons, result.violations)

    return result.annotated_image, summary, pdf_path


def process_video(video_path):
    """
    Main wiring function for videos: runs frame-by-frame detection,
    logs an aggregated violation summary, and generates a PDF summary report.
    """
    if video_path is None:
        return None, "Please upload a video first.", None

    # 1. Detect (processes the whole video, frame by frame)
    output_video_path = os.path.join(REPORTS_DIR, "_temp_output_video.mp4")
    video_result = detector.detect_video(video_path, output_video_path)

    # 2. Save the representative frame (needed to embed it in the PDF)
    frame_path = os.path.join(REPORTS_DIR, "_temp_video_frame.jpg")
    if video_result.representative_frame is not None:
        detector.save_annotated_image(video_result.representative_frame, frame_path)

    # 3. Build the text summary — now with a timestamped timeline instead of raw counts
    if video_result.is_compliant:
        summary = (
            f"✅ No violations detected in the video.\n"
            f"Max workers seen in a single frame: {video_result.max_persons_in_a_frame}\n"
            f"Frames analyzed: {video_result.total_frames_processed}"
        )
    else:
        timeline_lines = []
        for event in video_result.violation_events:
            start = format_timestamp(event["start_seconds"])
            end = format_timestamp(event["end_seconds"])
            timeline_lines.append(f"⚠️ {event['type']} - from minute {start} to {end}")
        timeline_text = "\n".join(timeline_lines)
        summary = (
            f"{timeline_text}\n\n"
            f"Frames analyzed: {video_result.total_frames_processed}"
        )

    # 4. Log the timestamped violation events
    video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_video_violations(video_result.violation_events, video_name)

    # 5. Generate PDF report
    pdf_path = generate_video_pdf_report(video_result, frame_path)

    return video_result.output_video_path, summary, pdf_path


def build_interface() -> gr.Blocks:
    """Builds and returns the full Gradio interface, ready to launch."""
    with gr.Blocks(title="Worker Safety Monitoring System") as demo:
        gr.Markdown("# 🦺 Worker Safety Monitoring System")
        gr.Markdown("Upload an image or video of a worker or job site, and the system will check for hardhat and safety vest compliance.")

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="Upload Image")
                        image_submit_btn = gr.Button("Analyze Image", variant="primary")

                    with gr.Column():
                        image_output = gr.Image(label="Result with Bounding Boxes")
                        image_summary = gr.Textbox(label="Safety Summary", lines=4)
                        image_pdf = gr.File(label="Download PDF Report")

                image_submit_btn.click(
                    fn=process_image,
                    inputs=input_image,
                    outputs=[image_output, image_summary, image_pdf],
                )

            with gr.Tab("Video"):
                gr.Markdown(
                    "⏳ Video processing takes longer than an image, especially for longer videos. "
                    "Please wait after clicking the analyze button."
                )
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="Upload Video")
                        video_submit_btn = gr.Button("Analyze Video", variant="primary")

                    with gr.Column():
                        video_output = gr.Video(label="Result with Bounding Boxes")
                        video_summary = gr.Textbox(label="Safety Summary", lines=4)
                        video_pdf = gr.File(label="Download PDF Report")

                video_submit_btn.click(
                    fn=process_video,
                    inputs=input_video,
                    outputs=[video_output, video_summary, video_pdf],
                )

        gr.Markdown(
            "---\n"
            "📁 All detected violations are automatically logged to `violations_log.csv`"
        )

    return demo