"""
Report Service
Responsible only for: generating a PDF report from a ready detection result.
Knows nothing about the model or CSV — it just "receives a result and produces a PDF".
"""

import os
from datetime import datetime

from fpdf import FPDF

from app.config import REPORTS_DIR
from app.services.detector import format_timestamp


def generate_pdf_report(annotated_image_path: str, total_persons: int, violations: list) -> str:
    """
    Generates a simple PDF report containing:
    - date and time
    - number of workers detected
    - list of violations (if any)
    - the annotated result image
    Returns the path to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Worker Safety Inspection Report", ln=True, align="C")

    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Workers Detected: {total_persons}", ln=True)
    pdf.cell(0, 8, f"Violations Found: {len(violations)}", ln=True)
    pdf.ln(4)

    if violations:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Violation Details:", ln=True)
        pdf.set_font("Arial", size=11)
        for v in violations:
            pdf.cell(0, 7, f"- {v['type']} (confidence: {v['confidence']:.2f})", ln=True)
    else:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "No violations detected. All workers compliant.", ln=True)

    pdf.ln(6)
    pdf.image(annotated_image_path, x=15, w=180)

    report_path = os.path.join(
        REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    pdf.output(report_path)
    return report_path


def generate_video_pdf_report(video_result, temp_frame_path: str) -> str:
    """
    Generates a simple PDF summary report for a processed video:
    - total frames analyzed
    - max workers seen in a single frame
    - violation counts by type across the whole video
    - one representative frame (the one with the most violations) as a thumbnail
    Returns the path to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Worker Safety Video Inspection Report", ln=True, align="C")

    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Frames Analyzed: {video_result.total_frames_processed}", ln=True)
    pdf.cell(0, 8, f"Max Workers Seen in a Single Frame: {video_result.max_persons_in_a_frame}", ln=True)
    pdf.cell(0, 8, f"Total Violation Events: {video_result.total_violations}", ln=True)
    pdf.ln(4)

    if video_result.violation_events:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Violation Timeline:", ln=True)
        pdf.set_font("Arial", size=11)
        for event in video_result.violation_events:
            start = format_timestamp(event["start_seconds"])
            end = format_timestamp(event["end_seconds"])
            pdf.cell(0, 7, f"- {event['type']} : from {start} to {end}", ln=True)
    else:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "No violations detected across the video. All workers compliant.", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, "Representative frame (highest violation count):", ln=True)
    pdf.image(temp_frame_path, x=15, w=180)

    report_path = os.path.join(
        REPORTS_DIR, f"video_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    pdf.output(report_path)
    return report_path