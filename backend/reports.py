from io import BytesIO
from pathlib import Path


def build_pdf_report(analysis, static_dir: Path) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"RoadSight Evidence Report {analysis.id}",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=22, leading=26, textColor=colors.HexColor("#0B514C"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#18333B"), spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#53666D")))

    summary = analysis.summary or {}
    story = [
        Paragraph("RoadSight Traffic Evidence Report", styles["ReportTitle"]),
        Paragraph("AI-assisted traffic safety review — human verification required", styles["Small"]),
        Spacer(1, 8 * mm),
    ]

    details = [
        ["Case ID", analysis.id],
        ["Timestamp", analysis.created_at.isoformat()],
        ["Source file", analysis.original_filename],
        ["Selected modules", ", ".join(summary.get("selected_modules") or [])],
        ["Decision", str(summary.get("final_status", analysis.final_status)).replace("_", " ").title()],
        ["Risk / Severity", f"{summary.get('risk_score', 0)}/100 — {str(summary.get('severity', 'low')).title()}"],
        ["Processing time", f"{summary.get('processing_time_ms', 'Not measured')} ms"],
    ]
    detail_table = Table(details, colWidths=[42 * mm, 128 * mm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F4F1")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0B635C")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CCDCD8")),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#FAFCFB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([detail_table, Paragraph("Decision rationale", styles["Section"])])
    for reason in summary.get("decision_reasons") or ["No decision rationale was recorded."]:
        story.append(Paragraph(f"• {reason}", styles["BodyText"]))
    story.extend([
        Spacer(1, 3 * mm),
        Paragraph(f"Recommendation: {summary.get('recommendation', 'Manual review is recommended.')}", styles["BodyText"]),
        Paragraph("Annotated evidence", styles["Section"]),
    ])

    image_path = static_dir / "outputs" / analysis.annotated_name
    if image_path.exists():
        evidence = Image(str(image_path))
        max_width, max_height = 170 * mm, 95 * mm
        scale = min(max_width / evidence.imageWidth, max_height / evidence.imageHeight)
        evidence.drawWidth = evidence.imageWidth * scale
        evidence.drawHeight = evidence.imageHeight * scale
        story.append(evidence)
    else:
        story.append(Paragraph("Annotated image is unavailable.", styles["Small"]))

    story.extend([PageBreak(), Paragraph("Detection evidence", styles["Section"])])
    evidence_rows = [["Module", "Object", "Confidence", "Assessment", "OCR / rule"]]
    for detection in analysis.detections[:60]:
        confidence = "—" if detection.confidence is None else f"{detection.confidence * 100:.1f}%"
        detail = detection.ocr_text or detection.rule or "—"
        evidence_rows.append([
            detection.module.replace("_", " ").title(),
            (detection.class_name or "—").replace("_", " ").title(),
            confidence,
            (detection.status or "—").replace("_", " ").title(),
            detail.replace("_", " "),
        ])
    evidence_table = Table(evidence_rows, repeatRows=1, colWidths=[34 * mm, 30 * mm, 22 * mm, 35 * mm, 49 * mm])
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123840")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#D3DEDB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9F8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([
        evidence_table,
        Spacer(1, 6 * mm),
        Paragraph(
            "Important: This report documents automated screening results. It is not a legal determination and must be verified against original evidence, local traffic rules, and qualified human review.",
            styles["Small"],
        ),
    ])
    document.build(story)
    return buffer.getvalue()
