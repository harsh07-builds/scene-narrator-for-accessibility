"""
Script to generate a professional PDF Manual for the Spatial Reasoning Module.
Outputs to docs/Spatial_Reasoning_Manual.pdf (next to this script).
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "AI Scene Narrator - Spatial Reasoning Engine Manual")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, footer_text)
        self.drawString(54, 36, "Confidential - Assistive Edge AI System (Group 168)")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)

        self.restoreState()


def build_manual_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Cobalt blue
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal body
    bg_code = colors.HexColor("#F7FAFC")
    border_code = colors.HexColor("#CBD5E0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_neutral,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        bulletIndent=6,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # Title block
    story.append(Paragraph("AI Scene Narrator for Accessibility", title_style))
    story.append(Paragraph("Technical Manual: Spatial Reasoning Engine (app/spatial.py) - From Scratch", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceBefore=0, spaceAfter=14))

    # Executive Overview
    story.append(Paragraph("1. Executive Overview & Problem Context", h1_style))
    story.append(Paragraph(
        "Traditional object detection models (e.g., YOLO) output isolated object classifications with bounding boxes "
        "(e.g., <i>'chair', 'table'</i>). For visually impaired users, discrete label lists do not convey situational awareness. "
        "The user immediately needs to know: <b>Where is the object? What is it resting on? What is beside it? Is it in front of me?</b>",
        body_style
    ))
    story.append(Paragraph(
        "The <b>Spatial Reasoning Engine</b> acts as the critical bridge between raw detections and Natural Language Generation (NLG). "
        "It evaluates bounding-box geometry using deterministic 2D heuristics without requiring heavy 3D neural nets, LiDAR, "
        "or cloud connectivity. This ensures 100% offline edge execution on commodity CPU hardware with sub-millisecond latency.",
        body_style
    ))

    # System Architecture Diagram / Flow
    story.append(Paragraph("2. System Architecture & Information Pipeline", h1_style))
    story.append(Paragraph(
        "The complete Scene Narrator architecture operates as a strict four-stage linear pipeline:",
        body_style
    ))

    arch_data = [
        [
            Paragraph("<b>Stage 1: Detection Layer</b><br/><font color='#718096'>app/detection.py</font>", body_style),
            Paragraph("YOLOv8n extracts labels, confidence scores, and bounding boxes <i>(x1, y1, x2, y2)</i> per frame.", body_style)
        ],
        [
            Paragraph("<b>Stage 2: Spatial Reasoning</b><br/><font color='#718096'>app/spatial.py</font>", body_style),
            Paragraph("<b>(Current Module)</b> Analyzes bounding box intervals, centroids, ground contact base lines, and scale ratios to infer grounded predicates.", body_style)
        ],
        [
            Paragraph("<b>Stage 3: Natural Language Generation</b><br/><font color='#718096'>app/nlg.py</font>", body_style),
            Paragraph("Transforms spatial predicates and frame zones into grammatically diverse spoken English sentences using templates and synonym banks.", body_style)
        ],
        [
            Paragraph("<b>Stage 4: Text-To-Speech (TTS)</b><br/><font color='#718096'>app/tts_engine.py</font>", body_style),
            Paragraph("Synthesizes speech locally using pyttsx3 (SAPI5 on Windows / NSSpeechSynthesizer on macOS) for low-latency voice narration.", body_style)
        ],
    ]
    t_arch = Table(arch_data, colWidths=[2.2 * inch, 4.8 * inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 1), (1, 1), colors.HexColor("#EBF8FF")), # Highlight spatial stage
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # Core Mathematical Formulations
    story.append(Paragraph("3. Geometric Heuristics & Predicate Definitions (from Scratch)", h1_style))
    story.append(Paragraph(
        "Given camera coordinates where <i>(0, 0)</i> is top-left, <i>x</i> increases rightward, and <i>y</i> increases downward, "
        "each detected bounding box A is represented by top-left (x1, y1) and bottom-right (x2, y2):",
        body_style
    ))
    story.append(Paragraph("• <b>Centroids</b>: cx = (x1 + x2) / 2, cy = (y1 + y2) / 2", bullet_style))
    story.append(Paragraph("• <b>Dimensions</b>: w = x2 - x1, h = y2 - y1, Area = w x h", bullet_style))
    story.append(Paragraph("• <b>1D Horizontal Overlap</b>: overlap_x = max(0, min(A.x2, B.x2) - max(A.x1, B.x1))", bullet_style))
    story.append(Paragraph("• <b>1D Vertical Overlap</b>: overlap_y = max(0, min(A.y2, B.y2) - max(A.y1, B.y1))", bullet_style))
    story.append(Paragraph("• <b>Horizontal Gap</b>: gap_x = max(0, max(A.x1, B.x1) - min(A.x2, B.x2))", bullet_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Detailed Predicate Rules:", h2_style))

    pred_data = [
        [
            Paragraph("<b>Predicate</b>", body_style),
            Paragraph("<b>Geometric Conditions &amp; Rationale</b>", body_style),
            Paragraph("<b>Assistive Example</b>", body_style)
        ],
        [
            Paragraph("<b>on</b><br/>(Support)", body_style),
            Paragraph(
                "1. Subject is above object (A.cy &lt; B.cy)<br/>"
                "2. High horizontal overlap: overlap_x / A.w &gt;= 0.45<br/>"
                "3. Base contact: A.y2 aligns near B.y1 within top margin [-0.20 A.h, +0.35 B.h]<br/>"
                "4. Plausibility: A.Area &lt;= 1.8 x B.Area (a sofa is not on a cup)",
                body_style
            ),
            Paragraph("<i>'A cup is on the desk.'</i>", body_style)
        ],
        [
            Paragraph("<b>in front of</b><br/>(Depth Proxy)", body_style),
            Paragraph(
                "1. In perspective, closer objects tend to have a lower base in frame (A.y2 &gt; B.y2 by &gt;= 0.06 x max(h))<br/>"
                "2. Shared line of sight: overlap_x / min(w) &gt;= 0.15<br/>"
                "3. Scale factor: Area ratio &gt;= 1.6x or base difference &gt;= 15%",
                body_style
            ),
            Paragraph("<i>'A person is in front of the couch.'</i>", body_style)
        ],
        [
            Paragraph("<b>next to</b><br/>(Proximity)", body_style),
            Paragraph(
                "1. Vertical plane alignment: vertical overlap &gt;= 0.30 or |A.cy - B.cy| &lt;= 0.50 x max(h)<br/>"
                "2. Horizontal proximity: gap_x &lt;= 1.25 x min(A.w, B.w)<br/>"
                "3. Not vertically stacked or resting on top",
                body_style
            ),
            Paragraph("<i>'A chair is next to the table.'</i>", body_style)
        ],
        [
            Paragraph("<b>inside</b><br/>(Containment)", body_style),
            Paragraph(
                "Subject bounding box is almost entirely contained inside object: "
                "Intersection / A.Area &gt;= 0.82 with A.Area &lt; 0.85 x B.Area.",
                body_style
            ),
            Paragraph("<i>'An apple is inside the bowl.'</i>", body_style)
        ],
        [
            Paragraph("<b>to left / right of</b><br/>(Fallback)", body_style),
            Paragraph(
                "When objects are horizontally distant without contact: A.cx &lt; B.cx evaluates to 'to the left of', "
                "suppressing pairs separated by &gt;75% of frame width.",
                body_style
            ),
            Paragraph("<i>'A lamp is to the left of the bed.'</i>", body_style)
        ],
        [
            Paragraph("<b>frame_zone()</b><br/>(Single Object)", body_style),
            Paragraph(
                "Normalized centroid (cx/W, cy/H) mapped across a 3x3 matrix: "
                "Center, Left, Right, Top, Bottom, Top-Left, Top-Right, Bottom-Left, Bottom-Right.",
                body_style
            ),
            Paragraph("<i>'There is a laptop in the center.'</i>", body_style)
        ]
    ]

    t_pred = Table(pred_data, colWidths=[1.4 * inch, 4.1 * inch, 1.5 * inch])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 10))

    # Anti-Clutter & Scene Graph Optimization
    story.append(Paragraph("4. Anti-Clutter & Scene Graph Optimization", h1_style))
    story.append(Paragraph(
        "A scene with N objects generates N(N-1) ordered pair comparisons. Without filtering, an 8-object scene creates 56 relations, "
        "producing cognitive overload for a blind user. <b>app/spatial.py</b> employs three algorithmic optimizations:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Salience Hierarchy Ranking</b>: Relationships are weighted by informational urgency: "
        "Support (1.0) > Containment (0.95) > Depth (0.85) > Proximity (0.80) > Directional (0.50).",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Reciprocal Deduplication</b>: Prunes inverse pairs so only the canonical primary relation is retained "
        "(e.g., outputs <i>'A is on B'</i> rather than redundantly following up with <i>'B is below A'</i>).",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Transitive Support Suppression</b>: If object A is resting <i>on</i> object B, and B is <i>next to</i> C, "
        "any synthetic relation between A and C (e.g. <i>'cup next to chair'</i>) is suppressed because A is bound to B. "
        "This ensures narrations sound natural and human.",
        bullet_style
    ))
    story.append(Spacer(1, 10))

    # User Guide
    story.append(Paragraph("5. Step-by-Step Usage Guide (How to Use It Now)", h1_style))
    story.append(Paragraph(
        "Run everything from the project root (<b>ai-scene-narrator/</b>). Here is how to run and test it:",
        body_style
    ))

    code_block_1 = (
        "# 1. Run the full test suite (spatial, NLG, pipeline, API):\n"
        "python -m pytest\n\n"
        "# 2. Run the end-to-end demo (spatial + NLG on synthetic scenes):\n"
        "python -m app.demo\n\n"
        "# 3. Run the YOLO detector on a live webcam (press 'q' to quit):\n"
        "python -m app.detection\n\n"
        "# 4. Run the YOLO detector on an image file:\n"
        "python -m app.detection --image path/to/image.jpg\n\n"
        "# 5. Start the API (docs at http://localhost:8000/docs):\n"
        "uvicorn app.main:app --port 8000"
    )
    t_code1 = Table([[Paragraph(code_block_1.replace("\n", "<br/>"), code_style)]], colWidths=[7.0 * inch])
    t_code1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_code),
        ('BOX', (0, 0), (-1, -1), 1, border_code),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_code1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Calling the Spatial Module from Code:", h2_style))
    code_block_2 = (
        "from app.detection import Detection\n"
        "from app.spatial import analyze_scene_spatial, SpatialConfig\n\n"
        "# Create detections (or receive them from app.detection.detect(frame))\n"
        "desk = Detection(label='desk', confidence=0.95, x1=100, y1=200, x2=550, y2=450)\n"
        "laptop = Detection(label='laptop', confidence=0.92, x1=200, y1=160, x2=380, y2=220)\n\n"
        "# Run spatial scene analysis\n"
        "result = analyze_scene_spatial([desk, laptop], frame_width=640, frame_height=480)\n\n"
        "print(result['structured_narration'])\n"
        "# Output: 'A laptop is on the desk.'\n"
        "print(result['primary_relations'])\n"
        "# Output: [{'subject': 'laptop', 'predicate': 'on', 'object': 'desk', 'confidence': 0.91, ...}]"
    )
    t_code2 = Table([[Paragraph(code_block_2.replace("\n", "<br/>"), code_style)]], colWidths=[7.0 * inch])
    t_code2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_code),
        ('BOX', (0, 0), (-1, -1), 1, border_code),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_code2)
    story.append(Spacer(1, 10))

    # Engineering hardening
    story.append(Paragraph("6. Engineering Hardening (v2.0)", h1_style))
    for text in [
        "<b>Class-aware overlap preservation:</b> different object classes are never removed merely because their boxes overlap.",
        "<b>Conservative duplicate suppression:</b> same-label boxes are merged only at high IoU or near-total containment.",
        "<b>Ambiguity rejection:</b> near-vertical boxes are not forced into a left/right relation.",
        "<b>Evidence-bounded confidence:</b> relation confidence cannot exceed the weakest detector confidence.",
        "<b>Upload safety:</b> image dimensions and total pixels are checked before OpenCV decoding.",
        "<b>Operational readiness:</b> CPU work is threadpooled, `/ready` exposes model readiness, and live signatures include movement.",
    ]:
        story.append(Paragraph("• " + text, bullet_style))
    story.append(Spacer(1, 8))

    # Next Steps in Roadmap
    story.append(Paragraph("7. Project Roadmap & Remaining Steps", h1_style))
    story.append(Paragraph(
        "All stages are implemented with a hardened production-oriented baseline:",
        body_style
    ))
    story.append(Paragraph("• <b>app/nlg.py (Stage 3)</b>: Template-based Natural Language Generation with synonym diversification.", bullet_style))
    story.append(Paragraph("• <b>app/tts_engine.py (Stage 4)</b>: Offline pyttsx3 speech on a worker thread; degrades gracefully on headless servers.", bullet_style))
    story.append(Paragraph("• <b>app/pipeline.py</b>: <i>NarrationPipeline</i> links detection -> spatial -> NLG, plus a scene-change gate for live narration.", bullet_style))
    story.append(Paragraph("• <b>app/main.py</b>: FastAPI service (<i>/api/analyze</i>, <i>/api/narrate</i>, <i>/health</i>).", bullet_style))
    story.append(Paragraph("• <b>frontend/streamlit_app.py</b>: Streamlit UI (upload / camera snapshot, browser speech).", bullet_style))
    story.append(Paragraph("• <b>Dockerfile + docker-compose.yml</b>: one-command deployment of API and frontend.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[manual] Successfully created PDF at: {filename}")


if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spatial_Reasoning_Manual.pdf")
    build_manual_pdf(out_pdf)
