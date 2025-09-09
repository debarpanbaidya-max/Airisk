from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import json

def create_pdf():
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)

    path = "model_metrics_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [Paragraph("Model Performance Report", styles["Title"]), Spacer(1, 12)]

    for k, v in metrics.items():
        content.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        content.append(Spacer(1, 6))

    doc.build(content)
    return path
