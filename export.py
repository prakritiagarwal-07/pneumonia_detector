import os
import tempfile
import datetime
from pathlib import Path
from PIL import Image
from fpdf import FPDF

def make_pdf(orig, hmap, label, conf, elapsed, fname, tta=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, 15)
    
    # Header
    pdf.set_fill_color(11, 14, 20)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(232, 237, 248)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "PneumoScan - AI Chest X-Ray Report", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 101, 130)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   File: {fname}", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(6)

    # Diagnosis
    r, g, b = (61, 214, 140) if label == "NORMAL" else (247, 91, 91)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 13, f"  {label}   ({conf:.1f}% confidence)", ln=True, fill=True)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)

    # Metrics
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(240, 242, 248)
    metrics = [
        ("Model", "ResNet50 - FastAI 2.x"),
        ("Input", "224x224 px"),
        ("Time", f"{elapsed:.2f}s"),
        ("Certainty", "High" if conf >= 85 else "Moderate" if conf >= 65 else "Low")
    ]
    for k, v in metrics:
        pdf.set_x(10)
        pdf.cell(50, 7, k, border=1, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(v), border=1, ln=True)
        pdf.set_font("Helvetica", "B", 10)
    pdf.ln(5)

    # Images (Windows-safe temp directory)
    tmpdir = Path(tempfile.gettempdir())
    def save_tmp(img, tag):
        p = tmpdir / f"ps_{tag}_{os.getpid()}.jpg"
        img.convert("RGB").save(p, "JPEG", quality=90)
        return str(p)

    orig_p = save_tmp(orig, "orig")
    pdf.set_font("Helvetica", "B", 10)
    if hmap is not None:
        pdf.cell(90, 7, "Original X-ray", ln=False)
        pdf.cell(0, 7, "Grad-CAM Heatmap", ln=True)
        pdf.image(orig_p, x=10, y=pdf.get_y(), w=88, h=88)
        pdf.image(save_tmp(hmap, "heat"), x=112, y=pdf.get_y(), w=88, h=88)
        pdf.ln(92)
    else:
        pdf.cell(0, 7, "Original X-ray", ln=True)
        pdf.image(orig_p, x=10, y=pdf.get_y(), w=88, h=88)
        pdf.ln(92)

    # TTA
    if tta:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Second Opinion (TTA)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Votes Normal: {tta['votes_norm']}  |  Votes Pneumonia: {tta['votes_pneu']}", ln=True)
        pdf.cell(0, 6, f"  Mean PNEUMONIA prob: {tta['mean']*100:.1f}%  |  Std: {tta['std']*100:.1f}%", ln=True)
        pdf.ln(3)

    # Clinical note
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Clinical Observation", ln=True)
    pdf.set_font("Helvetica", "", 9)
    note = (
        "No significant opacification detected. Cardiac silhouette normal. Pattern consistent with healthy radiograph."
        if label == "NORMAL" else
        "Diffuse or focal opacification detected. May be consistent with bacterial/viral pneumonia.  "
        "Physician review strongly recommended."
    )
    pdf.multi_cell(0, 6, note)

    # Disclaimer
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(130, 140, 160)
    pdf.multi_cell(0, 5, "DISCLAIMER: Research prototype only. NOT a certified medical device.  "
                       "Never substitute professional radiological diagnosis.")
    return bytes(pdf.output())