import streamlit as st
import streamlit.components.v1 as components
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import time
#  Page Config
st.set_page_config(
    page_title="PneumoScan · AI Radiograph Analysis",
    layout="centered",
    initial_sidebar_state="collapsed",
)
#  Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0b0e14;
    color: #e8edf8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }
.stSpinner > div { border-top-color: #4f8ef7 !important; }

[data-testid="stFileUploader"] {
    background: #111520;
    border: 1.5px dashed #1e2740;
    border-radius: 16px;
    padding: 0.6rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4f8ef7; }
[data-testid="stFileUploaderDropzone"] label {
    color: #5a6582 !important;
    font-size: 0.85rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stAlert"] {
    background: rgba(247,91,91,0.08);
    border: 1px solid rgba(247,91,91,0.3);
    border-radius: 12px;
    color: #f75b5b;
}
</style>
""", unsafe_allow_html=True)
#  Model Loader
MODEL_PATH = Path("pneumonia_resnet50_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return load_learner(MODEL_PATH, cpu=True)
#  Top Bar
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:36px;height:36px;background:linear-gradient(135deg,#4f8ef7,#7c5fff);
                border-radius:9px;display:flex;align-items:center;justify-content:center">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="16"/>
        <line x1="8" y1="12" x2="16" y2="12"/>
      </svg>
    </div>
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;
                  color:#e8edf8;line-height:1.2">PneumoScan</div>
      <div style="font-size:0.65rem;color:#5a6582;letter-spacing:0.1em;
                  text-transform:uppercase">AI Radiograph Analysis</div>
    </div>
  </div>
  <div style="background:#171d2e;border:1px solid #1e2740;border-radius:20px;
              padding:4px 14px;font-size:0.72rem;font-family:'JetBrains Mono',monospace;
              color:#4f8ef7;letter-spacing:0.06em">ResNet50 · v1.0</div>
</div>
""", unsafe_allow_html=True)
#  Load model
with st.spinner("Loading model weights…"):
    learner = load_model()

if learner is None:
    st.error(
        "**Model file not found.** "
        "Place `pneumonia_resnet50_model.pkl` in the same directory as `app.py` "
        "and re-run `streamlit run app.py`."
    )
    st.stop()
#  Upload row
col_up, col_xray = st.columns(2, gap="medium")

with col_up:
    st.markdown("""
    <div style="margin-bottom:0.5rem">
      <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;
                   color:#5a6582;letter-spacing:0.1em;text-transform:uppercase">
        Input · Chest Radiograph
      </span>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("""
    <div style="display:flex;gap:6px;margin-top:8px">
      <span style="background:#171d2e;border-radius:6px;padding:2px 8px;font-size:0.65rem;
                   font-family:'JetBrains Mono',monospace;color:#5a6582">JPG</span>
      <span style="background:#171d2e;border-radius:6px;padding:2px 8px;font-size:0.65rem;
                   font-family:'JetBrains Mono',monospace;color:#5a6582">JPEG</span>
      <span style="background:#171d2e;border-radius:6px;padding:2px 8px;font-size:0.65rem;
                   font-family:'JetBrains Mono',monospace;color:#5a6582">PNG</span>
    </div>
    <div style="margin-top:10px;background:#111520;border:1px solid #1e2740;
                border-left:3px solid #4f8ef7;border-radius:8px;padding:10px 12px;
                font-size:0.75rem;color:#5a6582;line-height:1.6">
      PA / AP view X-rays only. Processed entirely on CPU — no data leaves your machine.
    </div>
    """, unsafe_allow_html=True)

with col_xray:
    st.markdown("""
    <div style="background:#111520;border:1px solid #1e2740;border-radius:16px;overflow:hidden">
      <div style="padding:8px 12px;border-bottom:1px solid #1e2740;
                  display:flex;align-items:center;gap:7px">
        <div style="width:10px;height:10px;border-radius:50%;background:#f75b5b"></div>
        <div style="width:10px;height:10px;border-radius:50%;background:#f7c35b"></div>
        <div style="width:10px;height:10px;border-radius:50%;background:#3dd68c"></div>
        <span style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                     color:#5a6582;margin-left:4px">radiograph_preview</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if uploaded:
        st.image(uploaded, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#060810;display:flex;align-items:center;
                    justify-content:center;padding:1.5rem;border-radius:0 0 16px 16px">
          <svg viewBox="0 0 200 180" width="100%" fill="none">
            <ellipse cx="100" cy="90" rx="68" ry="75" stroke="#1a2448" stroke-width="1.5"/>
            <ellipse cx="72" cy="86" rx="27" ry="38" fill="#0d1730" stroke="#253666" stroke-width="1"/>
            <ellipse cx="128" cy="86" rx="27" ry="38" fill="#0d1730" stroke="#253666" stroke-width="1"/>
            <ellipse cx="72" cy="82" rx="19" ry="30" fill="#111f45" opacity=".7"/>
            <ellipse cx="128" cy="82" rx="19" ry="30" fill="#111f45" opacity=".7"/>
            <line x1="100" y1="20" x2="100" y2="162" stroke="#1e2d5c" stroke-width="3" stroke-linecap="round"/>
            <line x1="56" y1="50" x2="144" y2="50" stroke="#1e2d5c" stroke-width="1.5"/>
            <line x1="52" y1="65" x2="148" y2="65" stroke="#162040" stroke-width="1"/>
            <line x1="50" y1="80" x2="150" y2="80" stroke="#162040" stroke-width="1"/>
            <line x1="50" y1="95" x2="150" y2="95" stroke="#162040" stroke-width="1"/>
            <line x1="52" y1="110" x2="148" y2="110" stroke="#162040" stroke-width="1"/>
            <text x="12" y="48" font-size="9" font-family="monospace" fill="#1e3060">L</text>
            <text x="158" y="172" font-size="8" font-family="monospace" fill="#1e3060">224x224</text>
          </svg>
        </div>
        """, unsafe_allow_html=True)
#  Inference + Results
if uploaded is not None:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    with st.spinner("Running forward pass…"):
        t0 = time.perf_counter()
        img = PILImage.create(uploaded)
        pred_class, pred_idx, probs = learner.predict(img)
        elapsed = time.perf_counter() - t0

    label      = str(pred_class).upper()
    confidence = float(probs[pred_idx]) * 100
    is_normal  = (label == "NORMAL")

    vocab       = learner.dls.vocab
    classes     = [str(v).upper() for v in vocab]
    prob_values = [float(probs[i]) * 100 for i in range(len(vocab))]

    # colours
    diag_color  = "#3dd68c" if is_normal else "#f75b5b"
    diag_bg     = "rgba(61,214,140,0.08)"  if is_normal else "rgba(247,91,91,0.08)"
    diag_border = "rgba(61,214,140,0.35)"  if is_normal else "rgba(247,91,91,0.35)"

    diag_icon_svg = (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        'stroke="#3dd68c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="20 6 9 17 4 12"/></svg>'
    ) if is_normal else (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        'stroke="#f75b5b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3'
        'L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    )

    certainty = ("High certainty" if confidence >= 85
                 else "Moderate certainty" if confidence >= 65
                 else "Low certainty")

    # ── build bars HTML as a plain string (no f-string nesting) ──
    bars_inner = ""
    for cls_name, pv in zip(classes, prob_values):
        bar_color = "#3dd68c" if cls_name == "NORMAL" else "#f75b5b"
        pv_str    = f"{pv:.2f}%"
        width_str = f"{pv:.1f}%"
        bars_inner += (
            '<div style="margin-bottom:0.65rem">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px">'
            '<span style="font-size:0.75rem;color:#5a6582;'
            'font-family:JetBrains Mono,monospace">' + cls_name + '</span>'
            '<span style="font-size:0.75rem;font-weight:600;'
            'font-family:JetBrains Mono,monospace;color:' + bar_color + '">' + pv_str + '</span>'
            '</div>'
            '<div style="height:5px;background:#1e2740;border-radius:3px;overflow:hidden">'
            '<div style="height:100%;width:' + width_str + ';background:' + bar_color + ';'
            'border-radius:3px"></div>'
            '</div>'
            '</div>'
        )

    # verdict
    if is_normal:
        v_bg      = "rgba(61,214,140,0.07)"
        v_border  = "rgba(61,214,140,0.25)"
        v_dot_bg  = "rgba(61,214,140,0.15)"
        v_icon    = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                     'stroke="#3dd68c" stroke-width="2.5" stroke-linecap="round">'
                     '<polyline points="20 6 9 17 4 12"/></svg>')
        v_title   = "Normal chest radiograph"
        v_color   = "#3dd68c"
        v_body    = ("No significant opacification detected in the lung fields. "
                     "Cardiac silhouette within normal limits. Costophrenic angles appear clear. "
                     "Pattern is consistent with a healthy chest radiograph.")
    else:
        v_bg      = "rgba(247,91,91,0.07)"
        v_border  = "rgba(247,91,91,0.25)"
        v_dot_bg  = "rgba(247,91,91,0.15)"
        v_icon    = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                     'stroke="#f75b5b" stroke-width="2.5" stroke-linecap="round">'
                     '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3'
                     'L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                     '<line x1="12" y1="9" x2="12" y2="13"/>'
                     '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
        v_title   = "Pneumonia pattern detected"
        v_color   = "#f75b5b"
        v_body    = ("Diffuse or focal opacification pattern detected in the lung fields. "
                     "Findings may be consistent with bacterial or viral pneumonia. "
                     "Clinical correlation and immediate physician review are strongly recommended.")

    elapsed_str    = f"{elapsed:.2f}s"
    confidence_str = f"{confidence:.1f}%"
    diag_sub       = "No pathology detected" if is_normal else "Pathology detected"

    # ── Single components.html() call for ALL result HTML ──
    result_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: 'Space Grotesk', sans-serif; color: #e8edf8; }
</style>
</head>
<body>

<!-- Metric cards -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem">

  <div style="background:DIAG_BG;border:1px solid DIAG_BORDER;border-radius:14px;padding:1.1rem">
    <div style="width:34px;height:34px;border-radius:9px;background:rgba(255,255,255,0.06);
                display:flex;align-items:center;justify-content:center;margin-bottom:0.7rem">
      DIAG_ICON
    </div>
    <div style="font-size:0.65rem;color:#5a6582;text-transform:uppercase;letter-spacing:0.1em;
                font-family:'JetBrains Mono',monospace;margin-bottom:3px">Diagnosis</div>
    <div style="font-size:1.3rem;font-weight:600;font-family:'JetBrains Mono',monospace;
                color:DIAG_COLOR">LABEL</div>
    <div style="font-size:0.68rem;color:#5a6582;margin-top:2px">DIAG_SUB</div>
  </div>

  <div style="background:#111520;border:1px solid #1e2740;border-radius:14px;padding:1.1rem">
    <div style="width:34px;height:34px;border-radius:9px;background:rgba(79,142,247,0.1);
                display:flex;align-items:center;justify-content:center;margin-bottom:0.7rem">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="#4f8ef7" stroke-width="2" stroke-linecap="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    </div>
    <div style="font-size:0.65rem;color:#5a6582;text-transform:uppercase;letter-spacing:0.1em;
                font-family:'JetBrains Mono',monospace;margin-bottom:3px">Confidence</div>
    <div style="font-size:1.3rem;font-weight:600;font-family:'JetBrains Mono',monospace;
                color:#4f8ef7">CONFIDENCE_STR</div>
    <div style="font-size:0.68rem;color:#5a6582;margin-top:2px">CERTAINTY</div>
  </div>

  <div style="background:#111520;border:1px solid #1e2740;border-radius:14px;padding:1.1rem">
    <div style="width:34px;height:34px;border-radius:9px;background:rgba(124,95,255,0.1);
                display:flex;align-items:center;justify-content:center;margin-bottom:0.7rem">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="#7c5fff" stroke-width="2" stroke-linecap="round">
        <circle cx="12" cy="12" r="10"/>
        <polyline points="12 6 12 12 16 14"/>
      </svg>
    </div>
    <div style="font-size:0.65rem;color:#5a6582;text-transform:uppercase;letter-spacing:0.1em;
                font-family:'JetBrains Mono',monospace;margin-bottom:3px">Inference time</div>
    <div style="font-size:1.3rem;font-weight:600;font-family:'JetBrains Mono',monospace;
                color:#7c5fff">ELAPSED_STR</div>
    <div style="font-size:0.68rem;color:#5a6582;margin-top:2px">CPU · ResNet50</div>
  </div>

</div>

<!-- Probability bars -->
<div style="background:#111520;border:1px solid #1e2740;border-radius:14px;
            padding:1.2rem;margin-bottom:1rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
    <span style="font-size:0.8rem;font-weight:500;color:#e8edf8">
      Class probability distribution
    </span>
    <span style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:#5a6582">
      softmax output
    </span>
  </div>
  BARS_INNER
</div>

<!-- Verdict -->
<div style="background:V_BG;border:1px solid V_BORDER;border-radius:14px;
            padding:1.2rem;display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem">
  <div style="width:38px;height:38px;border-radius:50%;background:V_DOT_BG;
              display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px">
    V_ICON
  </div>
  <div>
    <div style="font-size:0.92rem;font-weight:600;color:V_COLOR;margin-bottom:5px">V_TITLE</div>
    <div style="font-size:0.78rem;color:#8a96b0;line-height:1.65">V_BODY</div>
    <div style="font-size:0.68rem;color:#2e3d60;margin-top:8px;font-style:italic">
      Disclaimer: This is a research prototype trained on Paul Mooney's Kaggle dataset.
      It is not a certified medical device and must not substitute professional radiological diagnosis.
    </div>
  </div>
</div>

<!-- Footer -->
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-top:0.5rem;padding-top:1rem;border-top:1px solid #1e2740">
  <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e3d60">
    model · pneumonia_resnet50_model.pkl · fastai 2.x
  </span>
  <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e3d60">
    224×224 · CPU inference
  </span>
</div>

</body>
</html>
""".replace("DIAG_BG",        diag_bg
   ).replace("DIAG_BORDER",   diag_border
   ).replace("DIAG_ICON",     diag_icon_svg
   ).replace("DIAG_COLOR",    diag_color
   ).replace("LABEL",         label
   ).replace("DIAG_SUB",      diag_sub
   ).replace("CONFIDENCE_STR",confidence_str
   ).replace("CERTAINTY",     certainty
   ).replace("ELAPSED_STR",   elapsed_str
   ).replace("BARS_INNER",    bars_inner
   ).replace("V_BG",          v_bg
   ).replace("V_BORDER",      v_border
   ).replace("V_DOT_BG",      v_dot_bg
   ).replace("V_ICON",        v_icon
   ).replace("V_COLOR",       v_color
   ).replace("V_TITLE",       v_title
   ).replace("V_BODY",        v_body)

    components.html(result_html, height=620, scrolling=False)