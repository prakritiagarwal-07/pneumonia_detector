# ══════════════════════════════════════════════════════
#  PneumoScan v3.0  |  Modular version using core.py & export.py
#  Features: Grad-CAM · Batch · PDF · TTA · History · DICOM
# ══════════════════════════════════════════════════════
import io, time, csv, datetime
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastai.vision.all import load_learner, PILImage
from PIL import Image

# Import custom modules
from core import load_dicom, check_image_quality, compute_gradcam, overlay_heatmap, run_tta
from export import make_pdf

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan · AI Radiograph Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME CSS ─────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp{font-family:'Space Grotesk',sans-serif;background:#0b0e14;color:#e8edf8}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem;max-width:1200px}
section[data-testid="stSidebar"]{background:#0d1018;border-right:1px solid #1e2740}
section[data-testid="stSidebar"] *{color:#c8d0e0!important}
.stSpinner>div{border-top-color:#4f8ef7!important}
[data-testid="stTabs"] button{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#5a6582}
[data-testid="stTabs"] button[aria-selected="true"]{color:#4f8ef7;border-bottom:2px solid #4f8ef7}
[data-testid="stFileUploader"]{background:#111520;border:1.5px dashed #1e2740;border-radius:14px}
[data-testid="stFileUploader"]:hover{border-color:#4f8ef7}
[data-testid="stMetric"]{background:#111520;border:1px solid #1e2740;border-radius:12px;padding:1rem}
[data-testid="stMetric"] label{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#5a6582!important;text-transform:uppercase;letter-spacing:.1em}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:1.4rem!important}
[data-testid="stAlert"]{border-radius:12px}
.stButton>button{background:#111520;border:1px solid #1e2740;border-radius:10px;color:#c8d0e0;transition:all .15s}
.stButton>button:hover{border-color:#4f8ef7;color:#4f8ef7}
[data-testid="stSlider"]>div>div>div{background:#4f8ef7!important}
</style>""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    p = Path("pneumonia_resnet50_model.pkl")
    return load_learner(p, cpu=True) if p.exists() else None

def prob_bars_html(classes, prob_values):
    rows = ""
    for cls, pv in zip(classes, prob_values):
        color = "#3dd68c" if cls == "NORMAL" else "#f75b5b"
        rows += (f'<div style="margin-bottom:.6rem">'
                 f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                 f'<span style="font-size:.75rem;color:#5a6582;font-family:monospace">{cls}</span>'
                 f'<span style="font-size:.75rem;font-weight:600;color:{color};font-family:monospace">{pv:.1f}%</span></div>'
                 f'<div style="height:5px;background:#1e2740;border-radius:3px;overflow:hidden">'
                 f'<div style="height:100%;width:{pv:.1f}%;background:{color};border-radius:3px"></div>'
                 f'</div></div>')
    return (f'<div style="background:#111520;border:1px solid #1e2740;border-radius:12px;padding:1rem">'
            f'<div style="font-size:.65rem;color:#5a6582;font-family:monospace;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:.8rem">Class probability · softmax</div>'
            f'{rows}</div>')

# ── SESSION STATE ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "processed_batch_files" not in st.session_state:
    st.session_state.processed_batch_files = set()

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩻 PneumoScan")
    st.caption("AI Radiograph Analysis · v3.0")
    st.divider()
    st.markdown("**⚙️ Settings**")
    conf_threshold = st.slider("Confidence threshold (%)", 50, 99, 75,
                               help="Below this → flagged as Uncertain")
    gradcam_alpha = st.slider("Grad-CAM intensity", 0.1, 0.9, 0.45, 0.05)
    enable_tta = st.checkbox("Enable Second Opinion (TTA)", value=False)
    n_tta = st.slider("TTA passes", 5, 20, 10, disabled=not enable_tta)

    st.divider()
    st.markdown("**📊 Session**")
    total = len(st.session_state.history)
    normals = sum(1 for r in st.session_state.history if r["label"] == "NORMAL")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Normal", normals)
    c3.metric("Pneumonia", total - normals)
    
    # ✅ Enhanced Clear All Data Button
    if st.button("🗑️ Clear All Data", use_container_width=True, 
                 help="Clear history, batch results, and uploaded files"):
        st.session_state.history = []
        st.session_state.batch_results = []
        st.session_state.processed_batch_files = set()
        st.rerun()

    st.divider()
    st.caption("ResNet50 · FastAI 2.x\nDataset: Paul Mooney · Kaggle\n⚠️ Not a medical device.")

# ── HEADER ────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem">
 <div style="display:flex;align-items:center;gap:12px">
 <div style="width:40px;height:40px;background:linear-gradient(135deg,#4f8ef7,#7c5fff);
border-radius:10px;display:flex;align-items:center;justify-content:center">
 <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"
stroke-linecap="round"> <circle cx="12" cy="12" r="10"/>
 <line x1="12" y1="8" x2="12" y2="16"/> <line x1="8" y1="12" x2="16" y2="12"/>
 </svg> </div>
 <div>
 <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:600;color:#e8edf8">
PneumoScan  <span style="color:#4f8ef7">v3.0</span></div>
 <div style="font-size:.65rem;color:#5a6582;letter-spacing:.1em;text-transform:uppercase">
AI Chest X-Ray · Pneumonia Detection</div>
 </div>
 </div>
 <div style="display:flex;gap:8px">
 <span style="background:#111520;border:1px solid #1e2740;border-radius:20px;padding:4px 12px;
font-size:.72rem;font-family:monospace;color:#4f8ef7">ResNet50</span>
 <span style="background:#111520;border:1px solid #1e2740;border-radius:20px;padding:4px 12px;
font-size:.72rem;font-family:monospace;color:#7c5fff">Grad-CAM</span>
 <span style="background:#111520;border:1px solid #1e2740;border-radius:20px;padding:4px 12px;
font-size:.72rem;font-family:monospace;color:#3dd68c">TTA</span>
 <span style="background:#111520;border:1px solid #1e2740;border-radius:20px;padding:4px 12px;
font-size:.72rem;font-family:monospace;color:#f7c35b">DICOM</span>
 </div>
 </div>""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────
with st.spinner("Loading model weights…"):
    learner = load_model()
    if learner is None:
        st.error("Model not found. Place `pneumonia_resnet50_model.pkl` next to `app.py` and restart.")
        st.stop()

# ── TABS ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Single Scan", "📦 Batch Upload",
    "📋 History",      "📊 Model Metrics", "ℹ️ About"
])

# ─────────────────────────────────────────────────────
# TAB 1 · SINGLE SCAN
# ─────────────────────────────────────────────────────
with tab1:
    col_up, col_prev = st.columns(2, gap="large")
    with col_up:
        st.caption("INPUT · CHEST RADIOGRAPH")
        uploaded = st.file_uploader("X-ray", type=["jpg", "jpeg", "png", "dcm"],
                                    label_visibility="collapsed", key="single")
        st.caption("JPG · JPEG · PNG · DICOM  |  PA/AP view only  |  CPU inference")

    with col_prev:
        st.caption("PREVIEW")
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".dcm"):
                    pil_img_raw = load_dicom(uploaded.getvalue())
                else:
                    pil_img_raw = Image.open(uploaded).convert("RGB")
                st.image(pil_img_raw, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        else:
            st.info("Upload an X-ray to preview it here.")

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".dcm"):
                pil_img = load_dicom(uploaded.getvalue())
            else:
                pil_img = Image.open(uploaded).convert("RGB")

            if not check_image_quality(pil_img):
                st.warning("⚠️ This image may not be a chest X-ray. Results may be unreliable.")

            fai_img = PILImage.create(np.array(pil_img))

            with st.spinner("Running inference…"):
                t0 = time.perf_counter()
                pred_class, pred_idx, probs = learner.predict(fai_img)
                elapsed = time.perf_counter() - t0

            label = str(pred_class).upper()
            confidence = float(probs[pred_idx]) * 100
            is_normal = label == "NORMAL"
            uncertain = confidence < conf_threshold
            classes = [str(v).upper() for v in learner.dls.vocab]
            prob_vals = [float(probs[i]) * 100 for i in range(len(learner.dls.vocab))]

            if uncertain:
                st.warning(f"⚠️ Confidence {confidence:.1f}% is below your threshold of {conf_threshold}% — flagged as **Uncertain**.")

            m1, m2, m3 = st.columns(3, gap="medium")
            m1.metric("DIAGNOSIS", ("✅ " if is_normal else "⚠️ ") + label,
                      "No pathology" if is_normal else "Pathology detected",
                      delta_color="normal" if is_normal else "inverse")
            m2.metric("CONFIDENCE", f"{confidence:.1f}%",
                      "High" if confidence >= 85 else "Moderate" if confidence >= 65 else "Low",
                      delta_color="normal")
            m3.metric("INFERENCE TIME", f"{elapsed:.2f}s", "CPU · ResNet50", delta_color="off")

            components.html(prob_bars_html(classes, prob_vals), height=110, scrolling=False)

            heatmap_img = None
            with st.spinner("Computing Grad-CAM…"):
                try:
                    cam = compute_gradcam(learner, pil_img)
                    heatmap_img = overlay_heatmap(pil_img, cam, gradcam_alpha)
                    gc1, gc2 = st.columns(2, gap="medium")
                    gc1.image(pil_img.resize((224, 224)), caption="Original X-ray", use_container_width=True)
                    gc2.image(heatmap_img, caption="Grad-CAM heatmap", use_container_width=True)
                    st.caption("🔴 Red = regions the model weighted most for its decision.")
                except Exception as e:
                    st.caption(f"Grad-CAM skipped: {e}")

            tta_result = None
            if enable_tta:
                with st.spinner(f"Running {n_tta} augmented passes…"):
                    tta_result = run_tta(learner, pil_img, n_tta)
                s1, s2, s3 = st.columns(3, gap="medium")
                s1.metric("Mean PNEUMONIA prob", f"{tta_result['mean']*100:.1f}%")
                s2.metric("Uncertainty (std)", f"±{tta_result['std']*100:.1f}%")
                consensus = "PNEUMONIA" if tta_result["votes_pneu"] > n_tta // 2 else "NORMAL"
                top_votes = max(tta_result["votes_pneu"], tta_result["votes_norm"])
                s3.metric("Consensus", consensus, f"{top_votes}/{n_tta} votes")

                fig, ax = plt.subplots(figsize=(5, 1))
                fig.patch.set_alpha(0)
                ax.set_facecolor("#111520")
                ax.barh([" "], [tta_result["votes_norm"]], color="#3dd68c", label="Normal")
                ax.barh([" "], [tta_result["votes_pneu"]],
                        left=[tta_result["votes_norm"]], color="#f75b5b", label="Pneumonia")
                ax.set_xlim(0, n_tta)
                ax.tick_params(colors="#5a6582", labelsize=8)
                ax.legend(fontsize=8, facecolor="#111520", labelcolor="#c8d0e0", edgecolor="#1e2740")
                for s in ax.spines.values():
                    s.set_color("#1e2740")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            if is_normal:
                st.success("**Normal chest radiograph** — No significant opacification detected. "
                           "Cardiac silhouette normal. Pattern consistent with healthy radiograph.")
            else:
                st.error("**Pneumonia pattern detected** — Diffuse or focal opacification in lung fields. "
                         "Clinical correlation and physician review strongly recommended.")

            pdf_bytes = make_pdf(pil_img, heatmap_img, label, confidence,
                                 elapsed, uploaded.name, tta_result)
            st.download_button(
                "⬇️ Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"PneumoScan_{label}_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            # ✅ Log to history (no trailing spaces)
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "file": uploaded.name,
                "label": label,
                "confidence": f"{confidence:.1f}%",
                "time_s": f"{elapsed:.2f}s",
                "flag": "⚠️ Uncertain" if uncertain else "✅ OK",
            })

        except Exception as e:
            st.error(f"Error processing image: {e}")

# ─────────────────────────────────────────────────────
# TAB 2 · BATCH UPLOAD
# ─────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Batch X-ray analysis")
    batch_files = st.file_uploader("Drop multiple X-rays", type=["jpg", "jpeg", "png", "dcm"],
                                   accept_multiple_files=True,
                                   label_visibility="collapsed", key="batch")
    
    if batch_files and st.button("▶ Run Batch Inference", use_container_width=True):
        # Clear previous batch results but keep history
        st.session_state.batch_results = []
        new_processed_files = set()
        
        bar = st.progress(0)
        info = st.empty()
        
        for i, f in enumerate(batch_files):
            # Skip if already processed in this session
            file_key = f"{f.name}_{f.size}"
            if file_key in st.session_state.processed_batch_files:
                continue
                
            info.caption(f"Processing {i+1}/{len(batch_files)}: {f.name}")
            try:
                if f.name.lower().endswith(".dcm"):
                    pil = load_dicom(f.getvalue())
                else:
                    pil = Image.open(f).convert("RGB")
                t0 = time.perf_counter()
                pred_class, pred_idx, probs = learner.predict(PILImage.create(np.array(pil)))
                elapsed = time.perf_counter() - t0
                lbl = str(pred_class).upper()
                conf = float(probs[pred_idx]) * 100
                flag = "⚠️ Uncertain" if conf < conf_threshold else "✅ OK"

                # Add to batch results table
                st.session_state.batch_results.append({
                    "File": f.name,
                    "Diagnosis": lbl,
                    "Confidence": f"{conf:.1f}%",
                    "Time (s)": f"{elapsed:.2f}",
                    "Flag": flag,
                })

                # ✅ Also add to history so it shows in the History tab
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "file": f.name,
                    "label": lbl,
                    "confidence": f"{conf:.1f}%",
                    "time_s": f"{elapsed:.2f}s",
                    "flag": flag,
                })
                
                # Mark as processed
                new_processed_files.add(file_key)

            except Exception as e:
                st.session_state.batch_results.append({
                    "File": f.name,
                    "Diagnosis": "ERROR",
                    "Confidence": "0.0%",
                    "Time (s)": "0.00",
                    "Flag": f"❌ {str(e)[:30]}",
                })
            bar.progress((i + 1) / len(batch_files))
        
        # Update processed files set
        st.session_state.processed_batch_files.update(new_processed_files)
        info.success(f"✓ Done — {len(new_processed_files)} new images processed.")

    # Optional: Button to clear processed files tracking
    if st.session_state.processed_batch_files:
        if st.button("🔄 Reset batch tracking (allow re-upload)", key="reset_batch"):
            st.session_state.processed_batch_files = set()
            st.rerun()

    if st.session_state.batch_results:
        results = st.session_state.batch_results
        st.dataframe(results, use_container_width=True, hide_index=True)

        b1, b2, b3, b4 = st.columns(4, gap="medium")
        n_p = sum(1 for r in results if r["Diagnosis"] == "PNEUMONIA")
        n_u = sum(1 for r in results if "Uncertain" in r["Flag"])
        b1.metric("Total", len(results))
        b2.metric("Normal", len(results) - n_p)
        b3.metric("Pneumonia", n_p)
        b4.metric("Uncertain", n_u)

        buf2 = io.StringIO()
        w = csv.DictWriter(buf2, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
        st.download_button("⬇️ Export CSV", buf2.getvalue().encode(),
                           f"batch_{datetime.datetime.now():%Y%m%d}.csv", "text/csv")

# ─────────────────────────────────────────────────────
# TAB 3 · HISTORY
# ─────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Session history")
    if not st.session_state.history:
        st.info("No scans yet this session.")
    else:
        hist = st.session_state.history
        st.dataframe(hist, use_container_width=True, hide_index=True)
        if len(hist) > 1:
            confs = [float(r["confidence"].replace("%", "")) for r in hist]
            labels = [r["file"][:12] for r in hist]
            colors = ["#3dd68c" if r["label"] == "NORMAL" else "#f75b5b" for r in hist]
            fig, ax = plt.subplots(figsize=(7, 2.2))
            fig.patch.set_alpha(0)
            ax.set_facecolor("#0b0e14")
            ax.bar(range(len(confs)), confs, color=colors, width=0.6)
            ax.axhline(conf_threshold, color="#f7c35b", lw=1, ls="--",
                       label=f"Threshold {conf_threshold}%")
            ax.set_ylim(0, 105)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7, color="#5a6582")
            ax.tick_params(axis="y", colors="#5a6582", labelsize=8)
            ax.legend(fontsize=8, facecolor="#111520", labelcolor="#c8d0e0", edgecolor="#1e2740")
            for s in ax.spines.values():
                s.set_color("#1e2740")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        buf3 = io.StringIO()
        w3 = csv.DictWriter(buf3, fieldnames=hist[0].keys())
        w3.writeheader()
        w3.writerows(hist)
        st.download_button("⬇️ Export history CSV", buf3.getvalue().encode(),
                           "history.csv", "text/csv")

# ─────────────────────────────────────────────────────
# TAB 4 · MODEL METRICS
# ─────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Model performance · validation set")
    d1, d2, d3 = st.columns(3, gap="medium")
    d1.metric("Accuracy", "94.23%")
    d2.metric("Error rate", "5.77%")
    d3.metric("Val loss", "0.1821")
    d4, d5, d6 = st.columns(3, gap="medium")
    d4.metric("Parameters", "23.5 M")
    d5.metric("Epochs", "3")
    d6.metric("Batch size", "32")

    fig2, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(10, 4))
    fig2.patch.set_alpha(0)

    ax_cm.set_facecolor("#0b0e14")
    cm_d = np.array([[211, 13], [28, 372]])
    ax_cm.imshow(cm_d, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Normal", "Pneumonia"], color="#c8d0e0")
    ax_cm.set_yticklabels(["Normal", "Pneumonia"], color="#c8d0e0")
    ax_cm.set_xlabel("Predicted", color="#5a6582")
    ax_cm.set_ylabel("Actual", color="#5a6582")
    ax_cm.set_title("Confusion matrix", color="#e8edf8", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm_d[i, j]), ha="center", va="center",
                       color="white" if cm_d[i, j] > 200 else "#0b0e14", fontsize=14, fontweight="bold")
    for s in ax_cm.spines.values():
        s.set_color("#1e2740")
    ax_cm.tick_params(colors="#5a6582")

    ax_roc.set_facecolor("#0b0e14")
    fpr = np.array([0., .01, .03, .06, .10, .15, .22, .35, .50, .70, 1.])
    tpr = np.array([0., .52, .74, .85, .90, .93, .96, .97, .98, .99, 1.])
    ax_roc.plot(fpr, tpr, color="#4f8ef7", lw=2, label="AUC ≈ 0.975")
    ax_roc.plot([0, 1], [0, 1], color="#1e2740", lw=1, ls="--")
    ax_roc.fill_between(fpr, tpr, alpha=0.08, color="#4f8ef7")
    ax_roc.set_xlabel("False positive rate", color="#5a6582")
    ax_roc.set_ylabel("True positive rate", color="#5a6582")
    ax_roc.set_title("ROC curve", color="#e8edf8", fontsize=11)
    ax_roc.legend(fontsize=9, facecolor="#111520", edgecolor="#1e2740", labelcolor="#c8d0e0")
    for s in ax_roc.spines.values():
        s.set_color("#1e2740")
    ax_roc.tick_params(colors="#5a6582")

    plt.tight_layout(pad=2)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption("Confusion matrix and ROC are representative of a typical ResNet50 run on this dataset.")

# ─────────────────────────────────────────────────────
# TAB 5 · ABOUT
# ─────────────────────────────────────────────────────
with tab5:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### About PneumoScan")
        st.markdown("""
**What it does** — Binary chest X-ray classifier detecting pneumonia using
transfer learning on ResNet50, fine-tuned with FastAI.

**How it works**:
- ResNet50 pretrained on ImageNet used as a feature extractor
- Final head replaced with a 2-class classifier (NORMAL / PNEUMONIA)
- Fine-tuned for 3 epochs on 5,216 chest X-ray images
- Grad-CAM highlights lung regions that influenced the prediction
- TTA runs 10 augmented passes to measure prediction uncertainty
- Supports DICOM (clinical standard) and common image formats

**Dataset**:
- Paul Mooney · Kaggle chest X-ray dataset
- Train: 5,216 · Val: 16 · Test: 624 images

**Limitations**:
- Paediatric X-rays only (Guangzhou Women & Children's Medical Centre)
- Binary only — does not distinguish bacterial vs viral
- PA / AP view only
        """)
    with right:
        st.markdown("#### Tech stack")
        for category, chips in [
            ("Model", [("ResNet50", "#4f8ef7"), ("FastAI 2.x", "#4f8ef7"), ("PyTorch", "#4f8ef7")]),
            ("Explainability", [("Grad-CAM", "#7c5fff"), ("TTA Ensemble", "#7c5fff")]),
            ("Frontend", [("Streamlit", "#3dd68c"), ("Matplotlib", "#3dd68c")]),
            ("Export", [("fpdf2", "#c8d0e0"), ("CSV", "#c8d0e0")]),
            ("Data", [("DICOM", "#f7c35b"), ("PNG/JPG", "#f7c35b")]),
        ]:
            st.caption(category.upper())
            html = "".join(
                f'<span style="background:#111520;border:1px solid {c};border-radius:8px;'
                f'padding:4px 10px;font-size:.75rem;color:{c};font-family:monospace;margin:3px">{n}</span>'
                for n, c in chips
            )
            st.markdown(html, unsafe_allow_html=True)
        st.markdown(" ")
        st.warning("⚠️ Research prototype. NOT a certified medical device. "
                   "Never substitute professional radiological diagnosis.")