import streamlit as st
import requests
import base64
import json
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChestAI — Diagnostic Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
    
    /* Confidence bars */
    .finding-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
        gap: 0.75rem;
    }
    .finding-name {
        font-size: 0.85rem;
        color: #334155;
        width: 160px;
        flex-shrink: 0;
    }
    .bar-container {
        flex: 1;
        background: #f1f5f9;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .bar-fill-low { background: #22c55e; border-radius: 4px; height: 8px; }
    .bar-fill-mid { background: #f59e0b; border-radius: 4px; height: 8px; }
    .bar-fill-high { background: #ef4444; border-radius: 4px; height: 8px; }
    .finding-pct {
        font-size: 0.82rem;
        font-weight: 600;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }
    .pct-low { color: #16a34a; }
    .pct-mid { color: #d97706; }
    .pct-high { color: #dc2626; }
    
    /* Top diagnosis badge */
    .diagnosis-badge {
        display: inline-block;
        background: #eff6ff;
        border: 1.5px solid #3b82f6;
        color: #1d4ed8;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #92400e;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    
    /* Image labels */
    .image-label {
        text-align: center;
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.5rem;
    }
    
    /* Sidebar */
    .sidebar-metric {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3b82f6;
    }
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .sidebar-metric-value {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫁 ChestAI")
    st.markdown("---")
    
    api_url = st.text_input(
        "API Endpoint",
        value="https://saugatiwi-chest-xray-classifier.hf.space",
        help="FastAPI backend URL"
    )
    
    st.markdown("---")
    st.markdown("**Model Information**")
    
    try:
        info = requests.get(f"{api_url}/model-info", timeout=3).json()
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Architecture</div>
            <div class="sidebar-metric-value">EfficientNet-B3 + SE</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Mean AUC-ROC</div>
            <div class="sidebar-metric-value">{info['mean_auc_roc']:.4f}</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Training Images</div>
            <div class="sidebar-metric-value">{info['train_images']:,}</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Disease Classes</div>
            <div class="sidebar-metric-value">{info['num_classes']}</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Dataset</div>
            <div class="sidebar-metric-value">NIH ChestX-ray14</div>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.warning("⚠️ API offline")
    
    st.markdown("---")
    st.markdown("**Disease Classes**")
    try:
        for cls in info['disease_classes']:
            st.markdown(f"• {cls}")
    except:
        pass

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🫁 ChestAI Diagnostic Assistant</div>
    <div class="header-subtitle">
        AI-powered chest X-ray analysis · EfficientNet-B3 + SE Attention · NIH ChestX-ray14 · 
        Mean AUC-ROC: 0.7933 · For research use only
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([1, 1])

with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Chest X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a frontal chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown("""
    <div class="card">
        <div class="card-title">📋 Instructions</div>
        <ul style="color: #475569; font-size: 0.88rem; line-height: 1.8; padding-left: 1.2rem;">
            <li>Upload a <strong>frontal (PA or AP)</strong> chest X-ray</li>
            <li>Accepted formats: JPG, JPEG, PNG</li>
            <li>Model analyzes <strong>14 pathology classes</strong></li>
            <li>Grad-CAM highlights the region the model focused on</li>
            <li>Results appear within a few seconds</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_file:
    with st.spinner("🔬 Analyzing X-ray..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{api_url}/predict", files=files, timeout=60)
            result = response.json()

            predictions = result["predictions"]
            top_diagnosis = result["top_diagnosis"]
            confidence = result["confidence"]
            gradcam_b64 = result["gradcam_image"]

            # Decode Grad-CAM
            gradcam_bytes = base64.b64decode(gradcam_b64)
            gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
            original_img = Image.open(io.BytesIO(uploaded_file.getvalue()))

            st.markdown("---")

            # ── Top diagnosis ─────────────────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🩺 Primary Finding</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="diagnosis-badge">{top_diagnosis}</div>
            <div class="confidence-text">Model confidence: <strong>{confidence*100:.1f}%</strong></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Images ────────────────────────────────────────────────────────
            col_orig, col_grad = st.columns(2)
            with col_orig:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🖼️ Original X-Ray</div>', unsafe_allow_html=True)
                st.image(original_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_grad:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🔥 Grad-CAM Attention Map</div>', unsafe_allow_html=True)
                st.image(gradcam_img, use_container_width=True)
                st.markdown(
                    '<div style="font-size:0.78rem;color:#64748b;margin-top:0.4rem;">'
                    'Red/yellow regions indicate areas the model focused on for the primary diagnosis.'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Confidence bars ───────────────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 All Pathology Scores</div>', unsafe_allow_html=True)

            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            bars_html = ""
            for name, score in sorted_preds:
                pct = score * 100
                width = f"{pct:.1f}%"
                if pct < 30:
                    bar_class = "bar-fill-low"
                    pct_class = "pct-low"
                elif pct < 60:
                    bar_class = "bar-fill-mid"
                    pct_class = "pct-mid"
                else:
                    bar_class = "bar-fill-high"
                    pct_class = "pct-high"

                bars_html += f"""
                <div class="finding-row">
                    <div class="finding-name">{name}</div>
                    <div class="bar-container">
                        <div class="{bar_class}" style="width:{width};"></div>
                    </div>
                    <div class="finding-pct {pct_class}">{pct:.1f}%</div>
                </div>
                """

            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Disclaimer ────────────────────────────────────────────────────
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for 
                <strong>research and educational purposes only</strong>. 
                It is not a substitute for professional medical diagnosis. 
                Always consult a qualified radiologist or physician for clinical decisions.
                Model trained on NIH ChestX-ray14 dataset. Mean AUC-ROC: 0.7933.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Make sure the FastAPI backend is running at the API endpoint shown in the sidebar.")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #94a3b8;">
        <div style="font-size: 4rem;">🫁</div>
        <div style="font-size: 1.1rem; margin-top: 1rem; font-weight: 500;">
            Upload a chest X-ray to begin analysis
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            Supports JPG, JPEG, PNG formats
        </div>
    </div>
    """, unsafe_allow_html=True)
