"""
DocFusion — Enhanced Streamlit Web UI (Level 3)
==================================================
Features:
  - Upload a receipt image
  - Display extracted fields (vendor, date, total)
  - Anomaly detection with suspicious field highlighting
  - OCR word-level bounding boxes drawn on the image
  - JSON output view

Run:  streamlit run ui/app.py
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DocFusionPipeline
from src.anomaly.llm_summariser import generate_anomaly_summary

# ----- Page Config -----
st.set_page_config(
    page_title="DocFusion Dashboard",
    page_icon="📄",
    layout="wide",
)

# ----- Custom CSS -----
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .field-label { color: #a8b2d1; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }
    .field-value { color: #e6f1ff; font-size: 1.2em; font-weight: 600; }
    .status-genuine { background: #0d4228; border: 1px solid #2ecc71; border-radius: 8px; padding: 15px; text-align: center; }
    .status-forged  { background: #4a1c1c; border: 1px solid #e74c3c; border-radius: 8px; padding: 15px; text-align: center; }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Cache the pipeline so the Donut model only loads once."""
    return DocFusionPipeline(use_donut=True)


def draw_ocr_boxes(image: Image.Image) -> Image.Image:
    """
    Draw word-level bounding boxes from Tesseract on the image.
    Returns a copy with boxes drawn.
    """
    try:
        import pytesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
            if not text or conf < 30:
                continue

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

            # Color by confidence: green (high) → yellow → red (low)
            if conf >= 70:
                color = "#2ecc71"
            elif conf >= 50:
                color = "#f39c12"
            else:
                color = "#e74c3c"

            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        return img_copy
    except Exception:
        return image


# ----- Header -----
st.markdown('<p class="header-gradient">📄 DocFusion</p>', unsafe_allow_html=True)
st.markdown("**Operation Intelligent Documents** — Upload a receipt to extract data and detect anomalies.")
st.markdown("---")

# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Settings")
    show_boxes = st.toggle("Show OCR Bounding Boxes", value=True)
    show_raw_json = st.toggle("Show Raw JSON Output", value=False)
    st.markdown("---")
    st.markdown("**Pipeline:** Donut VDU + Tesseract fallback")
    st.markdown("**Anomaly:** IsolationForest + Rules")

# ----- Main Upload -----
uploaded_file = st.file_uploader(
    "Choose a document image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    pipeline = load_pipeline()
    image = Image.open(uploaded_file).convert("RGB")

    # Save temp image for the pipeline
    temp_path = "/tmp/docfusion_upload.jpg"
    image.save(temp_path)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Document")

        if show_boxes:
            with st.spinner("Running OCR bounding box detection..."):
                display_img = draw_ocr_boxes(image)
        else:
            display_img = image

        st.image(display_img, use_container_width=True)

    with col2:
        st.subheader("📊 Analysis Results")

        with st.spinner("Extracting fields & running anomaly detection..."):
            vendor, date, total = pipeline.extract(temp_path)
            is_forged = pipeline.predict_anomaly(vendor, date, total, temp_path)

        # --- Extracted Fields ---
        st.markdown("#### Extracted Fields")

        f1, f2, f3 = st.columns(3)
        with f1:
            st.metric("🏪 Vendor", vendor if vendor else "—")
        with f2:
            st.metric("📅 Date", date if date else "—")
        with f3:
            st.metric("💰 Total", f"${total}" if total else "—")

        # --- Field confidence indicators ---
        missing_fields = []
        if not vendor:
            missing_fields.append("Vendor")
        if not date:
            missing_fields.append("Date")
        if not total:
            missing_fields.append("Total")

        if missing_fields:
            st.warning(f"⚠️ Could not extract: {', '.join(missing_fields)}")

        st.markdown("---")

        # --- Anomaly Detection ---
        st.markdown("#### 🔍 Anomaly Detection")

        if is_forged == 1:
            st.markdown(
                '<div class="status-forged">'
                '<h3 style="color: #e74c3c; margin: 0;">🚨 SUSPICIOUS</h3>'
                '<p style="color: #ff6b6b; margin: 5px 0 0 0;">This document appears forged or highly anomalous.</p>'
                "</div>",
                unsafe_allow_html=True,
            )

            # Show reasons
            reasons = []
            if not vendor and not date and not total:
                reasons.append("All extraction fields are missing")
            if total:
                try:
                    tv = float(total)
                    if tv > 50000:
                        reasons.append(f"Unusually high total: ${total}")
                    if tv < 0:
                        reasons.append(f"Negative total: ${total}")
                except ValueError:
                    pass
            if reasons:
                st.markdown("**Suspicious indicators:**")
                for r in reasons:
                    st.markdown(f"- 🔴 {r}")
        else:
            st.markdown(
                '<div class="status-genuine">'
                '<h3 style="color: #2ecc71; margin: 0;">✅ GENUINE</h3>'
                '<p style="color: #82e6a8; margin: 5px 0 0 0;">This document appears normal and consistent.</p>'
                "</div>",
                unsafe_allow_html=True,
            )

        # --- LLM Anomaly Summary ---
        st.markdown("---")
        st.markdown("#### 🤖 AI Explanation")
        summary = generate_anomaly_summary(vendor, date, total, is_forged)
        st.markdown(summary)

        # --- Raw JSON ---
        if show_raw_json:
            st.markdown("---")
            st.markdown("#### 📋 Raw Output")
            st.json(
                {
                    "vendor": vendor,
                    "date": date,
                    "total": str(total) if total else None,
                    "is_forged": is_forged,
                }
            )

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    # Empty state
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #8892b0;">
            <p style="font-size: 4em; margin-bottom: 10px;">📄</p>
            <h3>Upload a receipt or invoice to get started</h3>
            <p>The pipeline will extract vendor, date, and total amount,<br>
            then check for signs of forgery or tampering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
