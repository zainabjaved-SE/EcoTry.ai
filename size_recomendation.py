# pages/size_recommendation.py
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from utils import get_body_measurements
from size_model import recommend_size


ECO_DEEP = "#2E7D32"   # primary green
ECO_LIME = "#81C784"   # accent mint
BG0 = "#F4FBF6"        # light background
BG1 = "#E8F5E9"        # soft green background


def _read_image_bytes(uploaded_file) -> Optional[np.ndarray]:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img


def _img_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def set_ecotry_theme(logo_path: str = "static/assets/ecotry-logo.png") -> None:
    st.set_page_config(
        page_title="EcoTry — AI Size Recommendation",
        page_icon="🧵",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    logo_file = Path(logo_path)
    logo_b64 = _img_to_base64(logo_file) if logo_file.exists() else ""

    st.markdown(
        f"""
        <style>
          :root {{
            --eco-deep: #2E7D32;
            --eco-lime: #81C784;

            --bg0: #F4FBF6;
            --bg1: #E8F5E9;

            --panel: rgba(255,255,255,0.75);
            --panel2: rgba(255,255,255,0.9);

            --text: #1f2d1f;
            --muted: #5f6f5f;

            --line: rgba(0,0,0,0.08);

            --shadow: 0 10px 30px rgba(0,0,0,0.08);

            --r: 16px;
          }}

          html, body, [data-testid="stAppViewContainer"] {{
              background: linear-gradient(180deg, #E8F5E9 0%, #F4FBF6 100%) !important;
              color: var(--text) !important;
          }}

          [data-testid="stHeader"] {{
            background: transparent !important;
          }}

          [data-testid="stSidebar"] {{
            display: none;
          }}

          .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }}

          .ecotry-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 16px;
            border: 1px solid var(--line);
            background: rgba(0,0,0,0.25);
            border-radius: var(--r);
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
          }}

          .ecotry-brand {{
            display:flex;
            align-items:center;
            gap: 12px;
          }}

          .ecotry-brand img {{
            width: 52px;
            height: 52px;
            object-fit: cover;
            border-radius: 50%;
            border: 2px solid #81C784;
            padding: 3px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
          }}

          .ecotry-title {{
            margin: 0;
            font-size: 16px;
            letter-spacing: .3px;
            font-weight: 800;
          }}

          .ecotry-sub {{
            margin: 2px 0 0;
            font-size: 12px;
            color: var(--muted);
          }}

          .chip {{
            font-size: 12px;
            padding: 9px 11px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
            color: var(--muted);
            white-space: nowrap;
          }}

          .panel {{
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 18px;
          }}

          .panel h2 {{
            margin: 0 0 6px;
            font-size: 16px;
          }}

          .muted {{
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
          }}

          .btn-primary button {{
            background: #2E7D32 !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 10px 14px !important;
          }}

          .btn-primary button:hover {{
            background: #256628 !important;
          }}

          /* Inputs */
          [data-testid="stNumberInput"] input,
          [data-testid="stTextInput"] input,
          [data-testid="stSelectbox"] div {{
            border-radius: 14px !important;
          }}

          /* File uploader */
          section[data-testid="stFileUploaderDropzone"] {{
            border: 1px dashed rgba(255,255,255,0.20) !important;
            background: rgba(255,255,255,0.04) !important;
            border-radius: 18px !important;
          }}

          .result-card {{
            border: 1px solid rgba(46,125,50,0.35);
            background: rgba(129,199,132,0.20);
            border-radius: 16px;
            padding: 16px;
          }}

          .warn-card {{
            border: 1px solid rgba(180,120,120,0.4);
            background: rgba(180,120,120,0.15);
            border-radius: 18px;
            padding: 14px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.main {
    flex: 1;
}

.footer {
   position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;

  margin-top: 40px;
  padding: 18px;

  background: #E8F5E9;
  color: #1f2d1f;

  text-align: center;
  font-size: 14px;
  border-top: 1px solid rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="EcoTry logo" />'
        if logo_b64
        else '<div style="width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg, rgba(123,220,59,0.9), rgba(11,61,46,0.9));"></div>'
    )

    st.markdown(
        f"""
        <div class="ecotry-nav">
          <div class="ecotry-brand">
            {logo_html}
            <div>
              <div class="ecotry-title">EcoTry</div>
              <div class="ecotry-sub">AI Size Recommendation — Think. Try. Sustain.</div>
            </div>
          </div>
          <div class="chip">Tip: Use a front-facing full-body photo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_ecotry_theme()

    st.write("")

    # Single column layout
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("<h2>Upload & Details</h2>", unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Upload a clear full-body image. Enter height/weight. We estimate body measurements and recommend the best fit.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        help="Front view, good lighting, full body visible.",
    )

    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (cm)", min_value=120, max_value=230, value=170, step=1)
    with c2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65, step=1)

    gender_option = st.selectbox("Gender", ["Female", "Male"], index=0)
    gender = 0 if gender_option == "Female" else 1

    st.write("")
    run = st.button("Generate Recommendation", type="primary")

    # 🔽 PREVIEW SECTION 
    st.write("")
    st.markdown("<h2>Preview & Result</h2>", unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">We show the preview here. If pose detection fails, try a clearer image.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="Uploaded image preview", use_container_width=True)
        else:
            st.markdown('<div class="warn-card">Could not decode the image.</div>', unsafe_allow_html=True)

    if run:
        if not uploaded_file:
            st.markdown('<div class="warn-card">Please upload an image first.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing pose and estimating measurements..."):
                img = _read_image_bytes(uploaded_file)
                if img is None:
                    st.markdown('<div class="warn-card">Image decode failed.</div>', unsafe_allow_html=True)
                else:
                    measurements = get_body_measurements(img, float(height))
                    if measurements:
                        size = recommend_size(measurements, float(height), float(weight), int(gender))
                        st.markdown(
                            f"""
                            <div class="result-card">
                              <div style="font-size:12px;color:rgba(255,255,255,0.70);">Recommended Size</div>
                              <div style="font-size:28px;font-weight:900;letter-spacing:.5px;">{size}</div>
                              <div class="muted" style="margin-top:6px;">
                                Based on estimated measurements + height/weight.
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander("See estimated measurements"):
                          if measurements:
                           shoulder = measurements.get("shoulder", "-")
                           chest = measurements.get("chest", "-")
                           waist = measurements.get("waist", "-")
                           hip = measurements.get("hip", "-")
                           chest_depth = measurements.get("chest_depth", "-")
                           hip_depth = measurements.get("hip_depth", "-")

                           st.write(f"Shoulder: {shoulder} cm")
                           st.write(f"Chest: {chest} cm")
                           st.write(f"Waist: {waist} cm")
                           st.write(f"Hip: {hip} cm")
                           st.write(f"Chest Depth: {chest_depth} cm")
                           st.write(f"Hip Depth: {hip_depth} cm")
                    else:
                        st.markdown(
                            """
                            <div class="warn-card">
                              Pose not detected properly.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="footer">
    © 2026 EcoTry • Think. Try. Sustain. • Final Year Project
</div>
""", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()