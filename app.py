"""
app.py — Streamlit Colorization Dashboard
==========================================
Responsive frontend client for the Image Colorization Engine.
Communicates with the FastAPI backend via HTTP POST to /translate,
displaying side-by-side grayscale input vs. colorized RGB output
with real-time inference telemetry and quality metrics.

Launch:  streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# ─── Background Server Autostart ──────────────────────────────────────────
# Automatically starts main.py if backend server is not detected (for single-container cloud hosting)
try:
    requests.get("http://localhost:4500/health", timeout=1)
except Exception:
    # Launch main.py in the background using the same Python interpreter
    subprocess.Popen([sys.executable, "main.py"])
    time.sleep(6)

BACKEND_URL = "http://localhost:4500"

# ─── Page Configuration ───────────────────────────────────────────────────

st.set_page_config(
    page_title="ChromaSynth: Neural Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.35rem 0;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #e5e7eb;
        font-size: 1.45rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }

    .metric-value.good { color: #34d399; }
    .metric-value.warn { color: #fbbf24; }
    .metric-value.bad  { color: #f87171; }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .status-online {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .status-offline {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    /* Pipeline stage indicator */
    .pipeline-stage {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .pipeline-stage .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
    }

    /* Divider */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    # Health check button
    st.markdown("### 🏥 System Status")

    if st.button("🔍 Check Backend Health", use_container_width=True):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                st.markdown(
                    '<span class="status-badge status-online">● Online</span>',
                    unsafe_allow_html=True,
                )
                
                # Fetch metrics safely to avoid NoneType format string issues
                gpu_allocated = health.get("gpu_allocated_mb")
                gpu_allocated_str = f"{gpu_allocated:.1f} MB" if gpu_allocated is not None else "N/A"
                ram_used = health.get("ram_used_mb")
                ram_used_str = f"{ram_used:.1f} MB" if ram_used is not None else "N/A"
                
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">GPU Device</div>
                        <div class="metric-value">{health.get("gpu", "N/A")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">VRAM Allocated</div>
                        <div class="metric-value">{gpu_allocated_str}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">RAM Used</div>
                        <div class="metric-value">{ram_used_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge status-offline">● Unhealthy</span>',
                    unsafe_allow_html=True,
                )
                st.error(f"Status code: {resp.status_code}")
        except requests.ConnectionError:
            st.markdown(
                '<span class="status-badge status-offline">● Offline</span>',
                unsafe_allow_html=True,
            )
            st.error(
                "Cannot reach backend. Ensure the FastAPI server is running:\n\n"
                "`python main.py`"
            )
        except Exception as e:
            st.error(f"Health check failed: {e}")

# ─── Main Content & Tabs ──────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🎨 ChromaSynth</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Transform grayscale images into vivid color using deep learning."
    "</p>",
    unsafe_allow_html=True,
)

tab_app, tab_pipeline = st.tabs(["🎨 Colorize Image", "🔬 Pipeline & Architecture"])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1: COLORIZE IMAGE
# ──────────────────────────────────────────────────────────────────────────
with tab_app:
    uploaded_file = st.file_uploader(
        "📁 Upload a grayscale or black-and-white image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPEG. Any resolution — the model resizes internally to 256×256.",
    )

    # Manage session state for the uploaded file to ensure persistent output
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # Reset session state if a brand new file is uploaded
        if st.session_state.get("last_uploaded_name") != file_name:
            st.session_state["last_uploaded_name"] = file_name
            st.session_state["colorized_bytes"] = None
            st.session_state["telemetry_headers"] = None

        # Display preview of original image
        st.image(uploaded_file, caption="Uploaded Original", width=300)

        # Colourise Button
        if st.button("🎨 Colourise", use_container_width=True, type="primary"):
            status_container = st.empty()
            status_container.markdown(
                '<div class="pipeline-stage"><span class="dot"></span> Sending image to GPU backend...</div>',
                unsafe_allow_html=True,
            )

            try:
                response = requests.post(
                    f"{BACKEND_URL}/translate",
                    files={"file": (file_name, file_bytes, uploaded_file.type)},
                    timeout=30,
                )

                if response.status_code == 200:
                    status_container.markdown(
                        '<div class="pipeline-stage"><span class="dot"></span> Colorization complete!</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state["colorized_bytes"] = response.content
                    st.session_state["telemetry_headers"] = {k.lower(): v for k, v in response.headers.items()}
                else:
                    status_container.empty()
                    error_detail = "Unknown error"
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", error_json.get("error", str(error_json)))
                    except Exception:
                        error_detail = response.text[:500]
                    st.error(f"**Backend returned {response.status_code}:** {error_detail}")

            except requests.ConnectionError:
                status_container.empty()
                st.error(
                    "**Cannot connect to the backend server.**\n\n"
                    "Make sure the FastAPI server is running.\n"
                    f"Expected at: `{BACKEND_URL}`"
                )
            except requests.Timeout:
                status_container.empty()
                st.error("**Request timed out** after 30 seconds.")
            except Exception as e:
                status_container.empty()
                st.error(f"**Unexpected error:** {str(e)}")

        # Render output if it exists in session state
        if st.session_state.get("colorized_bytes") is not None:
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            
            # Side-by-side layout
            col_left, col_right = st.columns(2)
            input_image = Image.open(io.BytesIO(file_bytes))
            colorized_image = Image.open(io.BytesIO(st.session_state["colorized_bytes"]))

            with col_left:
                st.markdown("#### 🖤 Original Input")
                st.image(
                    input_image,
                    use_container_width=True,
                    caption=f"Original — {input_image.size[0]}×{input_image.size[1]}",
                )

            with col_right:
                st.markdown("#### 🌈 Colorized Output")
                st.image(
                    colorized_image,
                    use_container_width=True,
                    caption=f"Colorized — {colorized_image.size[0]}×{colorized_image.size[1]}",
                )

            # Telemetry Metrics Panel
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            with st.expander("📊 **Inference Telemetry & Performance Metrics**", expanded=True):
                headers = st.session_state["telemetry_headers"]

                m1, m2, m3, m4 = st.columns(4)
                latency = headers.get("x-inference-latency-ms", "N/A")
                ram_delta = headers.get("x-ram-delta-mb", "N/A")
                vram_delta = headers.get("x-vram-delta-mb", "N/A")
                orig_size = headers.get("x-original-size", "N/A")

                latency_class = "good"
                if latency != "N/A":
                    lat_val = float(latency)
                    if lat_val > 1000:
                        latency_class = "bad"
                    elif lat_val > 500:
                        latency_class = "warn"

                with m1:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Latency</div><div class="metric-value {latency_class}">{latency} ms</div></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">VRAM Delta</div><div class="metric-value">{vram_delta} MB</div></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">RAM Delta</div><div class="metric-value">{ram_delta} MB</div></div>',
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Original Size</div><div class="metric-value">{orig_size}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="metric-card" style="margin-top: 0.5rem;"><div class="metric-label">Compute Device</div><div class="metric-value">{headers.get("x-model-device", "N/A")}</div></div>',
                    unsafe_allow_html=True,
                )

            # Download Button
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Colorized Image",
                data=st.session_state["colorized_bytes"],
                file_name=f"colorized_{file_name.rsplit('.', 1)[0]}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 4rem 2rem;
                color: #6b7280;
                border: 2px dashed rgba(102, 126, 234, 0.2);
                border-radius: 16px;
                margin: 2rem 0;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                <div style="font-size: 1.15rem; font-weight: 500; color: #9ca3af;">
                    Drop a grayscale image above and press "Colourise"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────
# TAB 2: PIPELINE EXPLANATION
# ──────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("### 🔬 System Architecture & CIELAB Pipeline")
    
    st.markdown(
        """
        The engine maps a single-channel Lightness ($L$) layer to 2-channel ($a$, $b$) chrominance attributes 
        in the **CIELAB color space** using an 8-block convolutional neural network (U-Net/Pix2Pix class).
        
        #### 🗺️ Data Flow Diagram
        ```
        [ Input Image (RGB/Grayscale) ]
                      │
                      ▼
        [ Convert to CIELAB space ] ──────────────────────┐
                      │                                   │
                      ▼ (Extract L Channel)               │
        [ Normalize: (L - 50) / 100 ]                     │
                      │                                   │
                      ▼ (Resize to 256x256)               │
        [ Input Tensor [1, 1, 256, 256] ]                 │
                      │                                   │
                      ▼                                   │
        =======================================           │ (Keep original L size)
             GPU INFERENCE (ECCV 2016 Net)                │
        =======================================           │
                      │                                   │
                      ▼                                   │
        [ Output ab Tensor [1, 2, 64, 64] ]               │
                      │                                   │
                      ▼ (Bilinear Upsample)               │
        [ Predicted ab [1, 2, H_orig, W_orig] ]           │
                      │                                   │
                      ▼                                   ▼
        [ Concatenate L + ab ─────────────────────────────┘ ]
                      │
                      ▼
        [ Convert CIELAB back to RGB ]
                      │
                      ▼
        [ Output Colorized Image (JPEG Stream) ]
        ```
        
        #### 🧠 Model Details (ECCV 2016)
        1. **Encoder Blocks (1-3)**: Spatial downsampling via strided convolutions from 256x256 down to 32x32.
        2. **Dilated Convolution Blocks (4-7)**: Keeps spatial size at 32x32 but uses dilated convolutions to expand the network's receptive field.
        3. **Quantized Classification Head (Block 8)**: Predicts a probability distribution over 313 quantized $ab$ color bins.
        4. **Regression Decoding**: Computes the expected value from the distribution to yield smooth, natural-looking chrominance channels.
        """
    )
