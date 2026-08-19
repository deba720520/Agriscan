from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriScan — UAV Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CANVAS = "#131F19"
CANVAS2 = "#0E1712"
PAPER = "#EEF0E4"
PAPER_DIM = "#C9CDB9"
HEALTHY = "#2F6E44"
AMBER = "#E8A33D"
STRESS = "#C1440E"
LINE = "rgba(238,240,228,0.14)"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {CANVAS}; color: {PAPER}; }}
    section[data-testid="stSidebar"] {{ background-color: {CANVAS2}; border-right: 1px solid {LINE}; }}
    h1, h2, h3 {{ font-family: 'Space Grotesk','Segoe UI',sans-serif !important; letter-spacing:-0.01em; }}
    p, li, span, div {{ color: {PAPER}; }}
    .dim {{ color: {PAPER_DIM}; font-size: 0.85rem; }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}
    div[data-testid="stMetric"] {{
        background: {CANVAS2}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {PAPER_DIM} !important; }}
    .badge {{
        display:inline-flex; align-items:center; gap:6px; padding:4px 11px;
        border-radius:20px; border:1px solid rgba(238,240,228,0.28); font-size:0.8rem;
        font-family:'IBM Plex Mono',monospace;
    }}
    .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .card {{
        border:1px solid {LINE}; border-radius:6px; padding:20px 22px; background:{CANVAS2};
    }}
    .stButton>button {{
        background: {HEALTHY}; color: {PAPER}; border: 1px solid {HEALTHY};
        border-radius: 4px; font-family:'IBM Plex Mono',monospace;
    }}
    .stButton>button:hover {{ background:#1F4A30; border-color:#1F4A30; }}
    hr {{ border-color: {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, most recent first

# ----------------------------------------------------------------------------
# CORE ANALYSIS (real pixel math + real model classification)
# ----------------------------------------------------------------------------
def make_sample_field(seed=7, size=420):
    """Procedurally generate a synthetic aerial field image (RGB) so the demo
    always works even without an upload."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base_g = 150 + 25 * np.sin(xx / 18) * np.cos(yy / 22)
    base_r = 90 + 15 * np.sin(xx / 30 + 1)
    base_b = 60 + 10 * np.cos(yy / 25)

    # carve a few "stressed" patches (lower green/NIR-proxy response)
    patches = [(0.25, 0.3, 0.09), (0.7, 0.55, 0.07), (0.5, 0.8, 0.05)]
    for cx, cy, r in patches:
        dist = np.sqrt((xx / size - cx) ** 2 + (yy / size - cy) ** 2)
        mask = np.clip(1 - dist / r, 0, 1)
        base_g -= mask * 70
        base_r += mask * 40

    noise = rng.normal(0, 6, (size, size))
    r = np.clip(base_r + noise, 0, 255)
    g = np.clip(base_g + noise, 0, 255)
    b = np.clip(base_b + noise, 0, 255)
    arr = np.dstack([r, g, b]).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def compute_vegetation_index(img: Image.Image, max_dim=380):
    """Compute VARI (Visible Atmospherically Resistant Index) as an
    NDVI-style proxy from RGB channels: (G-R) / (G+R-B).
    This is a real, commonly used vegetation-health proxy when no
    near-infrared band is available."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    denom = g + r - b
    denom[np.abs(denom) < 1e-3] = 1e-3
    vari = (g - r) / denom
    vari = np.clip(vari, -1, 1)

    # normalize per-image via percentile clipping for contrast
    lo, hi = np.percentile(vari, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((vari - lo) / (hi - lo), 0, 1)
    return img, norm


def severity_grade(norm_index):
    """Relative severity grading: rank each pixel's vegetation index within
    the image and bucket the lowest-performing zones as more severe.
    This mirrors how real precision-ag zoning grades a field against its
    own baseline rather than an absolute global threshold."""
    p5, p15, p30 = np.percentile(norm_index, [5, 15, 30])
    severe = norm_index <= p5
    moderate = (norm_index > p5) & (norm_index <= p15)
    mild = (norm_index > p15) & (norm_index <= p30)
    healthy = norm_index > p30

    total = norm_index.size
    return {
        "severe": 100 * severe.sum() / total,
        "moderate": 100 * moderate.sum() / total,
        "mild": 100 * mild.sum() / total,
        "healthy": 100 * healthy.sum() / total,
        "severe_mask": severe,
        "moderate_mask": moderate,
        "mild_mask": mild,
    }


def heatmap_overlay(norm_index):
    cmap = matplotlib.colormaps["RdYlGn"]
    colored = cmap(norm_index)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored, "RGB")


MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
BASE_MODEL_ID = "google/mobilenet_v2_1.0_224"  # actively maintained repo, used only for image preprocessing


@st.cache_resource(show_spinner="Loading disease classification model (first run only, ~10MB)…")
def load_classifier():
    """Real pretrained model: MobileNetV2 fine-tuned on the PlantVillage
    dataset (38 classes across common crops).

    We load the image *processor* from the base google/mobilenet_v2_1.0_224
    repo (actively maintained, standard config) and the fine-tuned
    *weights + labels* from the PlantVillage checkpoint. The fine-tuned
    repo is from 2023 and its own preprocessor_config.json format isn't
    reliably auto-detected by newer transformers versions — using the
    base repo's processor sidesteps that while still running the real
    fine-tuned model for the actual prediction."""
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify_disease(img: Image.Image, affected_pct: float):
    """Runs real model inference. PlantVillage labels look like
    'Tomato___Late_blight' or 'Potato___healthy' — split crop from
    disease and clean up formatting for display.

    Falls back to a clearly-labeled placeholder if the model can't be
    loaded (e.g. a transient network issue reaching Hugging Face), so a
    hiccup there doesn't take down the whole app — the vegetation-index
    severity grading still works either way."""
    try:
        processor, model = load_classifier()
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top5_idx = torch.topk(probs, k=min(5, probs.shape[-1])).indices.tolist()
        preds = [{"label": model.config.id2label[i], "score": probs[i].item()} for i in top5_idx]

        top = preds[0]
        label = top["label"]
        confidence = round(top["score"] * 100, 1)

        parts = label.split("___") if "___" in label else [None, label]
        crop_guess = parts[0].replace("_", " ").strip() if parts[0] else None
        disease_raw = parts[1].replace("_", " ").strip()
        is_healthy = "healthy" in disease_raw.lower()
        disease = "Healthy" if is_healthy else disease_raw.title()

        return {
            "disease": disease,
            "confidence": confidence,
            "crop_guess": crop_guess,
            "top5": preds,
            "model_ok": True,
        }
    except Exception as e:
        return {
            "disease": "Model unavailable",
            "confidence": 0,
            "crop_guess": None,
            "top5": [],
            "model_ok": False,
            "error": str(e),
        }


def recommended_action(affected_pct, severe_pct, field_label):
    if affected_pct < 6:
        return "HEALTHY", HEALTHY, "No intervention needed. Continue routine monitoring on the next scheduled flight."
    if severe_pct < 2:
        return "MONITOR", AMBER, f"Stress detected but contained to mild zones. Re-scan {field_label} in 7 days before treating."
    return (
        "ACTION RECOMMENDED",
        STRESS,
        f"Apply targeted treatment to zones graded severe within 48 hours. Re-scan {field_label} in 5 days to confirm containment.",
    )


# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🌾 AgriScan")
st.sidebar.caption("UAV multispectral crop disease detection & severity estimation")
page = st.sidebar.radio("Navigate", ["Home", "Analyze", "Dashboard"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="dim mono">Scans this session: {len(st.session_state.history)}</span>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.markdown("## See the disease before the leaf shows it.")
    st.markdown(
        '<p class="dim">AgriScan reads a field image the way multispectral analysis reads UAV imagery — '
        "isolating vegetation stress from a vegetation index, grading severity zone by zone, and naming "
        "the likely disease — so treatment goes only where it's needed.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("01", "Capture", "UAV flight logs RGB / NIR / red-edge bands per plot, geotagged."),
        ("02", "Orthomosaic", "Frames are stitched into one continuous, corrected field map."),
        ("03", "Index extraction", "A vegetation index is computed per pixel from the bands."),
        ("04", "Classification", "Stressed zones are scored against known disease signatures."),
        ("05", "Severity grading", "Each zone is graded mild / moderate / severe and mapped."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(
                f'<div class="card" style="min-height:180px;">'
                f'<div class="mono dim">{num}</div>'
                f"<h4 style='margin-top:10px;'>{title}</h4>"
                f'<p class="dim" style="font-size:0.82rem; margin-top:8px;">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    st.markdown("#### Why this build is functional, not a mockup")
    st.markdown(
        """
- **Real pixel math** — the "Analyze" page computes an actual vegetation index (VARI:
  a standard RGB-only proxy for NDVI) from whatever image you upload, not random numbers.
- **Relative severity grading** — stress zones are ranked against that image's own
  distribution (bottom 5% = severe, next 10% = moderate, next 15% = mild), the same
  logic real precision-ag zoning uses within a single field.
- **Real disease classification** — the diagnosis comes from a pretrained MobileNetV2
  model fine-tuned on the PlantVillage dataset (38 classes across common crops),
  run live via Hugging Face Transformers. Its raw top-5 output is shown for transparency.
- **Session dashboard** — every scan you run is logged in this session with a real trend chart.

*Note: the classifier is trained on close-up leaf photos, so it works best on those —
a wide aerial field shot will still get graded for severity correctly, but the disease
label will be less reliable on that kind of image.*
        """
    )
    st.info("Go to **Analyze** in the sidebar to run a scan.", icon="🛰️")

# ----------------------------------------------------------------------------
# ANALYZE
# ----------------------------------------------------------------------------
elif page == "Analyze":
    st.markdown("## Run a scan")
    st.markdown(
        '<p class="dim">Upload a field or crop photo, or use the synthetic sample field. '
        "AgriScan computes a vegetation index from it live and grades severity.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        field_name = st.text_input("Field label", value="Sector 14 · Plot B")
        crop = st.selectbox("Crop", ["Wheat — HD-3086", "Maize — DKC-9108", "Soybean — JS-335", "Cotton — Bt-II"])
        uploaded = st.file_uploader("Upload field image", type=["jpg", "jpeg", "png"])
        use_sample = st.button("Use synthetic sample field instead")

        img = None
        if uploaded is not None:
            img = Image.open(uploaded)
        elif use_sample or "sample_seed" in st.session_state:
            seed = st.session_state.get("sample_seed", 7)
            if use_sample:
                seed = int(datetime.now().timestamp()) % 1000
                st.session_state["sample_seed"] = seed
            img = make_sample_field(seed=seed)

        run = st.button("Run scan →", type="primary", disabled=(img is None))

    with right:
        if img is None:
            st.markdown(
                '<div class="card" style="text-align:center; padding:60px 20px;">'
                '<p class="dim mono">NO SCAN LOADED<br><br>Upload an image or use the sample field, then run a scan.</p>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif not run and "last_result" not in st.session_state:
            st.image(img, caption="Loaded — click 'Run scan' to analyze", use_container_width=True)
        else:
            if run:
                with st.spinner("Reading bands → computing index → running disease model…"):
                    _, norm = compute_vegetation_index(img)
                    grades = severity_grade(norm)
                    heat = heatmap_overlay(norm)
                    affected_total = grades["severe"] + grades["moderate"] + grades["mild"]
                    clf_result = classify_disease(img, affected_total)
                    status, color, action = recommended_action(
                        affected_total, grades["severe"], field_name
                    )
                    result = {
                        "field": field_name, "crop": crop, "disease": clf_result["disease"],
                        "confidence": clf_result["confidence"], "crop_guess": clf_result["crop_guess"],
                        "top5": clf_result["top5"],
                        "affected": round(affected_total, 1),
                        "healthy": round(grades["healthy"], 1), "mild": round(grades["mild"], 1),
                        "moderate": round(grades["moderate"], 1), "severe": round(grades["severe"], 1),
                        "status": status, "color": color, "action": action,
                        "time": datetime.now(), "thumb": img.resize((64, 64)),
                    }
                    st.session_state["last_result"] = result
                    st.session_state["last_heat"] = heat
                    st.session_state["last_img"] = img
                    st.session_state.history.insert(0, result)

            result = st.session_state["last_result"]
            heat = st.session_state["last_heat"]
            src_img = st.session_state["last_img"]

            t1, t2 = st.tabs(["Severity heatmap", "Source image"])
            with t1:
                st.image(heat, use_container_width=True, caption="Green = healthy · red = high-stress zone (computed live)")
            with t2:
                st.image(src_img, use_container_width=True)

            if not result.get("model_ok", True):
                st.warning(
                    "⚠️ The disease classification model couldn't be reached this run "
                    "(likely a temporary network hiccup fetching it from Hugging Face). "
                    "Severity grading below is still real, computed from your image's pixels — "
                    "just the disease *label* is unavailable this time. Try running the scan again."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Affected area (from index)", f"{result['affected']}%")
            m2.metric(
                "Diagnosis (model)",
                result["disease"],
                f"{result['confidence']}% conf." if result["disease"] not in ("Healthy", "Model unavailable") else None,
            )
            m3.markdown(
                f'<div style="padding-top:8px;"><span class="badge">'
                f'<span class="dot" style="background:{result["color"]}"></span>{result["status"]}</span></div>',
                unsafe_allow_html=True,
            )

            if result.get("crop_guess") and result["crop_guess"].lower() not in crop.lower():
                st.caption(f"ℹ️ Model's underlying crop guess: **{result['crop_guess']}** — for best accuracy, upload a close-up leaf photo matching the selected crop.")

            if result.get("top5"):
                with st.expander("Model's top-5 predictions (raw output)"):
                    top5_df = pd.DataFrame(
                        [{"label": p["label"], "confidence %": round(p["score"] * 100, 1)} for p in result["top5"]]
                    )
                    st.dataframe(top5_df, use_container_width=True, hide_index=True)
                    st.caption(f"Model: `{MODEL_ID}` — MobileNetV2 fine-tuned on PlantVillage (38 classes). Real inference, not simulated.")

            st.write("")
            st.markdown("**Severity distribution**")
            dist_df = pd.DataFrame(
                {"share (%)": [result["healthy"], result["mild"], result["moderate"], result["severe"]]},
                index=["Healthy", "Mild", "Moderate", "Severe"],
            )
            st.bar_chart(dist_df, color=HEALTHY, height=200)

            st.markdown(f'<div class="card"><b>Recommended action</b><br><span class="dim">{result["action"]}</span></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
elif page == "Dashboard":
    st.markdown("## Field history")
    st.markdown('<p class="dim">Every scan run this session, logged with a severity trend.</p>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="card" style="text-align:center; padding:50px 20px;">'
            '<p class="dim mono">NO SCANS YET — run one from the Analyze page.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scans", len(df))
        c2.metric("Fields tracked", df["field"].nunique())
        c3.metric("Avg. affected area", f"{df['affected'].mean():.1f}%")
        c4.metric("Flagged severe", int((df["severe"] >= 2).sum()))

        st.write("")
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown("**Scan log**")
            show_df = df[["field", "crop", "disease", "affected", "status", "time"]].copy()
            show_df["time"] = show_df["time"].apply(lambda t: t.strftime("%H:%M:%S"))
            show_df.columns = ["Field", "Crop", "Diagnosis", "Affected %", "Status", "Time"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown("**Affected area trend (oldest → newest)**")
            trend = df.iloc[::-1][["affected"]].reset_index(drop=True)
            trend.columns = ["Affected %"]
            st.line_chart(trend, color=AMBER, height=250)

        st.write("")
        if st.button("Clear session history"):
            st.session_state.history = []
            for k in ["last_result", "last_heat", "last_img"]:
                st.session_state.pop(k, None)
            st.rerun()    h1, h2, h3 {{ font-family: 'Space Grotesk','Segoe UI',sans-serif !important; letter-spacing:-0.01em; }}
    p, li, span, div {{ color: {PAPER}; }}
    .dim {{ color: {PAPER_DIM}; font-size: 0.85rem; }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}
    div[data-testid="stMetric"] {{
        background: {CANVAS2}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {PAPER_DIM} !important; }}
    .badge {{
        display:inline-flex; align-items:center; gap:6px; padding:4px 11px;
        border-radius:20px; border:1px solid rgba(238,240,228,0.28); font-size:0.8rem;
        font-family:'IBM Plex Mono',monospace;
    }}
    .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .card {{
        border:1px solid {LINE}; border-radius:6px; padding:20px 22px; background:{CANVAS2};
    }}
    .stButton>button {{
        background: {HEALTHY}; color: {PAPER}; border: 1px solid {HEALTHY};
        border-radius: 4px; font-family:'IBM Plex Mono',monospace;
    }}
    .stButton>button:hover {{ background:#1F4A30; border-color:#1F4A30; }}
    hr {{ border-color: {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, most recent first

# ----------------------------------------------------------------------------
# CORE ANALYSIS (real pixel math + real model classification)
# ----------------------------------------------------------------------------
def make_sample_field(seed=7, size=420):
    """Procedurally generate a synthetic aerial field image (RGB) so the demo
    always works even without an upload."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base_g = 150 + 25 * np.sin(xx / 18) * np.cos(yy / 22)
    base_r = 90 + 15 * np.sin(xx / 30 + 1)
    base_b = 60 + 10 * np.cos(yy / 25)

    # carve a few "stressed" patches (lower green/NIR-proxy response)
    patches = [(0.25, 0.3, 0.09), (0.7, 0.55, 0.07), (0.5, 0.8, 0.05)]
    for cx, cy, r in patches:
        dist = np.sqrt((xx / size - cx) ** 2 + (yy / size - cy) ** 2)
        mask = np.clip(1 - dist / r, 0, 1)
        base_g -= mask * 70
        base_r += mask * 40

    noise = rng.normal(0, 6, (size, size))
    r = np.clip(base_r + noise, 0, 255)
    g = np.clip(base_g + noise, 0, 255)
    b = np.clip(base_b + noise, 0, 255)
    arr = np.dstack([r, g, b]).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def compute_vegetation_index(img: Image.Image, max_dim=380):
    """Compute VARI (Visible Atmospherically Resistant Index) as an
    NDVI-style proxy from RGB channels: (G-R) / (G+R-B).
    This is a real, commonly used vegetation-health proxy when no
    near-infrared band is available."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    denom = g + r - b
    denom[np.abs(denom) < 1e-3] = 1e-3
    vari = (g - r) / denom
    vari = np.clip(vari, -1, 1)

    # normalize per-image via percentile clipping for contrast
    lo, hi = np.percentile(vari, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((vari - lo) / (hi - lo), 0, 1)
    return img, norm


def severity_grade(norm_index):
    """Relative severity grading: rank each pixel's vegetation index within
    the image and bucket the lowest-performing zones as more severe.
    This mirrors how real precision-ag zoning grades a field against its
    own baseline rather than an absolute global threshold."""
    p5, p15, p30 = np.percentile(norm_index, [5, 15, 30])
    severe = norm_index <= p5
    moderate = (norm_index > p5) & (norm_index <= p15)
    mild = (norm_index > p15) & (norm_index <= p30)
    healthy = norm_index > p30

    total = norm_index.size
    return {
        "severe": 100 * severe.sum() / total,
        "moderate": 100 * moderate.sum() / total,
        "mild": 100 * mild.sum() / total,
        "healthy": 100 * healthy.sum() / total,
        "severe_mask": severe,
        "moderate_mask": moderate,
        "mild_mask": mild,
    }


def heatmap_overlay(norm_index):
    cmap = matplotlib.colormaps["RdYlGn"]
    colored = cmap(norm_index)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored, "RGB")


MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
BASE_MODEL_ID = "google/mobilenet_v2_1.0_224"  # actively maintained repo, used only for image preprocessing


@st.cache_resource(show_spinner="Loading disease classification model (first run only, ~10MB)…")
def load_classifier():
    """Real pretrained model: MobileNetV2 fine-tuned on the PlantVillage
    dataset (38 classes across common crops).

    We load the image *processor* from the base google/mobilenet_v2_1.0_224
    repo (actively maintained, standard config) and the fine-tuned
    *weights + labels* from the PlantVillage checkpoint. The fine-tuned
    repo is from 2023 and its own preprocessor_config.json format isn't
    reliably auto-detected by newer transformers versions — using the
    base repo's processor sidesteps that while still running the real
    fine-tuned model for the actual prediction."""
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify_disease(img: Image.Image, affected_pct: float):
    """Runs real model inference. PlantVillage labels look like
    'Tomato___Late_blight' or 'Potato___healthy' — split crop from
    disease and clean up formatting for display."""
    processor, model = load_classifier()
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top5_idx = torch.topk(probs, k=min(5, probs.shape[-1])).indices.tolist()
    preds = [{"label": model.config.id2label[i], "score": probs[i].item()} for i in top5_idx]

    top = preds[0]
    label = top["label"]
    confidence = round(top["score"] * 100, 1)

    parts = label.split("___") if "___" in label else [None, label]
    crop_guess = parts[0].replace("_", " ").strip() if parts[0] else None
    disease_raw = parts[1].replace("_", " ").strip()
    is_healthy = "healthy" in disease_raw.lower()
    disease = "Healthy" if is_healthy else disease_raw.title()

    return {
        "disease": disease,
        "confidence": confidence,
        "crop_guess": crop_guess,
        "top5": preds,
    }


def recommended_action(affected_pct, severe_pct, field_label):
    if affected_pct < 6:
        return "HEALTHY", HEALTHY, "No intervention needed. Continue routine monitoring on the next scheduled flight."
    if severe_pct < 2:
        return "MONITOR", AMBER, f"Stress detected but contained to mild zones. Re-scan {field_label} in 7 days before treating."
    return (
        "ACTION RECOMMENDED",
        STRESS,
        f"Apply targeted treatment to zones graded severe within 48 hours. Re-scan {field_label} in 5 days to confirm containment.",
    )


# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🌾 AgriScan")
st.sidebar.caption("UAV multispectral crop disease detection & severity estimation")
page = st.sidebar.radio("Navigate", ["Home", "Analyze", "Dashboard"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="dim mono">Scans this session: {len(st.session_state.history)}</span>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.markdown("## See the disease before the leaf shows it.")
    st.markdown(
        '<p class="dim">AgriScan reads a field image the way multispectral analysis reads UAV imagery — '
        "isolating vegetation stress from a vegetation index, grading severity zone by zone, and naming "
        "the likely disease — so treatment goes only where it's needed.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("01", "Capture", "UAV flight logs RGB / NIR / red-edge bands per plot, geotagged."),
        ("02", "Orthomosaic", "Frames are stitched into one continuous, corrected field map."),
        ("03", "Index extraction", "A vegetation index is computed per pixel from the bands."),
        ("04", "Classification", "Stressed zones are scored against known disease signatures."),
        ("05", "Severity grading", "Each zone is graded mild / moderate / severe and mapped."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(
                f'<div class="card" style="min-height:180px;">'
                f'<div class="mono dim">{num}</div>'
                f"<h4 style='margin-top:10px;'>{title}</h4>"
                f'<p class="dim" style="font-size:0.82rem; margin-top:8px;">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    st.markdown("#### Why this build is functional, not a mockup")
    st.markdown(
        """
- **Real pixel math** — the "Analyze" page computes an actual vegetation index (VARI:
  a standard RGB-only proxy for NDVI) from whatever image you upload, not random numbers.
- **Relative severity grading** — stress zones are ranked against that image's own
  distribution (bottom 5% = severe, next 10% = moderate, next 15% = mild), the same
  logic real precision-ag zoning uses within a single field.
- **Real disease classification** — the diagnosis comes from a pretrained MobileNetV2
  model fine-tuned on the PlantVillage dataset (38 classes across common crops),
  run live via Hugging Face Transformers. Its raw top-5 output is shown for transparency.
- **Session dashboard** — every scan you run is logged in this session with a real trend chart.

*Note: the classifier is trained on close-up leaf photos, so it works best on those —
a wide aerial field shot will still get graded for severity correctly, but the disease
label will be less reliable on that kind of image.*
        """
    )
    st.info("Go to **Analyze** in the sidebar to run a scan.", icon="🛰️")

# ----------------------------------------------------------------------------
# ANALYZE
# ----------------------------------------------------------------------------
elif page == "Analyze":
    st.markdown("## Run a scan")
    st.markdown(
        '<p class="dim">Upload a field or crop photo, or use the synthetic sample field. '
        "AgriScan computes a vegetation index from it live and grades severity.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        field_name = st.text_input("Field label", value="Sector 14 · Plot B")
        crop = st.selectbox("Crop", ["Wheat — HD-3086", "Maize — DKC-9108", "Soybean — JS-335", "Cotton — Bt-II"])
        uploaded = st.file_uploader("Upload field image", type=["jpg", "jpeg", "png"])
        use_sample = st.button("Use synthetic sample field instead")

        img = None
        if uploaded is not None:
            img = Image.open(uploaded)
            img_bytes = uploaded.getvalue()
        elif use_sample or "sample_seed" in st.session_state:
            seed = st.session_state.get("sample_seed", 7)
            if use_sample:
                seed = int(datetime.now().timestamp()) % 1000
                st.session_state["sample_seed"] = seed
            img = make_sample_field(seed=seed)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

        run = st.button("Run scan →", type="primary", disabled=(img is None))

    with right:
        if img is None:
            st.markdown(
                '<div class="card" style="text-align:center; padding:60px 20px;">'
                '<p class="dim mono">NO SCAN LOADED<br><br>Upload an image or use the sample field, then run a scan.</p>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif not run and "last_result" not in st.session_state:
            st.image(img, caption="Loaded — click 'Run scan' to analyze", use_container_width=True)
        else:
            if run:
                with st.spinner("Reading bands → computing index → running disease model…"):
                    _, norm = compute_vegetation_index(img)
                    grades = severity_grade(norm)
                    heat = heatmap_overlay(norm)
                    affected_total = grades["severe"] + grades["moderate"] + grades["mild"]
                    clf_result = classify_disease(img, affected_total)
                    status, color, action = recommended_action(
                        affected_total, grades["severe"], field_name
                    )
                    result = {
                        "field": field_name, "crop": crop, "disease": clf_result["disease"],
                        "confidence": clf_result["confidence"], "crop_guess": clf_result["crop_guess"],
                        "top5": clf_result["top5"],
                        "affected": round(affected_total, 1),
                        "healthy": round(grades["healthy"], 1), "mild": round(grades["mild"], 1),
                        "moderate": round(grades["moderate"], 1), "severe": round(grades["severe"], 1),
                        "status": status, "color": color, "action": action,
                        "time": datetime.now(), "thumb": img.resize((64, 64)),
                    }
                    st.session_state["last_result"] = result
                    st.session_state["last_heat"] = heat
                    st.session_state["last_img"] = img
                    st.session_state.history.insert(0, result)

            result = st.session_state["last_result"]
            heat = st.session_state["last_heat"]
            src_img = st.session_state["last_img"]

            t1, t2 = st.tabs(["Severity heatmap", "Source image"])
            with t1:
                st.image(heat, use_container_width=True, caption="Green = healthy · red = high-stress zone (computed live)")
            with t2:
                st.image(src_img, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Affected area (from index)", f"{result['affected']}%")
            m2.metric(
                "Diagnosis (model)",
                result["disease"],
                f"{result['confidence']}% conf." if result["disease"] != "Healthy" else None,
            )
            m3.markdown(
                f'<div style="padding-top:8px;"><span class="badge">'
                f'<span class="dot" style="background:{result["color"]}"></span>{result["status"]}</span></div>',
                unsafe_allow_html=True,
            )

            if result.get("crop_guess") and result["crop_guess"].lower() not in crop.lower():
                st.caption(f"ℹ️ Model's underlying crop guess: **{result['crop_guess']}** — for best accuracy, upload a close-up leaf photo matching the selected crop.")

            with st.expander("Model's top-5 predictions (raw output)"):
                top5_df = pd.DataFrame(
                    [{"label": p["label"], "confidence %": round(p["score"] * 100, 1)} for p in result["top5"]]
                )
                st.dataframe(top5_df, use_container_width=True, hide_index=True)
                st.caption(f"Model: `{MODEL_ID}` — MobileNetV2 fine-tuned on PlantVillage (38 classes). Real inference, not simulated.")

            st.write("")
            st.markdown("**Severity distribution**")
            dist_df = pd.DataFrame(
                {"share (%)": [result["healthy"], result["mild"], result["moderate"], result["severe"]]},
                index=["Healthy", "Mild", "Moderate", "Severe"],
            )
            st.bar_chart(dist_df, color=HEALTHY, height=200)

            st.markdown(f'<div class="card"><b>Recommended action</b><br><span class="dim">{result["action"]}</span></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
elif page == "Dashboard":
    st.markdown("## Field history")
    st.markdown('<p class="dim">Every scan run this session, logged with a severity trend.</p>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="card" style="text-align:center; padding:50px 20px;">'
            '<p class="dim mono">NO SCANS YET — run one from the Analyze page.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scans", len(df))
        c2.metric("Fields tracked", df["field"].nunique())
        c3.metric("Avg. affected area", f"{df['affected'].mean():.1f}%")
        c4.metric("Flagged severe", int((df["severe"] >= 2).sum()))

        st.write("")
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown("**Scan log**")
            show_df = df[["field", "crop", "disease", "affected", "status", "time"]].copy()
            show_df["time"] = show_df["time"].apply(lambda t: t.strftime("%H:%M:%S"))
            show_df.columns = ["Field", "Crop", "Diagnosis", "Affected %", "Status", "Time"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown("**Affected area trend (oldest → newest)**")
            trend = df.iloc[::-1][["affected"]].reset_index(drop=True)
            trend.columns = ["Affected %"]
            st.line_chart(trend, color=AMBER, height=250)

        st.write("")
        if st.button("Clear session history"):
            st.session_state.history = []
            for k in ["last_result", "last_heat", "last_img"]:
                st.session_state.pop(k, None)
            st.rerun()    p, li, span, div {{ color: {PAPER}; }}
    .dim {{ color: {PAPER_DIM}; font-size: 0.85rem; }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}
    div[data-testid="stMetric"] {{
        background: {CANVAS2}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {PAPER_DIM} !important; }}
    .badge {{
        display:inline-flex; align-items:center; gap:6px; padding:4px 11px;
        border-radius:20px; border:1px solid rgba(238,240,228,0.28); font-size:0.8rem;
        font-family:'IBM Plex Mono',monospace;
    }}
    .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .card {{
        border:1px solid {LINE}; border-radius:6px; padding:20px 22px; background:{CANVAS2};
    }}
    .stButton>button {{
        background: {HEALTHY}; color: {PAPER}; border: 1px solid {HEALTHY};
        border-radius: 4px; font-family:'IBM Plex Mono',monospace;
    }}
    .stButton>button:hover {{ background:#1F4A30; border-color:#1F4A30; }}
    hr {{ border-color: {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, most recent first

# ----------------------------------------------------------------------------
# CORE ANALYSIS (real pixel math + real model classification)
# ----------------------------------------------------------------------------
def make_sample_field(seed=7, size=420):
    """Procedurally generate a synthetic aerial field image (RGB) so the demo
    always works even without an upload."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base_g = 150 + 25 * np.sin(xx / 18) * np.cos(yy / 22)
    base_r = 90 + 15 * np.sin(xx / 30 + 1)
    base_b = 60 + 10 * np.cos(yy / 25)

    # carve a few "stressed" patches (lower green/NIR-proxy response)
    patches = [(0.25, 0.3, 0.09), (0.7, 0.55, 0.07), (0.5, 0.8, 0.05)]
    for cx, cy, r in patches:
        dist = np.sqrt((xx / size - cx) ** 2 + (yy / size - cy) ** 2)
        mask = np.clip(1 - dist / r, 0, 1)
        base_g -= mask * 70
        base_r += mask * 40

    noise = rng.normal(0, 6, (size, size))
    r = np.clip(base_r + noise, 0, 255)
    g = np.clip(base_g + noise, 0, 255)
    b = np.clip(base_b + noise, 0, 255)
    arr = np.dstack([r, g, b]).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def compute_vegetation_index(img: Image.Image, max_dim=380):
    """Compute VARI (Visible Atmospherically Resistant Index) as an
    NDVI-style proxy from RGB channels: (G-R) / (G+R-B).
    This is a real, commonly used vegetation-health proxy when no
    near-infrared band is available."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    denom = g + r - b
    denom[np.abs(denom) < 1e-3] = 1e-3
    vari = (g - r) / denom
    vari = np.clip(vari, -1, 1)

    # normalize per-image via percentile clipping for contrast
    lo, hi = np.percentile(vari, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((vari - lo) / (hi - lo), 0, 1)
    return img, norm


def severity_grade(norm_index):
    """Relative severity grading: rank each pixel's vegetation index within
    the image and bucket the lowest-performing zones as more severe.
    This mirrors how real precision-ag zoning grades a field against its
    own baseline rather than an absolute global threshold."""
    p5, p15, p30 = np.percentile(norm_index, [5, 15, 30])
    severe = norm_index <= p5
    moderate = (norm_index > p5) & (norm_index <= p15)
    mild = (norm_index > p15) & (norm_index <= p30)
    healthy = norm_index > p30

    total = norm_index.size
    return {
        "severe": 100 * severe.sum() / total,
        "moderate": 100 * moderate.sum() / total,
        "mild": 100 * mild.sum() / total,
        "healthy": 100 * healthy.sum() / total,
        "severe_mask": severe,
        "moderate_mask": moderate,
        "mild_mask": mild,
    }


def heatmap_overlay(norm_index):
    cmap = matplotlib.colormaps["RdYlGn"]
    colored = cmap(norm_index)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored, "RGB")


MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


@st.cache_resource(show_spinner="Loading disease classification model (first run only, ~10MB)…")
def load_classifier():
    """Real pretrained model: MobileNetV2 fine-tuned on the PlantVillage
    dataset (38 classes across common crops). Downloaded once from
    Hugging Face and cached for the life of the app process."""
    return hf_pipeline("image-classification", model=MODEL_ID)


def classify_disease(img: Image.Image, affected_pct: float):
    """Runs real model inference. PlantVillage labels look like
    'Tomato___Late_blight' or 'Potato___healthy' — split crop from
    disease and clean up formatting for display."""
    clf = load_classifier()
    preds = clf(img.convert("RGB"))
    top = preds[0]
    label = top["label"]
    confidence = round(top["score"] * 100, 1)

    parts = label.split("___") if "___" in label else [None, label]
    crop_guess = parts[0].replace("_", " ").strip() if parts[0] else None
    disease_raw = parts[1].replace("_", " ").strip()
    is_healthy = "healthy" in disease_raw.lower()
    disease = "Healthy" if is_healthy else disease_raw.title()

    return {
        "disease": disease,
        "confidence": confidence,
        "crop_guess": crop_guess,
        "top5": preds[:5],
    }


def recommended_action(affected_pct, severe_pct, field_label):
    if affected_pct < 6:
        return "HEALTHY", HEALTHY, "No intervention needed. Continue routine monitoring on the next scheduled flight."
    if severe_pct < 2:
        return "MONITOR", AMBER, f"Stress detected but contained to mild zones. Re-scan {field_label} in 7 days before treating."
    return (
        "ACTION RECOMMENDED",
        STRESS,
        f"Apply targeted treatment to zones graded severe within 48 hours. Re-scan {field_label} in 5 days to confirm containment.",
    )


# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🌾 AgriScan")
st.sidebar.caption("UAV multispectral crop disease detection & severity estimation")
page = st.sidebar.radio("Navigate", ["Home", "Analyze", "Dashboard"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="dim mono">Scans this session: {len(st.session_state.history)}</span>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.markdown("## See the disease before the leaf shows it.")
    st.markdown(
        '<p class="dim">AgriScan reads a field image the way multispectral analysis reads UAV imagery — '
        "isolating vegetation stress from a vegetation index, grading severity zone by zone, and naming "
        "the likely disease — so treatment goes only where it's needed.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("01", "Capture", "UAV flight logs RGB / NIR / red-edge bands per plot, geotagged."),
        ("02", "Orthomosaic", "Frames are stitched into one continuous, corrected field map."),
        ("03", "Index extraction", "A vegetation index is computed per pixel from the bands."),
        ("04", "Classification", "Stressed zones are scored against known disease signatures."),
        ("05", "Severity grading", "Each zone is graded mild / moderate / severe and mapped."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(
                f'<div class="card" style="min-height:180px;">'
                f'<div class="mono dim">{num}</div>'
                f"<h4 style='margin-top:10px;'>{title}</h4>"
                f'<p class="dim" style="font-size:0.82rem; margin-top:8px;">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    st.markdown("#### Why this build is functional, not a mockup")
    st.markdown(
        """
- **Real pixel math** — the "Analyze" page computes an actual vegetation index (VARI:
  a standard RGB-only proxy for NDVI) from whatever image you upload, not random numbers.
- **Relative severity grading** — stress zones are ranked against that image's own
  distribution (bottom 5% = severe, next 10% = moderate, next 15% = mild), the same
  logic real precision-ag zoning uses within a single field.
- **Real disease classification** — the diagnosis comes from a pretrained MobileNetV2
  model fine-tuned on the PlantVillage dataset (38 classes across common crops),
  run live via Hugging Face Transformers. Its raw top-5 output is shown for transparency.
- **Session dashboard** — every scan you run is logged in this session with a real trend chart.

*Note: the classifier is trained on close-up leaf photos, so it works best on those —
a wide aerial field shot will still get graded for severity correctly, but the disease
label will be less reliable on that kind of image.*
        """
    )
    st.info("Go to **Analyze** in the sidebar to run a scan.", icon="🛰️")

# ----------------------------------------------------------------------------
# ANALYZE
# ----------------------------------------------------------------------------
elif page == "Analyze":
    st.markdown("## Run a scan")
    st.markdown(
        '<p class="dim">Upload a field or crop photo, or use the synthetic sample field. '
        "AgriScan computes a vegetation index from it live and grades severity.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        field_name = st.text_input("Field label", value="Sector 14 · Plot B")
        crop = st.selectbox("Crop", ["Wheat — HD-3086", "Maize — DKC-9108", "Soybean — JS-335", "Cotton — Bt-II"])
        uploaded = st.file_uploader("Upload field image", type=["jpg", "jpeg", "png"])
        use_sample = st.button("Use synthetic sample field instead")

        img = None
        if uploaded is not None:
            img = Image.open(uploaded)
            img_bytes = uploaded.getvalue()
        elif use_sample or "sample_seed" in st.session_state:
            seed = st.session_state.get("sample_seed", 7)
            if use_sample:
                seed = int(datetime.now().timestamp()) % 1000
                st.session_state["sample_seed"] = seed
            img = make_sample_field(seed=seed)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

        run = st.button("Run scan →", type="primary", disabled=(img is None))

    with right:
        if img is None:
            st.markdown(
                '<div class="card" style="text-align:center; padding:60px 20px;">'
                '<p class="dim mono">NO SCAN LOADED<br><br>Upload an image or use the sample field, then run a scan.</p>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif not run and "last_result" not in st.session_state:
            st.image(img, caption="Loaded — click 'Run scan' to analyze", use_container_width=True)
        else:
            if run:
                with st.spinner("Reading bands → computing index → running disease model…"):
                    _, norm = compute_vegetation_index(img)
                    grades = severity_grade(norm)
                    heat = heatmap_overlay(norm)
                    affected_total = grades["severe"] + grades["moderate"] + grades["mild"]
                    clf_result = classify_disease(img, affected_total)
                    status, color, action = recommended_action(
                        affected_total, grades["severe"], field_name
                    )
                    result = {
                        "field": field_name, "crop": crop, "disease": clf_result["disease"],
                        "confidence": clf_result["confidence"], "crop_guess": clf_result["crop_guess"],
                        "top5": clf_result["top5"],
                        "affected": round(affected_total, 1),
                        "healthy": round(grades["healthy"], 1), "mild": round(grades["mild"], 1),
                        "moderate": round(grades["moderate"], 1), "severe": round(grades["severe"], 1),
                        "status": status, "color": color, "action": action,
                        "time": datetime.now(), "thumb": img.resize((64, 64)),
                    }
                    st.session_state["last_result"] = result
                    st.session_state["last_heat"] = heat
                    st.session_state["last_img"] = img
                    st.session_state.history.insert(0, result)

            result = st.session_state["last_result"]
            heat = st.session_state["last_heat"]
            src_img = st.session_state["last_img"]

            t1, t2 = st.tabs(["Severity heatmap", "Source image"])
            with t1:
                st.image(heat, use_container_width=True, caption="Green = healthy · red = high-stress zone (computed live)")
            with t2:
                st.image(src_img, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Affected area (from index)", f"{result['affected']}%")
            m2.metric(
                "Diagnosis (model)",
                result["disease"],
                f"{result['confidence']}% conf." if result["disease"] != "Healthy" else None,
            )
            m3.markdown(
                f'<div style="padding-top:8px;"><span class="badge">'
                f'<span class="dot" style="background:{result["color"]}"></span>{result["status"]}</span></div>',
                unsafe_allow_html=True,
            )

            if result.get("crop_guess") and result["crop_guess"].lower() not in crop.lower():
                st.caption(f"ℹ️ Model's underlying crop guess: **{result['crop_guess']}** — for best accuracy, upload a close-up leaf photo matching the selected crop.")

            with st.expander("Model's top-5 predictions (raw output)"):
                top5_df = pd.DataFrame(
                    [{"label": p["label"], "confidence %": round(p["score"] * 100, 1)} for p in result["top5"]]
                )
                st.dataframe(top5_df, use_container_width=True, hide_index=True)
                st.caption(f"Model: `{MODEL_ID}` — MobileNetV2 fine-tuned on PlantVillage (38 classes). Real inference, not simulated.")

            st.write("")
            st.markdown("**Severity distribution**")
            dist_df = pd.DataFrame(
                {"share (%)": [result["healthy"], result["mild"], result["moderate"], result["severe"]]},
                index=["Healthy", "Mild", "Moderate", "Severe"],
            )
            st.bar_chart(dist_df, color=HEALTHY, height=200)

            st.markdown(f'<div class="card"><b>Recommended action</b><br><span class="dim">{result["action"]}</span></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
elif page == "Dashboard":
    st.markdown("## Field history")
    st.markdown('<p class="dim">Every scan run this session, logged with a severity trend.</p>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="card" style="text-align:center; padding:50px 20px;">'
            '<p class="dim mono">NO SCANS YET — run one from the Analyze page.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scans", len(df))
        c2.metric("Fields tracked", df["field"].nunique())
        c3.metric("Avg. affected area", f"{df['affected'].mean():.1f}%")
        c4.metric("Flagged severe", int((df["severe"] >= 2).sum()))

        st.write("")
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown("**Scan log**")
            show_df = df[["field", "crop", "disease", "affected", "status", "time"]].copy()
            show_df["time"] = show_df["time"].apply(lambda t: t.strftime("%H:%M:%S"))
            show_df.columns = ["Field", "Crop", "Diagnosis", "Affected %", "Status", "Time"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown("**Affected area trend (oldest → newest)**")
            trend = df.iloc[::-1][["affected"]].reset_index(drop=True)
            trend.columns = ["Affected %"]
            st.line_chart(trend, color=AMBER, height=250)

        st.write("")
        if st.button("Clear session history"):
            st.session_state.history = []
            for k in ["last_result", "last_heat", "last_img"]:
                st.session_state.pop(k, None)
            st.rerun()
