import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import plotly.graph_objects as go
import os
import io

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PaveDetect — Deteksi Kerusakan Jalan",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; }

.hero {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    border: 1px solid rgba(255,140,0,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,140,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #ff8c00; margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.55);
    margin: 0; font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,140,0,0.15);
    border: 1px solid rgba(255,140,0,0.4);
    color: #ff8c00;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 0.8rem; letter-spacing: 1px;
}
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: #0f1923;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    line-height: 1; margin-bottom: 4px;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase; letter-spacing: 0.8px;
}
.metric-total .val  { color: #ff8c00; }
.metric-classes .val { color: #4fc3f7; }
.metric-conf .val   { color: #81c784; }

.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 500; }
.badge-D00 { background: rgba(55,138,221,0.2); color: #64b5f6; border: 1px solid rgba(55,138,221,0.4); }
.badge-D10 { background: rgba(29,158,117,0.2); color: #4db6ac; border: 1px solid rgba(29,158,117,0.4); }
.badge-D20 { background: rgba(186,117,23,0.2); color: #ffb74d; border: 1px solid rgba(186,117,23,0.4); }
.badge-D40 { background: rgba(216,90,48,0.2);  color: #ef9a9a; border: 1px solid rgba(216,90,48,0.4); }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem; letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.stButton > button {
    background: rgba(255,140,0,0.08) !important;
    border: 1px solid rgba(255,140,0,0.3) !important;
    color: rgba(255,255,255,0.8) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(255,140,0,0.18) !important;
    border-color: #ff8c00 !important;
    color: #ff8c00 !important;
}
[data-testid="stSidebar"] {
    background: #0d1821;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.85rem; color: rgba(255,255,255,0.6);
}
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1rem 0 !important; }
.info-box {
    background: rgba(79,195,247,0.07);
    border-left: 3px solid #4fc3f7;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem; color: rgba(255,255,255,0.7);
    margin-bottom: 1rem;
}
.warn-box {
    background: rgba(255,183,77,0.07);
    border-left: 3px solid #ffb74d;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem; color: rgba(255,255,255,0.7);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS - 4 kelas
# ──────────────────────────────────────────────
CLASS_INFO = {
    "D00": {
        "name"   : "Longitudinal Crack",
        "id_name": "Retak Memanjang",
        "color"  : "#378ADD",
        "badge"  : "badge-D00",
        "desc"   : "Retakan sejajar arah perjalanan kendaraan",
        "risk"   : "🟡 Sedang",
    },
    "D10": {
        "name"   : "Transverse Crack",
        "id_name": "Retak Melintang",
        "color"  : "#1D9E75",
        "badge"  : "badge-D10",
        "desc"   : "Retakan tegak lurus terhadap arah perjalanan",
        "risk"   : "🟡 Sedang",
    },
    "D20": {
        "name"   : "Alligator Crack",
        "id_name": "Retak Buaya",
        "color"  : "#BA7517",
        "badge"  : "badge-D20",
        "desc"   : "Pola retakan menyerupai kulit buaya, indikasi kelelahan struktur",
        "risk"   : "🔴 Tinggi",
    },
    "D40": {
        "name"   : "Pothole",
        "id_name": "Lubang Jalan",
        "color"  : "#D85A30",
        "badge"  : "badge-D40",
        "desc"   : "Lubang pada permukaan jalan akibat erosi material",
        "risk"   : "🔴 Tinggi",
    },
}

MODEL_PATH = "best.pt"

SAMPLE_IMAGES = [
    ("Contoh 1 — Retak Memanjang & Buaya",  "samples/sample_1.jpg"),
    ("Contoh 2 — Lubang Jalan",              "samples/sample_2.jpg"),
    ("Contoh 3 — Multi Kerusakan",           "samples/sample_3.jpg"),
]

# ──────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(
            f"❌ File model tidak ditemukan: `{path}`\n\n"
            "Pastikan file `best.pt` ada di direktori yang sama dengan `app.py`."
        )
        st.stop()
    return YOLO(path)

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def run_inference(model, image: np.ndarray, conf: float, iou: float):
    return model.predict(source=image, conf=conf, iou=iou,
                         imgsz=640, verbose=False)[0]

def parse_results(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame()
    records = []
    for box in boxes:
        cls_id   = int(box.cls[0])
        cls_code = result.names[cls_id]
        conf_val = float(box.conf[0])
        xyxy     = box.xyxy[0].cpu().numpy()
        records.append({
            "Kode"           : cls_code,
            "Jenis Kerusakan": CLASS_INFO.get(cls_code, {}).get("id_name", cls_code),
            "Confidence"     : round(conf_val * 100, 1),
            "Risiko"         : CLASS_INFO.get(cls_code, {}).get("risk", "-"),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────
def make_bar_chart(df: pd.DataFrame):
    counts = df["Kode"].value_counts().reset_index()
    counts.columns = ["Kode", "Jumlah"]
    counts["Warna"] = counts["Kode"].map(lambda k: CLASS_INFO.get(k, {}).get("color", "#888"))
    counts["Label"] = counts["Kode"].map(lambda k: CLASS_INFO.get(k, {}).get("id_name", k))

    fig = go.Figure(go.Bar(
        x=counts["Label"], y=counts["Jumlah"],
        marker_color=counts["Warna"],
        text=counts["Jumlah"], textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        height=220, showlegend=False,
    )
    return fig

def make_donut_chart(df: pd.DataFrame):
    counts = df["Kode"].value_counts().reset_index()
    counts.columns = ["Kode", "Jumlah"]
    counts["Label"] = counts["Kode"].map(lambda k: CLASS_INFO.get(k, {}).get("id_name", k))
    colors = [CLASS_INFO.get(k, {}).get("color", "#888") for k in counts["Kode"]]

    fig = go.Figure(go.Pie(
        labels=counts["Label"], values=counts["Jumlah"],
        hole=0.6, marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=2)),
        textinfo="percent", hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", size=12),
        margin=dict(l=0, r=0, t=10, b=0), height=220,
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1.2rem;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                    color: #ff8c00; font-weight: 700;'>🛣️ PaveDetect</div>
        <div style='font-size: 0.75rem; color: rgba(255,255,255,0.35);
                    letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;'>
            YOLOv9c · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚙️ Parameter Deteksi</div>", unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Confidence Threshold", min_value=0.05, max_value=0.90,
        value=0.25, step=0.05,
        help="Turunkan untuk mendeteksi lebih banyak objek (lebih sensitif)"
    )
    iou_thresh = st.slider(
        "IoU Threshold (NMS)", min_value=0.10, max_value=0.90,
        value=0.50, step=0.05,
        help="Kontrol tumpang tindih bounding box"
    )

    st.markdown("<div class='section-title'>📖 Kelas Kerusakan</div>", unsafe_allow_html=True)
    for code, info in CLASS_INFO.items():
        st.markdown(
            f"<span class='badge {info['badge']}'>{code}</span> "
            f"<span style='font-size:0.82rem; color:rgba(255,255,255,0.65);'>"
            f"{info['id_name']}</span> "
            f"<span style='font-size:0.75rem; color:rgba(255,255,255,0.3);'>{info['risk']}</span>",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(255,255,255,0.25); text-align:center;'>
        Model: YOLOv9c (Fine-tuned)<br>
        Dataset: PaveDistress · 4 Kelas<br>
        mAP50: 0.694 · Precision: 0.799<br>
        Training: 75 + 20 epoch
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>YOLOv9c · OBJECT DETECTION · 4 KELAS</div>
    <div class='hero-title'>🛣️ PaveDetect</div>
    <p class='hero-sub'>
        Sistem deteksi otomatis kerusakan permukaan jalan aspal berbasis deep learning.<br>
        Mendeteksi 4 jenis kerusakan: Retak Memanjang (D00), Retak Melintang (D10),
        Retak Buaya (D20), dan Lubang Jalan (D40).
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
with st.spinner("⏳ Memuat model YOLOv9c..."):
    model = load_model(MODEL_PATH)

# ──────────────────────────────────────────────
# INPUT SECTION
# ──────────────────────────────────────────────
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown("<div class='section-title'>📂 Input Gambar</div>", unsafe_allow_html=True)

    tab_upload, tab_sample = st.tabs(["⬆️  Upload Gambar", "🖼️  Gambar Contoh"])
    uploaded_image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih gambar jalan (.jpg / .png)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_image, caption="Gambar yang diupload",
                     use_container_width=True)

    with tab_sample:
        st.markdown(
            "<div class='info-box'>Klik salah satu gambar contoh di bawah "
            "untuk langsung mencoba deteksi.</div>",
            unsafe_allow_html=True
        )
        for label, path in SAMPLE_IMAGES:
            if os.path.exists(path):
                img_sample = Image.open(path).convert("RGB")
                st.image(img_sample, caption=label, use_container_width=True)
                if st.button(f"🔍 Deteksi — {label}", key=f"btn_{path}"):
                    uploaded_image = img_sample
            else:
                st.caption(f"_{label}_ — tempatkan file di `{path}`")

    st.markdown("<br>", unsafe_allow_html=True)
    detect_btn = st.button(
        "🔍  Jalankan Deteksi",
        disabled=(uploaded_image is None),
        use_container_width=True,
        type="primary",
    )

# ──────────────────────────────────────────────
# RESULT SECTION
# ──────────────────────────────────────────────
with col_result:
    st.markdown("<div class='section-title'>📊 Hasil Deteksi</div>", unsafe_allow_html=True)

    if uploaded_image is None:
        st.markdown(
            "<div class='warn-box'>Belum ada gambar yang dipilih. "
            "Upload gambar atau pilih gambar contoh di panel kiri.</div>",
            unsafe_allow_html=True
        )

    elif detect_btn or ("last_result" in st.session_state
                        and st.session_state.get("last_img") is uploaded_image):

        with st.spinner("🔍 Mendeteksi kerusakan..."):
            img_np   = np.array(uploaded_image)
            result   = run_inference(model, img_np, conf_thresh, iou_thresh)
            df       = parse_results(result)
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            st.session_state["last_result"] = (result, df, annotated)
            st.session_state["last_img"]    = uploaded_image

        result, df, annotated = st.session_state["last_result"]

        # Annotated image
        st.image(annotated, caption="Hasil deteksi dengan bounding box",
                 use_container_width=True)

        # Metric cards
        n_total   = len(df)
        n_classes = df["Kode"].nunique() if n_total > 0 else 0
        avg_conf  = df["Confidence"].mean() if n_total > 0 else 0

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card metric-total'>
                <div class='val'>{n_total}</div>
                <div class='lbl'>Total Deteksi</div>
            </div>
            <div class='metric-card metric-classes'>
                <div class='val'>{n_classes}</div>
                <div class='lbl'>Jenis Kerusakan</div>
            </div>
            <div class='metric-card metric-conf'>
                <div class='val'>{avg_conf:.1f}%</div>
                <div class='lbl'>Rata-rata Conf.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if n_total == 0:
            st.markdown(
                f"<div class='warn-box'>Tidak ada kerusakan terdeteksi pada confidence "
                f"{int(conf_thresh*100)}%. Coba turunkan confidence threshold di sidebar.</div>",
                unsafe_allow_html=True
            )
        else:
            # Charts
            st.markdown("<div class='section-title'>📈 Statistik Deteksi</div>",
                        unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(make_bar_chart(df), use_container_width=True,
                                config={"displayModeBar": False})
            with ch2:
                st.plotly_chart(make_donut_chart(df), use_container_width=True,
                                config={"displayModeBar": False})

            # Detail table
            st.markdown("<div class='section-title'>🗂️ Detail Objek Terdeteksi</div>",
                        unsafe_allow_html=True)
            display_df = df[["Kode", "Jenis Kerusakan", "Confidence", "Risiko"]].copy()
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1f}%")
            display_df = display_df.sort_values("Kode").reset_index(drop=True)
            display_df.index += 1
            st.dataframe(display_df, use_container_width=True,
                         height=min(35 * len(display_df) + 38, 320))

            # Per-class summary
            st.markdown("<div class='section-title'>📋 Ringkasan Per Kelas</div>",
                        unsafe_allow_html=True)
            summary = (
                df.groupby("Kode")
                  .agg(Jumlah=("Kode", "count"),
                       Conf_Avg=("Confidence", "mean"),
                       Conf_Max=("Confidence", "max"))
                  .reset_index()
            )
            summary["Persentase"] = (summary["Jumlah"] / n_total * 100).round(1)
            summary["Jenis"]      = summary["Kode"].map(
                lambda k: CLASS_INFO.get(k, {}).get("id_name", k))
            summary["Risiko"]     = summary["Kode"].map(
                lambda k: CLASS_INFO.get(k, {}).get("risk", "-"))

            for _, row in summary.iterrows():
                badge   = CLASS_INFO.get(row["Kode"], {}).get("badge", "")
                desc    = CLASS_INFO.get(row["Kode"], {}).get("desc", "")
                pct_bar = int(row["Persentase"] / 100 * 18)
                bar_str = "█" * pct_bar + "░" * (18 - pct_bar)
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03); border:1px solid
                     rgba(255,255,255,0.07); border-radius:10px;
                     padding:0.8rem 1rem; margin-bottom:0.5rem;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:4px;'>
                        <span>
                            <span class='badge {badge}'>{row['Kode']}</span>
                            <span style='font-size:0.88rem; color:rgba(255,255,255,0.8);
                                         margin-left:8px;'>{row['Jenis']}</span>
                            <span style='font-size:0.78rem; margin-left:6px;'>{row['Risiko']}</span>
                        </span>
                        <span style='font-family:Space Mono,monospace;
                                     font-size:0.85rem; color:#ff8c00;'>
                            {int(row['Jumlah'])}× &nbsp; {row['Persentase']}%
                        </span>
                    </div>
                    <div style='font-size:0.75rem; color:rgba(255,255,255,0.35);
                                margin-bottom:6px;'>{desc}</div>
                    <div style='font-family:Space Mono,monospace; font-size:0.7rem;
                                color:rgba(255,255,255,0.25); letter-spacing:1px;'>
                        {bar_str}
                        &nbsp; conf avg {row['Conf_Avg']:.1f}% · max {row['Conf_Max']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="JPEG", quality=92)
            st.download_button(
                label     = "⬇️  Download Hasil Deteksi",
                data      = buf.getvalue(),
                file_name = "pavedetect_result.jpg",
                mime      = "image/jpeg",
                use_container_width=True,
            )

    elif uploaded_image is not None:
        st.image(uploaded_image, caption="Preview gambar", use_container_width=True)
        st.markdown(
            "<div class='info-box'>Klik tombol <b>Jalankan Deteksi</b> "
            "di panel kiri untuk memulai analisis.</div>",
            unsafe_allow_html=True
        )
