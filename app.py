"""
Railway Track Defect Detection System
"""
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import time
import os

# ======================================================================================================================
# Page Config and Global Styles
# ======================================================================================================================
st.set_page_config(
    page_title="RailGuard AI - Deteksi Kerusakan Rel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more modern and appealing look
st.markdown("""
    <style>
        /* Main Theme Colors */
        :root {
            --primary-color: #3498DB;       /* A nice blue for accents */
            --background-color: #1C2833;   /* Very dark, blue-ish grey */
            --card-background-color: #283747; /* Slightly lighter card background */
            --text-color: #FDFEFE;         /* Almost white text */
            --subtext-color: #AAB7B8;       /* Muted grey for subtext */
            --border-color: #3C4F60;        /* Border color */
            --success-color: #2ECC71;
            --warning-color: #F39C12;
            --error-color: #E74C3C;
            --critical-color: #C0392B;
        }

        /* General Body */
        body {
            color: var(--text-color);
        }
        .main .block-container {
            padding: 1.5rem 2rem;
            max-width: 100%;
            background-color: var(--background-color);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }

        /* Sidebar */
        .st-emotion-cache-16txtl3 {
            padding: 1.5rem 1rem;
            background-color: var(--card-background-color);
            border-right: 1px solid var(--border-color);
        }
        .st-emotion-cache-16txtl3 h1 {
            color: var(--primary-color);
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid var(--primary-color);
            transition: background-color 0.3s, transform 0.1s;
        }
        .stButton>button:hover {
            background-color: #4A90C2; /* Slightly different blue on hover */
            border-color: #4A90C2;
            color: white;
            transform: scale(1.02);
        }
        .stButton>button:active {
            transform: scale(0.98);
        }


        /* File Uploader */
        .stFileUploader {
            border: 2px dashed var(--primary-color);
            border-radius: 12px;
            padding: 1.5rem;
            background-color: var(--background-color);
        }

        /* Expander */
        .st-emotion-cache-p5msec {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: none;
            background-color: var(--card-background-color);
        }
        .st-emotion-cache-p5msec .streamlit-expanderHeader {
            color: var(--text-color);
            font-weight: 600;
        }

        /* Card-like containers for metrics */
        .metric-card {
            background-color: var(--card-background-color);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: none;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        .metric-card h4 {
            color: var(--subtext-color);
            font-weight: 500;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        .metric-card p {
            color: var(--text-color);
            font-size: 2rem;
            font-weight: 600;
        }

        /* Custom alert-like div for risk level */
        .risk-display-div {
            background-color: var(--card-background-color);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 8px solid; /* Default border, color will be set inline */
        }
        .risk-display-div h3 {
            margin: 0;
            /* Color will be set inline */
        }
        .risk-display-div p {
            margin: 0.5rem 0 0 0;
            font-weight: 500;
            color: var(--subtext-color); /* Subtext color for explanation */
        }
    </style>
""", unsafe_allow_html=True)


# ======================================================================================================================
# Session State and Model Loading
# ======================================================================================================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'stop_webcam' not in st.session_state:
    st.session_state.stop_webcam = False

@st.cache_resource
def load_yolov5_model(model_path='best.pt'):
    try:
        if not os.path.exists(model_path):
            return None, False, f"File model '{model_path}' tidak ditemukan!"
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.conf = 0.25
        model.iou = 0.45
        return model, True, "Model berhasil dimuat!"
    except Exception as e:
        return None, False, f"Terjadi kesalahan saat memuat model: {e}"


# ======================================================================================================================
# Core Detection and Drawing Functions
# ======================================================================================================================
def draw_boxes_on_image(image, results):
    img_array = np.array(image.convert("RGB"))
    detections = results.pandas().xyxy[0]
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        class_name = detection['name']
        
        # Color based on confidence
        if confidence > 0.75: color = (76, 175, 80) # Green
        elif confidence > 0.5: color = (251, 140, 0) # Orange
        else: color = (211, 47, 47) # Red
        
        label = f'{class_name}: {confidence:.2f}'
        
        # Draw rectangle and label
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_array, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_array

def detect_objects(image, model):
    results = model(image)
    annotated_img = draw_boxes_on_image(image, results)
    detections_df = results.pandas().xyxy[0]
    return annotated_img, len(detections_df), detections_df

def process_video_file(video_path, model, progress_bar, status_text):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, _, _ = detect_objects(Image.fromarray(frame_rgb), model)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
            frame_num += 1
            progress_bar.progress(frame_num / total_frames)
            status_text.text(f"Memproses: {frame_num}/{total_frames} frame")
            
        cap.release()
        out.release()
        return output_path, True
    except Exception as e:
        st.error(f"Error pemrosesan video: {e}")
        return None, False


# ======================================================================================================================
# Main Application UI
# ======================================================================================================================
def main():
    # --- RISK DEFINITIONS ---
    RISK_DEFINITIONS = {
        "fishplate": {"level": "Rendah", "explanation": "Komponen penghubung rel (fishplate) terdeteksi. Disarankan untuk inspeksi visual rutin.", "order": 1},
        "fishplate_bolthead": {"level": "Rendah", "explanation": "Kepala baut pada fishplate terdeteksi. Disarankan untuk inspeksi visual rutin.", "order": 1},
        "fishplate_boltmissing": {"level": "Tinggi", "explanation": "Baut yang hilang pada fishplate sangat berbahaya karena dapat mengurangi kekuatan dan kekakuan sambungan rel, berisiko menyebabkan kegagalan sambungan.", "order": 3},
        "fishplate_boltnut": {"level": "Rendah", "explanation": "Mur baut pada fishplate terdeteksi. Disarankan untuk inspeksi visual rutin.", "order": 1},
        "track_bolt": {"level": "Sedang", "explanation": "Baut track terdeteksi, perlu dipastikan kekencangannya. Baut yang kendor dapat mengurangi stabilitas rel.", "order": 2},
        "track_boltmissing": {"level": "Tinggi", "explanation": "Baut track yang hilang dapat menyebabkan pergeseran rel dan ketidakstabilan, yang berisiko tinggi mengakibatkan anjloknya kereta.", "order": 3},
        "track_crack": {"level": "Kritis", "explanation": "Retakan pada rel adalah kerusakan kritis yang dapat menyebabkan rel patah secara tiba-tiba di bawah beban kereta. Perlu penanganan segera!", "order": 4},
        "default": {"level": "Rendah", "explanation": "Terdeteksi cacat yang tidak teridentifikasi. Disarankan untuk pemeriksaan manual.", "order": 1}
    }

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>🤖 RailGuard AI</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        detection_mode = st.radio(
            "Pilih Mode Deteksi:",
            ["🖼️ Deteksi Gambar", "🎬 Deteksi Video", "📹 Deteksi Webcam"],
            captions=["Analisis satu gambar", "Proses file video", "Deteksi real-time dari kamera"]
        )
        
        st.markdown("---")
        st.subheader("Informasi Sistem")
        st.info(f"""
            **Perangkat**: `{'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}`
            **Model**: `YOLOv5 Custom`
            **Confidence**: `25%`
            **IOU**: `45%`
        """)

        st.markdown("---")
        st.markdown("<p style='text-align: center; color: var(--subtext-color);'>Dibuat dengan ❤️ untuk Keselamatan Perkeretaapian</p>", unsafe_allow_html=True)

    # --- MODEL LOADING ---
    if not st.session_state.model_loaded:
        with st.spinner("Memuat model RailGuard AI... Mohon tunggu sebentar."):
            model, success, message = load_yolov5_model('best.pt')
            if success:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.toast("✅ Model berhasil dimuat!", icon="🤖")
            else:
                st.error(f"❌ {message}")
                st.info("Pastikan `best.pt` ada di folder yang sama dan coba lagi.")
                st.stop()
    model = st.session_state.model

    # --- MAIN HEADER ---
    st.markdown("<h1 style='text-align: center;'>Sistem Deteksi Kerusakan Rel Berbasis AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--subtext-color);'>Analisis visual real-time untuk meningkatkan keselamatan dan pemeliharaan infrastruktur kereta api.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- IMAGE DETECTION UI ---
    if detection_mode == "🖼️ Deteksi Gambar":
        uploaded_file = st.file_uploader("Pilih atau seret file gambar ke sini", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown("---")
            
            with st.spinner("🔍 Menganalisis gambar..."):
                annotated_img, num_detections, detections_df = detect_objects(image, model)

            # --- Display Results ---
            st.subheader("Hasil Analisis")
            
            # Risk Analysis
            overall_risk_level, risk_explanation, risk_color, highest_risk_order = "Tidak Ada Risiko", "Tidak ada kerusakan terdeteksi.", "var(--success-color)", 0
            if num_detections > 0:
                for _, row in detections_df.iterrows():
                    defect_info = RISK_DEFINITIONS.get(row['name'], RISK_DEFINITIONS["default"])
                    if defect_info["order"] > highest_risk_order:
                        highest_risk_order, overall_risk_level, risk_explanation = defect_info["order"], f"Risiko {defect_info['level']}", defect_info['explanation']
                
                if highest_risk_order == 4: risk_color = "var(--critical-color)"
                elif highest_risk_order == 3: risk_color = "var(--error-color)"
                elif highest_risk_order == 2: risk_color = "var(--warning-color)"
                else: risk_color = "gold"

            # Display Risk Level and Explanation
            st.markdown(f"<div class='risk-display-div' style='border-left-color: {risk_color};'>"
                        f"<h3 style='color: {risk_color};'>{overall_risk_level}</h3>"
                        f"<p>{risk_explanation}</p></div>", unsafe_allow_html=True)

            # Display Images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Asli")
            with col2:
                st.image(annotated_img, caption="Hasil Deteksi Anotasi")

            st.markdown("---")

            # Display Metrics in Cards
            st.subheader("Ringkasan Metrik")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"<div class='metric-card'><h4>Total Deteksi</h4><p>{num_detections}</p></div>", unsafe_allow_html=True)
            with m_col2:
                max_conf = f"{detections_df['confidence'].max():.0%}" if num_detections > 0 else "N/A"
                st.markdown(f"<div class='metric-card'><h4>Keyakinan Tertinggi</h4><p>{max_conf}</p></div>", unsafe_allow_html=True)
            with m_col3:
                min_conf = f"{detections_df['confidence'].min():.0%}" if num_detections > 0 else "N/A"
                st.markdown(f"<div class='metric-card'><h4>Keyakinan Terendah</h4><p>{min_conf}</p></div>", unsafe_allow_html=True)

            # Expander for Detailed Data
            if num_detections > 0:
                with st.expander("Lihat Rincian Kerusakan per Kategori"):
                    class_counts = detections_df['name'].value_counts()
                    st.table(class_counts)
                    
                    csv = detections_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Unduh Data Deteksi (CSV)", csv, "deteksi_kerusakan.csv", "text/csv")

    # --- VIDEO & WEBCAM UI ---
    elif detection_mode in ["🎬 Deteksi Video", "📹 Deteksi Webcam"]:
        if detection_mode == "🎬 Deteksi Video":
            st.header("🎬 Analisis File Video")
            uploaded_video = st.file_uploader("Pilih file video (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])
            if uploaded_video:
                if st.button("🚀 Mulai Analisis Video"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    
                    progress_bar, status_text = st.progress(0), st.empty()
                    output_path, success = process_video_file(tfile.name, model, progress_bar, status_text)
                    
                    if success:
                        st.success("✅ Pemrosesan video selesai!")
                        st.video(output_path)
                        with open(output_path, 'rb') as f:
                            st.download_button("Unduh Video Hasil Proses", f, "deteksi_video.mp4")
                    else:
                        st.error("❌ Gagal memproses video.")
        
        else: # Webcam
            st.header("📹 Deteksi Real-Time via Webcam")
            camera_source = st.text_input("Sumber Kamera (0 untuk default, atau URL RTSP/IP Cam)", "0")
            
            if st.button("▶️ Mulai Kamera"):
                st.session_state.stop_webcam = False
                try:
                    source = int(camera_source) if camera_source.isdigit() else camera_source
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        st.error("❌ Tidak dapat membuka kamera. Periksa sumber.")
                    else:
                        frame_placeholder = st.empty()
                        while not st.session_state.stop_webcam:
                            ret, frame = cap.read()
                            if not ret: break
                            
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            annotated_frame, _, _ = detect_objects(Image.fromarray(frame_rgb), model)
                            frame_placeholder.image(annotated_frame)
                        cap.release()
                        st.info("✅ Kamera dihentikan.")
                except Exception as e:
                    st.error(f"❌ Error kamera: {e}")
            
            if st.button("⏹️ Hentikan Kamera"):
                st.session_state.stop_webcam = True


if __name__ == "__main__":
    main()
