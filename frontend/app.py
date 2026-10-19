"""
Streamlit Frontend Dashboard for Smart Attendance & Analytics AI Platform.
Provides tabs for:
1. Live Attendance (Webcam feed, bounding boxes, liveness indicator)
2. Register Student (Biometric enrollment with photo)
3. Group Photo Attendance (Multi-face detection & mass attendance logging)
4. Attendance Analytics (Trend charts & student risk predictions)
"""

import os
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Smart Attendance & Analytics AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1")

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Smart Attendance & Analytics AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Research-grade biometric face recognition, anti-spoofing liveness, and predictive student analytics.</div>', unsafe_allow_html=True)

# Sidebar System Status
st.sidebar.title("System Status")
backend_online = False
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        backend_online = True
        health_data = health_resp.json()
        st.sidebar.success(f"Backend Status: {health_data.get('status', 'OK').upper()}")
        st.sidebar.info(f"Database: {health_data.get('database', 'connected')}")
        st.sidebar.caption(f"Detector: {health_data.get('components', {}).get('detector_backend', 'N/A')}")
        st.sidebar.caption(f"Recognizer: {health_data.get('components', {}).get('recognition_model', 'N/A')}")
    else:
        st.sidebar.warning(f"Backend returned status {health_resp.status_code}")
except Exception:
    st.sidebar.error("🚨 Backend Server Offline!")
    st.sidebar.markdown(
        "Run this command in a terminal to start the backend:\n"
        "```powershell\n"
        ".venv\\Scripts\\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload\n"
        "```"
    )

if not backend_online:
    st.warning(
        "⚠️ **FastAPI Backend Server is not running on port 8000!**\n\n"
        "Please start the backend server in a separate terminal using:\n\n"
        "` .venv\\Scripts\\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload `"
    )

# Main Tabs Navigation
tabs = st.tabs([
    "📸 Live Attendance",
    "📝 Register Student",
    "👥 Group Photo Attendance",
    "📊 Analytics & Risk Modeling"
])

# Tab 1: Live Attendance
with tabs[0]:
    st.header("Real-Time Face Recognition & Liveness Verification")
    st.info("Phase 2 & 3: Deep learning webcam feed with YOLOv8 face detection, FaceNet512 embeddings, and MediaPipe EAR blink detection.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Camera Feed**")
        st.caption("Live video stream with real-time bounding boxes and anti-spoofing indicators.")
        st.empty().text("Webcam pipeline activates in Phase 2 & 3.")
    with col2:
        st.markdown("**Verification Output**")
        st.metric("Detected Identity", "None")
        st.metric("Match Confidence", "0.0%")
        st.metric("Liveness Status", "Pending")

# Tab 2: Register Student
with tabs[1]:
    st.header("Enroll New Student")
    st.markdown("Register student metadata and biometric reference portraits via **multi-sample acquisition** (webcam or file upload) for maximum accuracy.")

    # Session State initialization for multi-sample webcam capture
    if "webcam_active" not in st.session_state:
        st.session_state["webcam_active"] = False
    if "captured_samples" not in st.session_state:
        st.session_state["captured_samples"] = []

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        reg_name = st.text_input("Full Name *", placeholder="e.g. Marie Curie", key="reg_name_input")
    with col_meta2:
        reg_roll = st.text_input("Roll / Matriculation Number *", placeholder="e.g. CS-2024-042", key="reg_roll_input")

    st.markdown("---")
    st.subheader("Biometric Reference Portraits (Multi-Sample Acquisition)")
    st.caption("Capturing multiple samples (different angles, lighting, or expressions) allows the system to compute a robust centroid embedding that maximizes recognition accuracy.")

    capture_mode = st.radio(
        "Choose Portrait Acquisition Method:",
        ["📷 Capture via Webcam", "📁 Upload Image Files"],
        horizontal=True,
        key="reg_capture_mode"
    )

    sample_buffers = []

    if capture_mode == "📷 Capture via Webcam":
        if not st.session_state["webcam_active"]:
            st.info("🔒 The camera is currently closed. Click the button below when ready to activate your webcam.")
            if st.button("📷 Open Camera & Start Taking Samples", type="secondary"):
                st.session_state["webcam_active"] = True
                st.rerun()
        else:
            col_cam_header, col_cam_btn = st.columns([3, 1])
            with col_cam_header:
                st.success("🟢 Camera is ACTIVE. Look at the camera and click **Take Photo** to record a sample.")
            with col_cam_btn:
                if st.button("❌ Close Camera"):
                    st.session_state["webcam_active"] = False
                    st.rerun()

            camera_img = st.camera_input("Webcam Multi-Sample Capture", key="reg_webcam_input")
            if camera_img is not None:
                shot_bytes = camera_img.getvalue()
                if "last_shot_hash" not in st.session_state or st.session_state["last_shot_hash"] != hash(shot_bytes):
                    st.session_state["captured_samples"].append(shot_bytes)
                    st.session_state["last_shot_hash"] = hash(shot_bytes)

            # Display captured samples gallery
            if st.session_state["captured_samples"]:
                st.markdown(f"**Captured Samples ({len(st.session_state['captured_samples'])}):**")
                cols = st.columns(min(len(st.session_state["captured_samples"]), 6))
                for i, s_bytes in enumerate(st.session_state["captured_samples"]):
                    cols[i % 6].image(s_bytes, caption=f"Sample #{i+1}", width=110)

                if st.button("🗑️ Clear Captured Samples"):
                    st.session_state["captured_samples"] = []
                    st.session_state.pop("last_shot_hash", None)
                    st.rerun()

            sample_buffers = [
                (f"webcam_sample_{idx+1}.jpg", s_bytes)
                for idx, s_bytes in enumerate(st.session_state["captured_samples"])
            ]

    else:
        st.caption("Upload 1 to 5 clear frontal portraits (JPG, JPEG, or PNG).")
        uploaded_imgs = st.file_uploader(
            "Select Portrait Files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="reg_file_uploader"
        )
        if uploaded_imgs:
            st.markdown(f"**Selected Files ({len(uploaded_imgs)}):**")
            cols = st.columns(min(len(uploaded_imgs), 6))
            for i, f_obj in enumerate(uploaded_imgs):
                cols[i % 6].image(f_obj, caption=f"Sample #{i+1}", width=110)
                sample_buffers.append((f_obj.name, f_obj.getvalue()))

    st.markdown("---")
    enroll_btn = st.button("Enroll Student", type="primary", use_container_width=True)
    if enroll_btn:
        clean_name = reg_name.strip()
        clean_roll = reg_roll.strip()

        if not clean_name or not clean_roll:
            st.warning("Please provide both Full Name and Roll Number.")
        elif not sample_buffers:
            st.warning("Please provide at least one face portrait sample (via webcam or file upload).")
        else:
            with st.spinner("Analyzing biometric face quality across samples and generating centroid embedding..."):
                try:
                    # Package multiple files for multipart upload
                    multipart_files = [
                        ("files", (filename, data, "image/jpeg"))
                        for filename, data in sample_buffers
                    ]
                    payload = {"name": clean_name, "roll_no": clean_roll}
                    res = requests.post(f"{BACKEND_URL}/register", data=payload, files=multipart_files, timeout=15)

                    if res.status_code == 201:
                        data = res.json()
                        samples_done = data.get("samples_processed", len(sample_buffers))
                        st.success(
                            f"🎉 Student **{clean_name}** ({clean_roll}) registered successfully with **{samples_done}** biometric face samples!"
                        )
                        st.session_state["captured_samples"] = []
                        st.session_state.pop("last_shot_hash", None)
                        st.balloons()
                    elif res.status_code == 409:
                        st.error(f"Student with Roll Number **'{clean_roll}'** already exists in the database.")
                    elif res.status_code == 422:
                        detail = res.json().get("detail", "Face detection failed.")
                        st.error(f"Biometric Validation Error: {detail}")
                    else:
                        st.error(f"Registration failed ({res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🚨 **Connection Refused! The FastAPI backend is not running.**\n\n"
                        "Please open a terminal and run:\n"
                        "```powershell\n"
                        ".venv\\Scripts\\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload\n"
                        "```"
                    )
                except Exception as err:
                    st.error(f"Failed to communicate with backend: {err}")

# Tab 3: Group Photo Attendance
with tabs[2]:
    st.header("Classroom Group Photo Recognition")
    st.markdown("Upload a high-resolution classroom photo to simultaneously detect, recognize, and mark attendance for all students.")
    group_file = st.file_uploader("Upload Classroom Image", type=["jpg", "jpeg", "png"], key="group_uploader")
    if group_file:
        st.image(group_file, caption="Uploaded Classroom Image", use_column_width=True)
        if st.button("Process Group Attendance"):
            st.info("Group photo multi-face recognition will be processed by YOLOv8 + FaceNet512 pipeline in Phase 2.")

# Tab 4: Analytics
with tabs[3]:
    st.header("Attendance Analytics & Risk Forecasting")
    st.markdown("Automated attendance trend visualization and predictive student engagement modeling.")
    try:
        att_res = requests.get(f"{BACKEND_URL}/attendance", timeout=3)
        if att_res.status_code == 200:
            records = att_res.json()
            st.write(f"Total Attendance Records Logged: **{len(records)}**")
            if records:
                st.dataframe(records, use_container_width=True)
        else:
            st.warning("Unable to fetch attendance records.")
    except Exception as e:
        st.warning(f"Backend not reachable: {e}")
