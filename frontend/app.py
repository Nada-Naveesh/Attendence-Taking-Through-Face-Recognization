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
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(f"Backend Status: {health_data.get('status', 'OK').upper()}")
        st.sidebar.info(f"Database: {health_data.get('database', 'connected')}")
        st.sidebar.caption(f"Detector: {health_data.get('components', {}).get('detector_backend', 'N/A')}")
        st.sidebar.caption(f"Recognizer: {health_data.get('components', {}).get('recognition_model', 'N/A')}")
    else:
        st.sidebar.warning(f"Backend returned status {health_resp.status_code}")
except Exception as e:
    st.sidebar.error("Backend Offline. Start FastAPI server.")
    st.sidebar.caption(str(e))

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
    with st.form("register_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input("Full Name", placeholder="e.g. Marie Curie")
            roll_no = st.text_input("Roll / Matriculation Number", placeholder="e.g. CS-2024-042")
        with col_b:
            uploaded_photo = st.file_uploader("Upload Biometric Reference Portrait", type=["jpg", "jpeg", "png"])

        submitted = st.form_submit_button("Enroll Student")
        if submitted:
            if not name or not roll_no:
                st.warning("Please provide student name and roll number.")
            else:
                try:
                    payload = {"name": name, "roll_no": roll_no}
                    res = requests.post(f"{BACKEND_URL}/students", json=payload)
                    if res.status_code == 201:
                        st.success(f"Student '{name}' registered successfully!")
                    elif res.status_code == 409:
                        st.error(f"Student with roll number '{roll_no}' already exists.")
                    else:
                        st.error(f"Registration failed: {res.text}")
                except Exception as err:
                    st.error(f"Backend connection error: {err}")

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
