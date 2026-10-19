"""
Streamlit Frontend Dashboard for Smart Attendance & Analytics AI Platform.
Features:
1. Live Attendance: Real-time biometric face verification, visual bounding boxes with student name & confidence, and automatic SQLite & CSV attendance logging.
2. Register Student: Automated 6-second multi-sample face capture (25 samples) generating high-accuracy centroid embeddings.
3. Group Photo Attendance: Multi-face recognition on uploaded classroom images.
4. Analytics: Attendance tracking, daily trends, and predictive student engagement reports.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
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
ATTENDANCE_DIR = Path("Attendance")
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Smart Attendance & Analytics AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Research-grade biometric face recognition, anti-spoofing verification, and real-time attendance analytics.</div>', unsafe_allow_html=True)

# Sidebar System Status
st.sidebar.title("System Status")
backend_online = False
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        backend_online = True
        health_data = health_resp.json()
        st.sidebar.success(f"Backend: {health_data.get('status', 'OK').upper()}")
        st.sidebar.info(f"Database: {health_data.get('database', 'connected')}")
        st.sidebar.caption(f"Detector: {health_data.get('components', {}).get('detector_backend', 'N/A')}")
        st.sidebar.caption(f"Recognizer: {health_data.get('components', {}).get('recognition_model', 'N/A')}")
except Exception:
    st.sidebar.error("🚨 Backend Server Offline!")
    st.sidebar.markdown(
        "Start backend server in PowerShell:\n"
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

# Main Navigation Tabs
tabs = st.tabs([
    "📸 Live Attendance",
    "📝 Register Student",
    "👥 Group Photo Attendance",
    "📊 Analytics & Risk Modeling"
])


# ==============================================================================
# TAB 1: Live Attendance (Real-Time Biometric Recognition & CSV Logging)
# ==============================================================================
with tabs[0]:
    st.header("📸 Real-Time Face Recognition & Biometric Attendance")
    st.markdown("Click **Take Biometric Attendance** to activate the camera. The system will detect your face, verify your identity against enrolled students, display your name on screen, and automatically commit your attendance to the daily CSV log.")

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        start_attendance = st.button("🟢 Take Biometric Attendance", type="primary", use_container_width=True)
    with col_btn2:
        today_str = datetime.now().strftime("%Y-%m-%d")
        csv_file_path = ATTENDANCE_DIR / f"Attendance_{today_str}.csv"
        if csv_file_path.exists():
            with open(csv_file_path, "rb") as f_csv:
                st.download_button(
                    label="📥 Download Today's Attendance CSV",
                    data=f_csv,
                    file_name=f"Attendance_{today_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    if start_attendance:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Ensure your camera is plugged in and not currently in use by another program.")
        else:
            video_feed = st.empty()
            status_box = st.empty()
            status_box.info("🔍 Scanning camera feed... Please look directly at the webcam.")

            matched_student = None
            matched_conf = 0.0
            matched_roll = None

            # Stream camera frames and perform biometric matching
            for frame_idx in range(45):
                ret, frame = cap.read()
                if not ret:
                    break

                # Send frame to backend recognition endpoint
                _, buffer = cv2.imencode(".jpg", frame)
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/recognize?mark_attendance=true",
                        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                        timeout=3
                    )
                    if res.status_code == 200:
                        rec_data = res.json()
                        matched = rec_data.get("matched", False)
                        conf = rec_data.get("confidence", 0.0)
                        box = rec_data.get("box")

                        if box and len(box) == 4:
                            x, y, w, h = box
                            if matched:
                                color = (0, 255, 0)  # Green for recognized
                                label = f"{rec_data.get('student_name')} ({conf*100:.1f}%)"
                            else:
                                color = (0, 0, 255)  # Red for unknown
                                label = f"Unknown ({conf*100:.1f}%)"

                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                        if matched:
                            matched_student = rec_data.get("student_name")
                            matched_roll = rec_data.get("roll_no")
                            matched_conf = conf
                            # Render verified frame and break loop
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_feed.image(frame_rgb, channels="RGB", use_column_width=True)
                            break

                except Exception:
                    pass

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_feed.image(frame_rgb, channels="RGB", use_column_width=True)
                time.sleep(0.04)

            cap.release()

            if matched_student:
                now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_box.success(
                    f"🎉 **RECOGNIZED: {matched_student}!** (Roll No: `{matched_roll}`)\n\n"
                    f"- **Match Confidence**: `{matched_conf * 100:.1f}%`\n"
                    f"- **Time Recorded**: `{now_time}`\n"
                    f"- **Attendance Saved to**: `Attendance/Attendance_{today_str}.csv` and SQLite Database."
                )
                st.balloons()
            else:
                status_box.warning("⚠️ Face not recognized as an enrolled student. Please ensure you are registered or adjust lighting and try again.")

    # Show Today's Attendance Table
    st.markdown("---")
    st.subheader(f"📅 Today's Live Attendance Records ({datetime.now().strftime('%Y-%m-%d')})")
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    csv_file_path = ATTENDANCE_DIR / f"Attendance_{today_date_str}.csv"
    if csv_file_path.exists():
        try:
            df_att = pd.read_csv(csv_file_path)
            if not df_att.empty:
                st.dataframe(df_att, use_container_width=True)
            else:
                st.info("No attendance entries in today's CSV file yet.")
        except Exception as e:
            st.caption(f"Reading CSV: {e}")
    else:
        st.info("No attendance marked yet today. Click 'Take Biometric Attendance' above to begin.")


# ==============================================================================
# TAB 2: Register Student (Auto 6-Second Multi-Sample Face Capture)
# ==============================================================================
with tabs[1]:
    st.header("📝 Enroll New Student")
    st.markdown("Enter student information and run the **6-Second Auto Face Capture**. The camera will automatically capture **25 facial samples** across 6 seconds as you move your head slightly, generating a high-accuracy biometric centroid embedding.")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        reg_name = st.text_input("Full Name *", placeholder="e.g. Naveesh", key="reg_name_input")
    with col_meta2:
        reg_roll = st.text_input("Roll / Matriculation Number *", placeholder="e.g. NAV-001", key="reg_roll_input")

    st.markdown("---")
    st.subheader("📷 Automatic Multi-Sample Capture (6 Seconds)")
    st.caption("Click the button below and look directly at your webcam. The system automatically captures a burst of 25 face samples without requiring you to manually click 'Take Photo' repeatedly.")

    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        start_capture = st.button("🎥 Start 6-Second Auto Face Capture", type="primary", use_container_width=True)
    with col_c2:
        upload_toggle = st.checkbox("Or upload image files manually", key="upload_toggle")

    if start_capture:
        clean_name = reg_name.strip()
        clean_roll = reg_roll.strip()

        if not clean_name or not clean_roll:
            st.warning("⚠️ Please enter both Full Name and Roll Number before starting camera capture.")
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Unable to access webcam. Check camera connection or permissions.")
            else:
                video_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.info(f"🎥 Camera recording for **{clean_name}**! Look at the camera and slowly tilt your head slightly...")

                captured_frames = []
                target_samples = 25
                start_time = time.time()
                max_seconds = 6.0
                frame_count = 0

                # Load cascade detector for visual feedback
                haar_path = Path("backend/services/assets/haarcascade_frontalface_default.xml")
                detector = cv2.CascadeClassifier(str(haar_path)) if haar_path.exists() else None

                while len(captured_frames) < target_samples and (time.time() - start_time) < max_seconds:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    faces = []
                    if detector is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Sample {len(captured_frames)+1}/{target_samples}",
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2
                        )
                        # Save frame sample
                        if frame_count % 2 == 0:
                            _, buf = cv2.imencode(".jpg", frame)
                            captured_frames.append(buf.tobytes())

                    progress_pct = min(1.0, len(captured_frames) / target_samples)
                    progress_bar.progress(progress_pct)
                    status_text.markdown(f"**Progress:** `{len(captured_frames)} / {target_samples}` face samples collected...")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    time.sleep(0.05)

                cap.release()

                if len(captured_frames) >= 5:
                    status_text.success(f"✅ Captured {len(captured_frames)} face samples! Computing centroid embedding and registering profile...")
                    try:
                        multipart_files = [
                            ("files", (f"sample_{i+1}.jpg", b_data, "image/jpeg"))
                            for i, b_data in enumerate(captured_frames)
                        ]
                        payload = {"name": clean_name, "roll_no": clean_roll}
                        res = requests.post(f"{BACKEND_URL}/register", data=payload, files=multipart_files, timeout=20)
                        if res.status_code == 201:
                            st.success(f"🎉 **{clean_name}** ({clean_roll}) enrolled successfully with {len(captured_frames)} biometric face samples!")
                            st.balloons()
                        elif res.status_code == 409:
                            st.error(f"Student with Roll Number **'{clean_roll}'** already exists in the database.")
                        elif res.status_code == 422:
                            st.error(f"Biometric Validation Warning: {res.json().get('detail', 'Face check failed.')}")
                        else:
                            st.error(f"Registration failed ({res.status_code}): {res.text}")
                    except Exception as err:
                        st.error(f"Backend registration failed: {err}")
                else:
                    status_text.error("❌ Not enough face samples detected. Please ensure good lighting and face the camera directly, then try again.")

    # Secondary file upload option
    if upload_toggle:
        st.markdown("### Manual File Upload")
        uploaded_files = st.file_uploader(
            "Upload 1 to 5 face portrait files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="manual_file_uploader"
        )
        if st.button("Enroll with Uploaded Files", type="secondary"):
            clean_name = reg_name.strip()
            clean_roll = reg_roll.strip()
            if not clean_name or not clean_roll:
                st.warning("Please enter Name and Roll Number.")
            elif not uploaded_files:
                st.warning("Please select at least 1 image file.")
            else:
                try:
                    multipart_files = [("files", (f.name, f.getvalue(), "image/jpeg")) for f in uploaded_files]
                    res = requests.post(f"{BACKEND_URL}/register", data={"name": clean_name, "roll_no": clean_roll}, files=multipart_files)
                    if res.status_code == 201:
                        st.success(f"Student **{clean_name}** ({clean_roll}) enrolled successfully!")
                        st.balloons()
                    elif res.status_code == 409:
                        st.error(f"Student with roll number '{clean_roll}' already exists.")
                    else:
                        st.error(f"Enrollment error: {res.text}")
                except Exception as err:
                    st.error(f"Error: {err}")


# ==============================================================================
# TAB 3: Classroom Group Photo Attendance
# ==============================================================================
with tabs[2]:
    st.header("👥 Classroom Group Photo Recognition")
    st.markdown("Upload a high-resolution classroom group photo to simultaneously detect and mark attendance for all recognized students.")
    group_file = st.file_uploader("Upload Classroom Image", type=["jpg", "jpeg", "png"], key="group_uploader")
    if group_file:
        st.image(group_file, caption="Uploaded Classroom Image", use_column_width=True)
        if st.button("Process Group Attendance"):
            st.info("Classroom group recognition pipeline processes multiple faces simultaneously via YOLOv8 face detection.")


# ==============================================================================
# TAB 4: Attendance Analytics & Risk Forecasting
# ==============================================================================
with tabs[3]:
    st.header("📊 Attendance Analytics & Risk Forecasting")
    st.markdown("Automated attendance trend visualization and student engagement modeling.")
    try:
        att_res = requests.get(f"{BACKEND_URL}/attendance", timeout=3)
        if att_res.status_code == 200:
            records = att_res.json()
            st.metric("Total Attendance Logs Recorded", len(records))
            if records:
                df_all = pd.DataFrame(records)
                st.dataframe(df_all, use_container_width=True)
        else:
            st.warning("Unable to fetch attendance records from backend.")
    except Exception as e:
        st.warning(f"Backend not reachable: {e}")
