"""
Streamlit User Interface for Real-Time Object Detection & Logging Platform.
Displays real-time video stream, annotated bounding boxes, detection metrics,
and live database event logs from MySQL.
"""

import time
import os
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from detector import ObjectDetector
import database


# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection & Logging Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom Styling for modern dark/light dashboard
st.markdown("""
<style>
    :root {
        --mobile-breakpoint: 768px;
    }

    .main-header {
        font-size: clamp(1.6rem, 3vw, 2.4rem);
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 1rem;
        }

        div[data-testid="stSidebar"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        div[data-testid="stSidebar"] > div {
            padding: 0.75rem !important;
        }

        .main-header {
            font-size: 1.7rem;
        }

        .sub-header {
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            min-height: 90px;
        }

        button[kind="primary"],
        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Cache the YOLO detector instance across Streamlit reruns."""
    return ObjectDetector(model_name="yolov8n.pt")


def main():
    detector = load_detector()
    available_classes = detector.get_available_classes()

    # ----------------- SIDEBAR -----------------
    st.sidebar.title("🛠️ Platform Settings")

    st.sidebar.subheader("🎯 Detection Filters")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Only log and display objects detected above this confidence score.",
    )

    default_classes = [c for c in ["person", "cell phone", "laptop", "bottle"] if c in available_classes]
    selected_classes = st.sidebar.multiselect(
        "Select Objects to Detect",
        options=["All"] + available_classes,
        default=default_classes if default_classes else ["person"],
        help="Filter which object categories to detect and log.",
    )
    if "All" in selected_classes:
        selected_classes = ["All"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📹 Video Source")
    video_source_type = st.sidebar.radio(
        "Choose Video Input",
        options=["Webcam", "Sample / Test Video", "Upload Video File"],
        index=0,
    )

    camera_idx = 0
    uploaded_file_path = None

    if video_source_type == "Webcam":
        camera_idx = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    elif video_source_type == "Upload Video File":
        uploaded_video = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            uploaded_file_path = f"/tmp/{uploaded_video.name}"
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_video.read())

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗄️ Database Logging")
    enable_db_logging = st.sidebar.checkbox("Enable Real-Time Event Logging", value=True)
    debounce_sec = st.sidebar.slider(
        "Logging Cooldown (Seconds)",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.2,
        help="Prevents duplicate flood inserts for the same object in consecutive video frames.",
    )

    # Test database connectivity
    db_ok, db_status_msg = database.check_connection()

    with st.sidebar.expander("🔌 Database Info & Status", expanded=False):
        st.write(f"**Host:** `{database.DB_HOST}:{database.DB_PORT}`")
        st.write(f"**Database:** `{database.DB_NAME}`")
        st.write(f"**User:** `{database.DB_USER}`")
        if db_ok:
            st.success("🟢 Connected to MySQL Server")
        else:
            st.warning("🟡 MySQL Offline – Active in SQLite Fallback")
            st.caption("To use MySQL, make sure MySQL server is running (e.g. XAMPP or MySQL Workbench).")
        if st.button("🔄 Test / Reconnect DB"):
            res = database.init_database()
            st.rerun()

    # ----------------- MAIN HEADER -----------------
    col_head_left, col_head_right = st.columns([3, 1])
    with col_head_left:
        st.markdown('<div class="main-header">🎯 Real-Time Object Detection & Logging Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Assignment 5 — Modular Computer Vision Architecture with YOLOv8, OpenCV, and MySQL Event Logging</div>', unsafe_allow_html=True)

    with col_head_right:
        if db_ok:
            st.markdown(f'<div style="text-align:right;"><span class="status-badge-ok">🟢 MySQL Live ({database.DB_NAME})</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:right;"><span class="status-badge-warn">🟡 Fallback DB Active</span></div>', unsafe_allow_html=True)

    # ----------------- TOP METRIC BAR -----------------
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metrics = database.fetch_metrics()
    total_db_logs = metrics.get("total_logs", 0)
    avg_confidence = metrics.get("avg_confidence", 0.0)

    fps_metric = metric_col1.empty()
    frame_objs_metric = metric_col2.empty()
    db_total_metric = metric_col3.empty()
    avg_conf_metric = metric_col4.empty()

    fps_metric.metric("Processing Speed", "0.0 FPS")
    frame_objs_metric.metric("Objects in Frame", "0")
    db_total_metric.metric("Total DB Logs", f"{total_db_logs}")
    avg_conf_metric.metric("Avg Logged Conf", f"{avg_confidence:.2f}")

    st.markdown("---")

    # ----------------- STREAM & LOGS VIEW -----------------
    stream_col, stats_col = st.columns([3, 2])

    with stream_col:
        st.subheader("📺 Live Video Stream")
        video_placeholder = st.empty()

        col_btn1, col_btn2 = st.columns(2)
        start_btn = col_btn1.button("▶️ Start Camera / Video", use_container_width=True)
        stop_btn = col_btn2.button("⏹️ Stop Stream", use_container_width=True)

    with stats_col:
        st.subheader("📊 Current Detections")
        current_dets_container = st.empty()

        st.subheader("📈 Detection Distribution")
        chart_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📜 Recent MySQL Database Logs")
    table_controls_col1, table_controls_col2, table_controls_col3 = st.columns([2, 1, 1])

    with table_controls_col1:
        log_class_filter = st.selectbox("Filter Logs by Object Class", ["All"] + available_classes, index=0)
    with table_controls_col2:
        refresh_logs_btn = st.button("🔄 Refresh Logs Table", use_container_width=True)
    with table_controls_col3:
        clear_logs_btn = st.button("🗑️ Clear All Logs", use_container_width=True)

    if clear_logs_btn:
        if database.clear_all_logs():
            st.success("All logs cleared successfully.")
            st.rerun()

    logs_table_placeholder = st.empty()

    # Load initial table
    recent_df = database.fetch_recent_logs(limit=25, object_class=log_class_filter)
    logs_table_placeholder.dataframe(recent_df, use_container_width=True)

    # ----------------- STREAM PROCESSING LOOP -----------------
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    if start_btn:
        st.session_state.streaming = True
    if stop_btn:
        st.session_state.streaming = False

    if st.session_state.streaming:
        # Open video source
        if video_source_type == "Webcam":
            cap = cv2.VideoCapture(camera_idx)
        elif video_source_type == "Upload Video File" and uploaded_file_path:
            cap = cv2.VideoCapture(uploaded_file_path)
        else:
            # Generate or use test stream
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error(f"Cannot access video source '{video_source_type}'. If using a webcam, check permissions in System Settings.")
            st.session_state.streaming = False
            return

        last_logged_time = {}
        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0

        try:
            while st.session_state.streaming:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream ended or frame could not be read.")
                    break

                frame_count += 1
                curr_time = time.time()
                if curr_time - fps_start_time >= 1.0:
                    fps = frame_count / (curr_time - fps_start_time)
                    frame_count = 0
                    fps_start_time = curr_time

                # Run YOLO detection
                annotated_frame, detections = detector.detect(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    selected_classes=selected_classes,
                    draw_boxes=True,
                )

                # Event logging with cooldown debounce
                if enable_db_logging and detections:
                    detections_to_log = []
                    for d in detections:
                        cls_name = d["object_class"]
                        last_t = last_logged_time.get(cls_name, 0)
                        if (curr_time - last_t) >= debounce_sec:
                            detections_to_log.append(d)
                            last_logged_time[cls_name] = curr_time

                    if detections_to_log:
                        database.log_detections_batch(detections_to_log)

                # Update live video feed (Convert BGR to RGB for Streamlit/PIL)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update KPI Metrics
                fps_metric.metric("Processing Speed", f"{fps:.1f} FPS")
                frame_objs_metric.metric("Objects in Frame", f"{len(detections)}")

                # Update live current detection breakdown
                if detections:
                    det_summary_df = pd.DataFrame([
                        {
                            "Class": d["object_class"],
                            "Confidence": f"{d['confidence']:.2f}",
                            "Bounding Box (X,Y,W,H)": f"({d['bbox_x']}, {d['bbox_y']}, {d['bbox_w']}, {d['bbox_h']})",
                        }
                        for d in detections
                    ])
                    current_dets_container.dataframe(det_summary_df, use_container_width=True)
                else:
                    current_dets_container.info("No target objects detected in the current frame.")

                # Update distribution chart periodically
                if frame_count % 10 == 0:
                    metrics = database.fetch_metrics()
                    db_total_metric.metric("Total DB Logs", f"{metrics.get('total_logs', 0)}")
                    avg_conf_metric.metric("Avg Logged Conf", f"{metrics.get('avg_confidence', 0.0):.2f}")
                    if metrics.get("class_counts"):
                        chart_df = pd.DataFrame(
                            list(metrics["class_counts"].items()),
                            columns=["Class", "Count"],
                        ).set_index("Class")
                        chart_placeholder.bar_chart(chart_df)

                    # Update logs table
                    refreshed_df = database.fetch_recent_logs(limit=25, object_class=log_class_filter)
                    logs_table_placeholder.dataframe(refreshed_df, use_container_width=True)

                time.sleep(0.01)

        finally:
            cap.release()
            st.session_state.streaming = False


if __name__ == "__main__":
    main()
