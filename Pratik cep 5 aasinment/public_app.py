import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer

import database
from detector import ObjectDetector

st.set_page_config(
    page_title="Object Detection Web App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_detector():
    return ObjectDetector(model_name="yolov8n.pt")


class DetectionTransformer:
    def __init__(self, detector, conf_threshold=0.5, selected_classes=None):
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.selected_classes = selected_classes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, detections = self.detector.detect(
            frame=img,
            conf_threshold=self.conf_threshold,
            selected_classes=self.selected_classes,
            draw_boxes=True,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def main():
    detector = load_detector()
    available_classes = detector.get_available_classes()

    st.title("🎯 Real-Time Object Detection Web App")
    st.caption("Browser webcam access works from the website itself, so it does not rely on a local laptop camera session.")

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("Confidence threshold", min_value=0.10, max_value=1.00, value=0.50, step=0.05)
        selected_classes = st.multiselect(
            "Classes to detect",
            options=available_classes,
            default=[c for c in ["person", "cell phone", "laptop", "bottle"] if c in available_classes],
        )
        st.markdown("---")
        st.subheader("Usage")
        st.write("1. Allow browser camera access when prompted.")
        st.write("2. Keep the page open while streaming.")
        st.write("3. Use a sample video upload if the camera is unavailable.")

    source = st.radio("Input source", ["Browser Camera", "Upload Video File"], index=0)

    if source == "Upload Video File":
        uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"], accept_multiple_files=False)
        if uploaded is not None:
            video_path = f"/tmp/{uploaded.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Unable to open the uploaded video file.")
                return

            st.subheader("Preview")
            frame_placeholder = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, detections = detector.detect(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    selected_classes=selected_classes or None,
                    draw_boxes=True,
                )
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                if st.button("Stop Video", key="stop_upload_video"):
                    break
            cap.release()
            st.session_state.pop("stop_upload_video", None)
        else:
            st.info("Upload a video file to test the detector without browser camera access.")
    else:
        camera_ok = st.checkbox("Enable browser webcam", value=True)
        if camera_ok:
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            webrtc_streamer(
                key="browser-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: DetectionTransformer(
                    detector,
                    conf_threshold=conf_threshold,
                    selected_classes=selected_classes or None,
                ),
                async_processing=True,
            )
        else:
            st.info("Browser webcam is disabled. Please enable it to start streaming.")

    st.markdown("---")
    st.subheader("Recent log snapshot")
    recent = database.fetch_recent_logs(limit=8)
    st.dataframe(recent, use_container_width=True)


if __name__ == "__main__":
    main()
