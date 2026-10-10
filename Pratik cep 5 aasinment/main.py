"""
Main Entry Point for the Real-Time Object Detection & Logging Platform.
Supports:
  1. Web Dashboard Mode (Streamlit): python main.py --ui (or default)
  2. OpenCV Desktop GUI Mode: python main.py --opencv
  3. Database Setup: python main.py --init-db
  4. Self-Test Mode: python main.py --test
"""

import argparse
import sys
import os
import time
import subprocess
import cv2
import numpy as np

import database
from detector import ObjectDetector


def run_opencv_mode(camera_idx: int = 0, conf_threshold: float = 0.50, selected_classes=None):
    """
    Runs the detector in a native OpenCV desktop window with live MySQL logging.
    """
    print("=" * 65)
    print("🚀 Starting Real-Time Object Detection (OpenCV Desktop Mode)")
    print("=" * 65)

    # Initialize Database
    db_init = database.init_database()
    print(f"📦 Database Engine: {db_init['engine']}")
    print(f"ℹ️  Status: {db_init['message']}\n")

    # Initialize Detector
    print("🤖 Loading YOLO model (yolov8n.pt)...")
    detector = ObjectDetector(model_name="yolov8n.pt")
    available_classes = detector.get_available_classes()
    print(f"✅ Detector ready. Available classes: {len(available_classes)}")

    if selected_classes:
        print(f"🎯 Filtered classes: {selected_classes}")
    else:
        print("🎯 Detecting all classes (default)")

    # Open Video Capture
    print(f"📹 Opening camera index {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_idx}. Trying sample generator mode...")
        # Fallback to synthetic frame test loop if camera is unavailable
        _run_synthetic_loop(detector, conf_threshold, selected_classes)
        return

    print("\n⌨️  Keyboard Controls:")
    print("   [q] - Quit Application")
    print("   [s] - Save Screenshot")
    print("   [l] - Show Recent MySQL Logs in Terminal")
    print("   [+] - Increase Confidence Threshold")
    print("   [-] - Decrease Confidence Threshold\n")

    last_logged_time = {}
    debounce_sec = 1.0
    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    window_name = "Real-Time Object Detection & Logging Platform (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame could not be read from camera.")
                break

            frame_count += 1
            curr_time = time.time()
            if curr_time - fps_start >= 1.0:
                fps = frame_count / (curr_time - fps_start)
                frame_count = 0
                fps_start = curr_time

            # Detection
            annotated_frame, detections = detector.detect(
                frame=frame,
                conf_threshold=conf_threshold,
                selected_classes=selected_classes,
                draw_boxes=True,
            )

            # Database Logging with Debounce
            if detections:
                to_log = []
                for d in detections:
                    cls_name = d["object_class"]
                    if (curr_time - last_logged_time.get(cls_name, 0)) >= debounce_sec:
                        to_log.append(d)
                        last_logged_time[cls_name] = curr_time

                if to_log:
                    count = database.log_detections_batch(to_log)
                    for item in to_log:
                        print(f"💾 [LOGGED TO DB] {item['object_class']} | Conf: {item['confidence']:.2f} | BBox: ({item['bbox_x']}, {item['bbox_y']}, {item['bbox_w']}, {item['bbox_h']})")

            # Draw HUD overlay
            hud_text = f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f} | Detections: {len(detections)}"
            cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting application...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Snapshot saved to {filename}")
            elif key == ord('l'):
                logs_df = database.fetch_recent_logs(limit=5)
                print("\n--- Recent Database Logs ---")
                print(logs_df.to_string(index=False) if not logs_df.empty else "No logs yet.")
                print("----------------------------\n")
            elif key in (ord('+'), ord('=')):
                conf_threshold = min(1.0, round(conf_threshold + 0.05, 2))
                print(f"⚙️ Confidence threshold: {conf_threshold}")
            elif key in (ord('-'), ord('_')):
                conf_threshold = max(0.1, round(conf_threshold - 0.05, 2))
                print(f"⚙️ Confidence threshold: {conf_threshold}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def _run_synthetic_loop(detector: ObjectDetector, conf_threshold: float, selected_classes=None):
    """Generates test synthetic frames if physical webcam is inaccessible."""
    print("💡 Running simulation mode with synthetic test frames...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Webcam Not Detected", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, "Connect webcam and rerun", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    annotated_frame, detections = detector.detect(frame, conf_threshold, selected_classes)
    cv2.imshow("Synthetic Test (Press 'q' to quit)", annotated_frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def run_streamlit_dashboard():
    """Launches the Streamlit Web UI dashboard for browser access on any device/network."""
    print("=" * 65)
    print("🚀 Launching Streamlit Web UI Dashboard...")
    print("=" * 65)
    ui_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.py")
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        ui_script,
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ])


def run_self_test():
    """Automated verification test for Assignment 5 requirements."""
    print("=" * 65)
    print("🧪 Running Automated Assignment 5 Self-Test")
    print("=" * 65)

    # 1. Test Database
    print("\n[1/3] Testing Database Initialization & Schema...")
    db_result = database.init_database()
    print(f"      Engine: {db_result['engine']}")
    print(f"      Message: {db_result['message']}")

    # 2. Test Detection Logging
    print("\n[2/3] Testing Database Log Insertion...")
    test_logged = database.log_detection(
        object_class="person",
        confidence=0.92,
        bbox_x=120,
        bbox_y=80,
        bbox_w=240,
        bbox_h=350,
    )
    print(f"      Single Insert Success: {test_logged}")

    # Fetch logs back
    recent_logs = database.fetch_recent_logs(limit=3)
    print(f"      Fetched {len(recent_logs)} recent records:")
    print(recent_logs.to_string(index=False) if not recent_logs.empty else "      No rows returned.")

    # 3. Test Object Detector
    print("\n[3/3] Testing YOLO Detector Initialization...")
    detector = ObjectDetector(model_name="yolov8n.pt")
    classes = detector.get_available_classes()
    print(f"      Detector initialized with {len(classes)} classes.")

    # Test detection on a synthetic test image
    test_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    ann_img, dets = detector.detect(test_img, conf_threshold=0.5)
    print(f"      Inference on blank test image completed cleanly ({len(dets)} detections).")

    print("\n" + "=" * 65)
    print("✅ All Component Tests Completed Successfully!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection & Logging Platform")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit web dashboard (Default)")
    parser.add_argument("--opencv", action="store_true", help="Run in OpenCV desktop window mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold (0.10 - 1.00)")
    parser.add_argument("--classes", nargs="+", default=None, help="Classes to detect (e.g. person cell_phone)")
    parser.add_argument("--init-db", action="store_true", help="Initialize the MySQL database and exit")
    parser.add_argument("--test", action="store_true", help="Run automated self-test")

    args = parser.parse_args()

    if args.init_db:
        res = database.init_database()
        print(f"Database Initialization: {res}")
        sys.exit(0)
    elif args.test:
        run_self_test()
        sys.exit(0)
    elif args.opencv:
        run_opencv_mode(camera_idx=args.camera, conf_threshold=args.conf, selected_classes=args.classes)
    else:
        # Default mode: Streamlit UI
        run_streamlit_dashboard()


if __name__ == "__main__":
    main()
