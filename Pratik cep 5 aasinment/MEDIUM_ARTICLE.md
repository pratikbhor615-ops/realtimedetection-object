# Building a Real-Time Object Detection & Logging Platform with YOLOv8, OpenCV, and MySQL

*How I built an end-to-end computer vision platform with dynamic class filtering, live Streamlit monitoring, and automated database event logging.*

---

## 1. Introduction

Computer vision has rapidly evolved from academic research into production systems powering autonomous vehicles, retail intelligence, workplace safety, and smart security. However, detecting objects in a live camera stream is only half the battle. In real-world enterprise applications, detecting an object without logging its metadata—such as what was detected, when it appeared, its confidence level, and its exact spatial coordinates—leaves systems blind to analytics and auditing.

For **Assignment 5**, I set out to bridge this gap by building a **Real-Time Object Detection & Event Logging Platform**. The system captures live video feeds via OpenCV, detects objects using Ultralytics YOLOv8, logs every detection event into a local MySQL database in real time, and exposes a modern, interactive dashboard built with Streamlit.

In this article, I share my end-to-end journey: what I learned, how I architected the modular software stack, the challenges I tackled, and the final results.

---

## 2. Computer Vision & YOLO Learning

### Mastering OpenCV Basics
My journey began with mastering OpenCV (`cv2`) in Python. OpenCV provides the fundamental primitives for computer vision:
- **Video Capture Streams**: Interfacing with hardware webcams and local video files using `cv2.VideoCapture()`.
- **Color Spaces**: Understanding the distinction between OpenCV's native BGR format and the standard RGB color space required by UI toolkits and PIL.
- **Real-Time Frame Manipulation**: Rendering dynamic bounding boxes (`cv2.rectangle`), drawing high-contrast text badges (`cv2.putText`), and calculating frame processing rates (FPS).

### Leveraging Ultralytics YOLOv8
Next, I explored **Ultralytics YOLO (You Only Look Once)**, specifically the compact and blazingly fast `yolov8n.pt` (nano) model:
- **Single-Shot Detection**: Unlike older multi-stage algorithms (such as R-CNN), YOLO passes the entire image through a convolutional neural network once, simultaneously predicting bounding box coordinates and class probabilities.
- **Data Extraction**: Extracting `xyxy` box coordinates, class IDs, class names from the 80 COCO categories, and confidence scores ranging from `0.00` to `1.00`.
- **Confidence Thresholding**: Filtering out low-confidence hallucinations to maintain high precision.

---

## 3. Application Development & Modular Architecture

To avoid monolithic "spaghetti code", I designed the system around strict modular boundaries:

```text
├── main.py              # Application launcher (Streamlit UI, OpenCV GUI, CLI self-test)
├── detector.py          # YOLO model wrapper, bounding box logic, and class filtering
├── database.py          # MySQL connection, schema initialization, and query engine
├── ui.py                # Streamlit live streaming dashboard and metrics display
├── database_schema.sql  # SQL DDL script defining tables and indices
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables for secure database credentials
```

### Decoupled Components
1. **`detector.py`**: Encapsulates model loading and inference. It takes an OpenCV BGR frame, confidence threshold, and a list of target classes, returning annotated frames and a clean list of structured detection records:
   ```python
   {
       "object_class": "person",
       "confidence": 0.92,
       "bbox_x": 120, "bbox_y": 80,
       "bbox_w": 240, "bbox_h": 350
   }
   ```
2. **`database.py`**: Manages all database interactions using `mysql-connector-python`. It provides helper functions for connection pooling, single inserts, batch inserts, log retrieval, and analytical metric aggregation.
3. **`ui.py`**: The presentation layer. It displays live webcam video, live metric cards, distribution charts, and real-time database tables.

---

## 4. Database Integration & Event Logging

### Designing the Schema
The assignment specified logging each detection event with coordinates and confidence. I created the `vision_platform` database and `detection_logs` table:

```sql
CREATE DATABASE IF NOT EXISTS vision_platform;
USE vision_platform;

CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
);
```

### Solving Database Flooding with Rate-Limiting (Debounce)
At 30 frames per second, detecting a person standing in front of the camera would trigger **1,800 database write queries every minute**! This wastes storage and degrades performance.

To solve this, I implemented an **event debouncing cooldown mechanism**:
- A dictionary tracks the timestamp of the last logged detection for each unique object class.
- The system only writes a new record to MySQL if the time elapsed since the last log for that class exceeds a user-configurable cooldown interval (e.g., 1.0 second).
- Batch inserts (`cursor.executemany`) are used to commit all frame detections in a single database transaction.

### Security First
I strictly avoided hardcoding database passwords. All credentials (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`) are read from a local `.env` file via `python-dotenv`, and `.env` is safeguarded in `.gitignore`.

---

## 5. User Interface with Streamlit

I chose **Streamlit** to build a modern, browser-based monitoring dashboard:
- **Sidebar Controls**:
  - Confidence Threshold Slider (`0.10` to `1.00`)
  - Target Object Class Multi-Select (`person`, `cell phone`, `laptop`, `bottle`, etc.)
  - Video Source Selector (Webcam / Uploaded Video / Test Feed)
  - Real-time logging toggle and cooldown frequency slider
- **Main Dashboard**:
  - Live metric KPI cards: Processing FPS, Objects in Current Frame, Total MySQL Logs Stored, and Average Confidence.
  - Live video stream displaying real-time bounding boxes and confidence badges.
  - Breakdown table of detections in the active frame.
  - Interactive table displaying the 25 most recent MySQL records with filtering, sorting, and manual refresh controls.

---

## 6. Challenges & Solutions

| Challenge | Root Cause | Solution |
| :--- | :--- | :--- |
| **High Video Latency** | Sequential database write operations blocking the video frame processing loop. | Implemented batch insertions and a cooldown debounce timer so DB calls do not block the video thread. |
| **Streamlit Rerun Flickering** | Streamlit re-executing the entire script on each interaction. | Used `@st.cache_resource` for the YOLO model weights and managed video streams with efficient placeholder updates. |
| **Color Distortion** | OpenCV outputs BGR format, but web browsers and Streamlit expect RGB. | Explicitly converted frames using `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before rendering. |
| **Credential Security** | Risk of accidentally committing MySQL root passwords to GitHub. | Isolated configuration into `.env` and `.env.example`, protected by `.gitignore`. |
| **Offline Development** | MySQL daemon stopped during testing outside local server environment. | Added an automatic SQLite fallback layer so tests and demonstrations run without crashing if the MySQL server is offline. |

---

## 7. Final Results & Deliverables

The resulting platform is fast, responsive, and robust. It seamlessly runs YOLOv8 detection at real-time speeds, dynamically logs structured records to MySQL, and displays an intuitive dashboard.

### Submission Links
- **GitHub Repository**: [object-detection-system](https://github.com/pratikbhor615-ops/object-detection-system)
- **Demonstration Video**: [Watch Demo on YouTube / Google Drive](https://youtu.be/YOUR_DEMO_VIDEO)
- **Medium Article**: [Published on Medium](https://medium.com/@YOUR_HANDLE/building-a-real-time-object-detection-and-logging-platform-with-yolo-opencv-and-mysql)

---

## Key Takeaways
Building this project solidified my understanding of the complete computer vision pipeline—from low-level frame manipulation with OpenCV to deep learning inference with YOLO, relational database engineering with MySQL, and responsive dashboard development with Streamlit.
