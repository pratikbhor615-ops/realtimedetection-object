"""
Object Detection Module using Ultralytics YOLO and OpenCV.
Extracts object classes, coordinates, and confidence scores,
with real-time bounding box annotations and class filtering.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


class ObjectDetector:
    """
    YOLOv8 based object detector with customizable confidence thresholds
    and class filtering.
    """

    # High-contrast distinctive color palette for bounding boxes (BGR format)
    PALETTE = [
        (0, 255, 127),    # Spring Green
        (255, 105, 180),  # Hot Pink
        (30, 144, 255),   # Dodger Blue
        (255, 165, 0),    # Orange
        (186, 85, 211),   # Medium Orchid
        (0, 255, 255),    # Yellow/Cyan
        (255, 69, 0),     # Red-Orange
        (50, 205, 50),    # Lime Green
        (0, 191, 255),    # Deep Sky Blue
        (238, 130, 238),  # Violet
    ]

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Load the Ultralytics YOLO model.
        """
        self.model_name = model_name
        self.model = None
        self.class_names: Dict[int, str] = {}
        self._load_model()

    def _load_model(self):
        """Initializes the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
        except Exception as e:
            print(f"[Detector] Warning: Could not initialize YOLO model '{self.model_name}': {e}")
            # Fallback to standard COCO 80 classes dictionary
            self._init_default_coco_classes()

    def _init_default_coco_classes(self):
        """Standard 80 COCO classes for YOLOv8."""
        coco_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        self.class_names = {i: name for i, name in enumerate(coco_names)}

    def get_available_classes(self) -> List[str]:
        """Returns sorted list of available class names."""
        if not self.class_names:
            self._init_default_coco_classes()
        return sorted(list(self.class_names.values()))

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        selected_classes: Optional[List[str]] = None,
        draw_boxes: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run YOLO detection on a frame (BGR numpy image).

        Args:
            frame: OpenCV BGR image array.
            conf_threshold: Minimum confidence score (0.0 to 1.0).
            selected_classes: List of object class names to filter (None or empty means all).
            draw_boxes: Whether to draw bounding boxes and labels on the returned frame.

        Returns:
            Tuple of (annotated_frame, list of detection dictionaries).
            Each detection dict contains:
                'object_class': str
                'confidence': float
                'bbox_x': int
                'bbox_y': int
                'bbox_w': int
                'bbox_h': int
        """
        annotated_frame = frame.copy()
        detections: List[Dict[str, Any]] = []

        if frame is None or frame.size == 0:
            return annotated_frame, detections

        # If model is not loaded yet, try to load it
        if self.model is None:
            self._load_model()
            if self.model is None:
                return annotated_frame, detections

        try:
            # Run inference
            results = self.model(annotated_frame, conf=conf_threshold, verbose=False)
            if not results:
                return annotated_frame, detections

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return annotated_frame, detections

            for box in boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # Confidence threshold check
                if confidence < conf_threshold:
                    continue

                # Class filter check
                if selected_classes and len(selected_classes) > 0:
                    if class_name not in selected_classes and "All" not in selected_classes:
                        continue

                # Calculate standard bbox format (x, y, w, h)
                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)

                det_record = {
                    "object_class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox_x": x1,
                    "bbox_y": y1,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                }
                detections.append(det_record)

                if draw_boxes:
                    # Pick color based on class ID
                    color = self.PALETTE[class_id % len(self.PALETTE)]

                    # Draw bounding box rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Label badge
                    label = f"{class_name} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 1

                    # Get text size
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    # Draw filled badge background
                    badge_y1 = max(0, y1 - text_h - 8)
                    badge_y2 = y1
                    cv2.rectangle(
                        annotated_frame,
                        (x1, badge_y1),
                        (x1 + text_w + 10, badge_y2),
                        color,
                        -1,
                    )

                    # Draw text in white / black for contrast
                    text_color = (0, 0, 0)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 5, y1 - 4),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA,
                    )

        except Exception as e:
            print(f"[Detector] Inference error: {e}")

        return annotated_frame, detections
