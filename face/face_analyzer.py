#!/usr/bin/env python3
"""
MediaPipe Face Analyzer - Core face detection functionality

Handles face detection, keypoint extraction, and analysis
using MediaPipe Face Detection with enhanced features for the standalone face service.
"""

import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Core MediaPipe face detection with enhanced keypoint analysis"""

    def __init__(self,
                 min_detection_confidence: float = 0.2,
                 model_selection: int = 1,
                 use_gpu: bool = True):
        """Initialize MediaPipe face analyzer"""

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.use_gpu = use_gpu

        # Initialize face detection model with enhanced settings
        # MediaPipe automatically uses GPU when available unless explicitly disabled
        self.face_detection_model = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,  # 1 for full range detection (better for diverse faces)
            min_detection_confidence=min_detection_confidence
        )

        gpu_status = "GPU" if use_gpu else "CPU"
        logger.info(f"✅ FaceAnalyzer initialized with MediaPipe Face Detection ({gpu_status})")

    def analyze_faces_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze faces from PIL Image (in-memory processing)

        Args:
            image: PIL Image object

        Returns:
            Dict containing face detection results
        """
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)

            # MediaPipe expects RGB format (PIL Images are already RGB)
            height, width, _ = img_array.shape

            # Process with MediaPipe face detection
            results = self.face_detection_model.process(img_array)

            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box

                    # Convert to pixel coordinates
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    # Get confidence
                    confidence = detection.score[0] if detection.score else 0.0

                    # Get key points if available
                    keypoints = {}
                    if detection.location_data.relative_keypoints:
                        keypoints = {
                            'right_eye': [
                                int(detection.location_data.relative_keypoints[0].x * width),
                                int(detection.location_data.relative_keypoints[0].y * height)
                            ],
                            'left_eye': [
                                int(detection.location_data.relative_keypoints[1].x * width),
                                int(detection.location_data.relative_keypoints[1].y * height)
                            ],
                            'nose_tip': [
                                int(detection.location_data.relative_keypoints[2].x * width),
                                int(detection.location_data.relative_keypoints[2].y * height)
                            ],
                            'mouth_center': [
                                int(detection.location_data.relative_keypoints[3].x * width),
                                int(detection.location_data.relative_keypoints[3].y * height)
                            ],
                            'right_ear_tragion': [
                                int(detection.location_data.relative_keypoints[4].x * width),
                                int(detection.location_data.relative_keypoints[4].y * height)
                            ],
                            'left_ear_tragion': [
                                int(detection.location_data.relative_keypoints[5].x * width),
                                int(detection.location_data.relative_keypoints[5].y * height)
                            ]
                        }

                    faces.append({
                        'bbox': [x, y, w, h],
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'method': 'mediapipe'
                    })

            return {
                'faces': faces,
                'dimensions': {'width': width, 'height': height}
            }

        except Exception as e:
            logger.error(f"MediaPipe face detection error: {str(e)}")
            return {
                'faces': [],
                'dimensions': {'width': 0, 'height': 0}
            }

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection_model'):
            self.face_detection_model.close()