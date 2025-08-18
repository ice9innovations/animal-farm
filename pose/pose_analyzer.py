#!/usr/bin/env python3
"""
MediaPipe Pose Analyzer - Core pose estimation functionality

Handles pose landmark detection, classification, and advanced pose analysis
using MediaPipe Pose with enhanced features for the standalone pose service.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """Core MediaPipe pose analysis with enhanced pose classification and analysis"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 2,
                 enable_segmentation: bool = True):
        """Initialize MediaPipe pose analyzer"""
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose model with enhanced settings
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,  # 2 for best accuracy
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark names for better readability
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        logger.info("✅ PoseAnalyzer initialized with MediaPipe Pose")
    
    def analyze_pose(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze pose from image file
        
        Returns:
            Dict containing pose predictions with enhanced analysis
        """
        try:
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose_model.process(image_rgb)
            
            predictions = []
            
            if results.pose_landmarks:
                # Extract and format landmarks
                landmarks = self._extract_landmarks(results.pose_landmarks)
                
                # Advanced pose analysis
                pose_analysis = self._analyze_pose_features(landmarks)
                
                # Build prediction
                prediction = {
                    "person_id": 1,  # Single person for now
                    "landmarks": landmarks,
                    "pose_analysis": pose_analysis
                }
                
                predictions.append(prediction)
            
            return {
                "predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"Pose analysis error: {str(e)}")
            raise
    
    def _extract_landmarks(self, pose_landmarks) -> Dict[str, Dict[str, float]]:
        """Extract landmarks with names and enhanced precision"""
        landmarks = {}
        
        for i, landmark in enumerate(pose_landmarks.landmark):
            landmark_name = self.landmark_names[i] if i < len(self.landmark_names) else f"landmark_{i}"
            
            landmarks[landmark_name] = {
                "x": round(landmark.x, 3),
                "y": round(landmark.y, 3), 
                "z": round(landmark.z, 3),
                "visibility": round(getattr(landmark, 'visibility', 1.0), 3)
            }
        
        return landmarks
    
    
    def _analyze_pose_features(self, landmarks: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Joint angle analysis for pose reconstruction"""
        try:
            # Joint angles - useful for pose reconstruction
            joint_angles = self._calculate_joint_angles(landmarks)
            
            return {
                "joint_angles": joint_angles
            }
            
        except Exception as e:
            logger.warning(f"Joint angle analysis error: {e}")
            return {
                "joint_angles": {}
            }
    
    
    def _calculate_joint_angles(self, landmarks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate joint angles in degrees"""
        angles = {}
        
        try:
            # Left elbow angle
            if all(point in landmarks for point in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angles['left_elbow'] = self._angle_between_points(
                    landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
                )
            
            # Right elbow angle  
            if all(point in landmarks for point in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angles['right_elbow'] = self._angle_between_points(
                    landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
                )
            
            # Left knee angle
            if all(point in landmarks for point in ['left_hip', 'left_knee', 'left_ankle']):
                angles['left_knee'] = self._angle_between_points(
                    landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle']
                )
            
            # Right knee angle
            if all(point in landmarks for point in ['right_hip', 'right_knee', 'right_ankle']):
                angles['right_knee'] = self._angle_between_points(
                    landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle']
                )
                
        except Exception as e:
            logger.warning(f"Joint angle calculation error: {e}")
        
        return {k: round(v, 1) for k, v in angles.items()}
    
    def _angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points (p2 is the vertex)"""
        try:
            # Vectors from p2 to p1 and p2 to p3
            v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'])
            v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
            
            # Calculate angle using dot product
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            
            angle_radians = math.acos(cos_angle)
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees
            
        except Exception:
            return 0.0
    
    
    
    
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose_model'):
            self.pose_model.close()