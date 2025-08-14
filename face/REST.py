#!/usr/bin/env python3
"""
Human Analysis Service using MediaPipe
Provides face detection, pose estimation, and facial expression analysis
Replaces the outdated and biased Haar Cascade/SSD models with Google's MediaPipe framework
"""

import os
import io
import time
import logging
import tempfile
import uuid
import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import requests
from datetime import datetime
from urllib.parse import urlparse

# Load environment variables
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')
PRIVATE_STR = os.getenv('PRIVATE')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT_STR:
    raise ValueError("API_TIMEOUT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR)
PRIVATE_MODE = PRIVATE_STR.lower() == 'true'

# Global emoji mappings - loaded from API on startup
emoji_mappings = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global MediaPipe models - initialize once at startup
face_detection_model = None
pose_model = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# MediaPipe model configuration
FACE_MIN_DETECTION_CONFIDENCE = 0.2
POSE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.1
POSE_MIN_TRACKING_CONFIDENCE = 0.5
CONFIDENCE_DECIMAL_PLACES = 3
LANDMARK_DECIMAL_PLACES = 3  # Precision for landmark coordinates and visibility
MIN_FACE_SIZE_FOR_MESH = 60  # Minimum face size (pixels) for reliable Face Mesh analysis

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_models():
    """Initialize MediaPipe models once at startup"""
    global face_detection_model, pose_model
    try:
        logger.info("Initializing MediaPipe models...")
        
        # Face Detection
        face_detection_model = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection (better for diverse faces)
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE
        )
        logger.info("✅ MediaPipe Face Detection model initialized")
        
        # Pose Detection
        pose_model = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=True,  # Enable person silhouette extraction
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
        )
        logger.info("✅ MediaPipe Pose model initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing MediaPipe models: {str(e)}")
        return False

def load_emoji_mappings():
    """Load emoji mappings from central API"""
    global emoji_mappings
    try:
        emoji_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
        response = requests.get(emoji_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        emoji_mappings = response.json()
        logger.info(f"✅ Loaded {len(emoji_mappings)} emoji mappings from {emoji_url}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load emoji mappings: {e}. Using empty mappings.")
        emoji_mappings = {}

def get_emoji(word):
    """Get emoji for a given word"""
    return emoji_mappings.get(word.lower(), "")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image(url):
    """Download image from URL and return as bytes"""
    try:
        headers = {'User-Agent': 'MediaPipe Face Analysis Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        return io.BytesIO(response.content)
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def convert_to_jpg(image_bytes):
    """Convert any image format to JPG for consistent processing"""
    try:
        image = Image.open(image_bytes)
        
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG
        jpg_bytes = io.BytesIO()
        image.save(jpg_bytes, format='JPEG', quality=85)
        jpg_bytes.seek(0)
        
        return jpg_bytes
        
    except Exception as e:
        raise Exception(f"Failed to convert image: {str(e)}")

def detect_faces_mediapipe(image_path):
    """Detect faces using MediaPipe"""
    global face_detection_model
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Could not load image from {image_path}")
        
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_detection_model.process(image_rgb)
        
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
        
        return faces, {'width': width, 'height': height}
        
    except Exception as e:
        logger.error(f"MediaPipe face detection error: {str(e)}")
        return [], {'width': 0, 'height': 0}

def detect_poses_mediapipe(image_path):
    """Detect poses using MediaPipe"""
    global pose_model
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return [], {'width': 0, 'height': 0}
        
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose_model.process(image_rgb)
        
        poses = []
        segmentation_data = None
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': round(landmark.x, LANDMARK_DECIMAL_PLACES),
                    'y': round(landmark.y, LANDMARK_DECIMAL_PLACES),
                    'z': round(landmark.z, LANDMARK_DECIMAL_PLACES),
                    'visibility': round(landmark.visibility if hasattr(landmark, 'visibility') else 1.0, LANDMARK_DECIMAL_PLACES)
                })
            
            # Classify pose
            pose_category = classify_pose(landmarks)
            
            # Get overall confidence (average visibility of key landmarks)
            key_landmarks = landmarks[:15]  # Focus on upper body for confidence
            confidence = sum(lm['visibility'] for lm in key_landmarks) / len(key_landmarks)
            
            pose_data = {
                'landmarks': landmarks,
                'pose_category': pose_category,
                'confidence': confidence,
                'method': 'mediapipe'
            }
            
            # Add segmentation data if available
            if results.segmentation_mask is not None:
                # Convert segmentation mask to base64 for JSON serialization
                segmentation_mask = (results.segmentation_mask > 0.1).astype(np.uint8) * 255
                segmentation_data = {
                    'width': width,
                    'height': height,
                    'has_segmentation': True
                }
                pose_data['segmentation'] = segmentation_data
            
            poses.append(pose_data)
        
        return poses, {'width': width, 'height': height, 'segmentation': segmentation_data}
        
    except Exception as e:
        logger.error(f"MediaPipe pose detection error: {str(e)}")
        return [], {'width': 0, 'height': 0}

def classify_pose(landmarks):
    """Classify pose based on landmark positions"""
    try:
        # Simple pose classification based on key landmark relationships
        if len(landmarks) < 16:
            return 'unknown'
        
        # Get key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25] if len(landmarks) > 25 else None
        right_knee = landmarks[26] if len(landmarks) > 26 else None
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        
        # Calculate hip-to-shoulder ratio
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        
        # Basic pose classification
        if abs(shoulder_y - hip_y) < 0.1:  # Very close together
            return 'lying'
        elif shoulder_y > hip_y:  # Shoulders below hips (unusual)
            return 'inverted'
        elif left_knee and right_knee:
            knee_y = (left_knee['y'] + right_knee['y']) / 2
            if knee_y > hip_y + 0.2:  # Knees significantly below hips
                return 'standing'
            elif abs(knee_y - hip_y) < 0.1:  # Knees close to hip level
                return 'sitting'
        
        return 'standing'  # Default
        
    except Exception as e:
        logger.warning(f"Pose classification error: {e}")
        return 'unknown'

def process_image(image_source, is_url=False, is_file_path=False):
    """Process image and perform comprehensive human analysis - returns raw data"""
    start_time = time.time()
    temp_path = None
    
    try:
        # Handle image source
        if is_url:
            image_bytes = download_image(image_source)
        elif is_file_path:
            # Direct file path - use directly without temporary conversion
            faces, dimensions = detect_faces_mediapipe(image_source)
            poses, _ = detect_poses_mediapipe(image_source)
            processing_time = time.time() - start_time
            return {
                'faces': faces,
                'poses': poses,
                'expressions': [],  # No longer using Face Mesh
                'dimensions': dimensions,
                'processing_time': processing_time,
                'error': None
            }
        else:
            image_bytes = image_source
        
        # Convert to JPG for consistent processing
        jpg_bytes = convert_to_jpg(image_bytes)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            jpg_bytes.seek(0)
            tmp_file.write(jpg_bytes.read())
            temp_path = tmp_file.name
        
        # Perform comprehensive analysis
        faces, dimensions = detect_faces_mediapipe(temp_path)
        poses, _ = detect_poses_mediapipe(temp_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'faces': faces,
            'poses': poses,
            'expressions': [],  # No longer using Face Mesh
            'dimensions': dimensions,
            'processing_time': processing_time,
            'error': None
        }
    
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        processing_time = time.time() - start_time
        return {
            'faces': [],
            'poses': [],
            'expressions': [],
            'dimensions': {'width': 0, 'height': 0},
            'processing_time': processing_time,
            'error': str(e)
        }
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global face_detection_model, pose_model
    try:
        # Check if models are initialized
        face_status = 'ready' if face_detection_model is not None else 'not_initialized'
        pose_status = 'ready' if pose_model is not None else 'not_initialized'
        
        all_ready = all([face_detection_model, pose_model])
        
        return jsonify({
            'status': 'healthy' if all_ready else 'degraded',
            'service': 'face',
            'capabilities': ['face_detection', 'pose_estimation', 'person_segmentation'],
            'models': {
                'face_detection': {
                    'status': face_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Face Detection (Full Range)',
                    'fairness': 'Tested across demographics'
                },
                'pose_estimation': {
                    'status': pose_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Pose',
                    'landmarks': 33
                }
            },
            'endpoints': [
                "GET /health - Health check",
                "GET /v3/analyze?url=<image_url> - Analyze image from URL", 
                "GET /v3/analyze?file=<file_path> - Analyze image from file",
                "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
                "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
            ],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    start_time = time.time()
    
    try:
        # Get parameters from query string
        url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input - exactly one parameter required
        if not url and not file_path:
            return jsonify({
                "service": "face",
                "status": "error", 
                "predictions": [],
                "error": {"message": "Must provide either 'url' or 'file' parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "MediaPipe"
                    }
                }
            }), 400
        
        if url and file_path:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "MediaPipe"
                    }
                }
            }), 400
        
        # Handle URL input
        if url:
            raw_data = process_image(url, is_url=True)
        
        # Handle file path input
        elif file_path:
            # Validate file path
            if not os.path.exists(file_path):
                return jsonify({
                    "service": "face",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file_path}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "MediaPipe"
                        }
                    }
                }), 404
            
            if not allowed_file(file_path):
                return jsonify({
                    "service": "face",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "File type not allowed"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "MediaPipe"
                        }
                    }
                }), 400
            
            raw_data = process_image(file_path, is_file_path=True)
        
        # Handle errors
        if raw_data['error']:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "metadata": {
                    "processing_time": round(raw_data['processing_time'], 3),
                    "model_info": {
                        "framework": "MediaPipe"
                    }
                },
                "error": {
                    "message": str(raw_data['error'])
                }
            }), 500
        
        # Build V3 predictions
        predictions = []
        
        # Add face predictions
        for face in raw_data['faces']:
            predictions.append({
                "label": "face",
                "emoji": get_emoji("face"),
                "confidence": round(float(face['confidence']), CONFIDENCE_DECIMAL_PLACES),
                "bbox": face['bbox'],
                "properties": {
                    "keypoints": face.get('keypoints', {}),
                    "method": face['method']
                }
            })
        
        # Add pose predictions
        for pose in raw_data.get('poses', []):
            pose_prediction = {
                "label": pose['pose_category'],
                "emoji": get_emoji(pose['pose_category']),
                "confidence": round(float(pose['confidence']), CONFIDENCE_DECIMAL_PLACES),
                "properties": {
                    "landmarks": pose['landmarks'],
                    "method": pose['method']
                }
            }
            
            # Add segmentation data if available
            if 'segmentation' in pose and pose['segmentation']:
                pose_prediction["properties"]["segmentation"] = pose['segmentation']
            
            predictions.append(pose_prediction)
        
        return jsonify({
            "service": "face",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(raw_data['processing_time'], 3),
                "model_info": {
                    "framework": "MediaPipe"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in V3 analysis: {str(e)}")
        return jsonify({
            'service': 'face',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {
                    'framework': 'MediaPipe'
                }
            },
            'error': {
                'message': str(e)
            }
        }), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Call V3 with translated parameters
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # Let V3 handle validation errors
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

if __name__ == '__main__':
    # Initialize all MediaPipe models at startup
    logger.info("Initializing MediaPipe models...")
    if not initialize_models():
        logger.error("Failed to initialize MediaPipe models. Service will not function properly.")
        exit(1)
    
    # Load emoji mappings on startup
    load_emoji_mappings()
    
    # Determine host based on private mode (like other services)
    host = "127.0.0.1" if PRIVATE_MODE else "0.0.0.0"
    
    logger.info(f"Starting Face Analysis API on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE_MODE}")
    logger.info("Using MediaPipe framework for face detection, pose estimation, and facial expressions")
    logger.info("V3 API available at /v3/analyze endpoint with V2 backward compatibility")
    app.run(host=host, port=PORT, debug=False)
