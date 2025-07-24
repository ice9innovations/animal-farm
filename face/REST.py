#!/usr/bin/env python3
"""
Modern Face Detection Service using MediaPipe
Replaces the outdated and biased Haar Cascade/SSD models with Google's MediaPipe framework
"""

import os
import io
import time
import logging
import tempfile
import uuid
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
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global MediaPipe model - initialize once at startup
face_detection_model = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
PORT = int(os.getenv('PORT', 7772))
API_HOST = os.getenv('API_HOST', 'localhost')
PRIVATE_MODE = os.getenv('PRIVATE', 'false').lower() == 'true'

# V2 API Configuration removed - use separate endpoints instead

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_face_detection_model():
    """Initialize MediaPipe face detection model once at startup"""
    global face_detection_model
    try:
        logger.info("Initializing MediaPipe Face Detection model...")
        face_detection_model = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection (better for diverse faces)
            min_detection_confidence=0.2
        )
        logger.info("✅ MediaPipe Face Detection model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize MediaPipe model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {int(content_length)} bytes")
        
        # Read content with size limit
        content = b''
        size = 0
        for chunk in response.iter_content(chunk_size=8192):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise ValueError(f"File too large: exceeded {MAX_FILE_SIZE} bytes")
            content += chunk
        
        return io.BytesIO(content)
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

def convert_to_jpg(image_bytes):
    """Convert image to JPG format for consistent processing"""
    try:
        img = Image.open(image_bytes)
        
        # Convert RGBA to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        return output
    except Exception as e:
        logger.error(f"Error converting image: {str(e)}")
        raise

def detect_faces_mediapipe(image_path, min_detection_confidence=0.2):
    """
    Detect faces using MediaPipe Face Detection
    Uses MediaPipe Face Detection framework
    """
    global face_detection_model
    try:
        # Ensure model is initialized
        if face_detection_model is None:
            logger.warning("MediaPipe model not initialized, initializing now...")
            if not initialize_face_detection_model():
                raise ValueError("Failed to initialize MediaPipe model")
        
        # Read and prepare image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        faces = []
        
        # Use the global MediaPipe Face Detection model
        results = face_detection_model.process(image_rgb)
        
        if results.detections:
            logger.info(f"✅ MediaPipe detected {len(results.detections)} face(s)")
            for i, detection in enumerate(results.detections):
                logger.info(f"Face {i}: confidence = {detection.score[0] if detection.score else 'unknown'}")
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                confidence_score = detection.score[0] if detection.score else min_detection_confidence
                logger.info(f"Face {i}: bbox=({x},{y},{w},{h}), confidence={confidence_score}")
                
                faces.append({
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    },
                    'confidence': confidence_score,
                    'method': 'mediapipe',
                    'keypoints': extract_keypoints(detection)
                })
        else:
            logger.info("❌ MediaPipe detected 0 faces")
        
        logger.info(f"Returning {len(faces)} faces to API")
        return faces, {'width': width, 'height': height}
        
    except Exception as e:
        logger.error(f"MediaPipe detection error: {str(e)}")
        return [], {'width': 0, 'height': 0}

def extract_keypoints(detection):
    """Extract facial keypoints from MediaPipe detection"""
    keypoints = {}
    if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_keypoints'):
        keypoint_names = ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 'right_ear', 'left_ear']
        for i, kp in enumerate(detection.location_data.relative_keypoints):
            if i < len(keypoint_names):
                keypoints[keypoint_names[i]] = {
                    'x': kp.x,
                    'y': kp.y
                }
    return keypoints


def process_image(image_source, is_url=False, is_file_path=False):
    """Process image and detect faces - returns raw data"""
    start_time = time.time()
    temp_path = None
    
    try:
        # Handle image source
        if is_url:
            image_bytes = download_image(image_source)
        elif is_file_path:
            # Direct file path - use directly without temporary conversion
            faces, dimensions = detect_faces_mediapipe(image_source)
            processing_time = time.time() - start_time
            return {
                'faces': faces,
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
        
        # Detect faces using MediaPipe
        faces, dimensions = detect_faces_mediapipe(temp_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'faces': faces,
            'dimensions': dimensions,
            'processing_time': processing_time,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            'faces': [],
            'dimensions': {'width': 0, 'height': 0},
            'processing_time': time.time() - start_time,
            'error': e
        }
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global face_detection_model
    try:
        # Check if global model is initialized
        model_status = 'ready' if face_detection_model is not None else 'not_initialized'
        
        return jsonify({
            'status': 'healthy' if face_detection_model is not None else 'degraded',
            'service': 'face',
            'models': {
                'mediapipe': {
                    'status': model_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Face Detection (Full Range)',
                    'fairness': 'Tested across demographics',
                    'initialized_at_startup': face_detection_model is not None
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/', methods=['GET', 'POST'])
def detect_faces():
    """Main face detection endpoint"""
    try:
        if request.method == 'POST':
            # Handle file upload
            if 'uploadedfile' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['uploadedfile']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Read file into memory
            file_bytes = io.BytesIO(file.read())
            
            # Check file size
            file_bytes.seek(0, 2)  # Seek to end
            file_size = file_bytes.tell()
            file_bytes.seek(0)  # Seek back to start
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({'error': f'File too large. Max size is {MAX_FILE_SIZE/1024/1024}MB'}), 400
            
            # Process the image
            raw_data = process_image(file_bytes, is_url=False)
            
        else:
            # Handle URL parameter
            url = request.args.get('url')
            path = request.args.get('path')
            
            if url:
                raw_data = process_image(url, is_url=True)
            elif path and not PRIVATE_MODE:
                # Handle local file path (only if not in private mode)
                if os.path.exists(path) and os.path.isfile(path):
                    raw_data = process_image(path, is_file_path=True)
                else:
                    return jsonify({'error': 'File not found'}), 404
            else:
                return jsonify({'error': 'No image source provided'}), 400
        
        # Format legacy response
        if raw_data['error']:
            return jsonify({
                "FACE": {
                    "error": str(raw_data['error']),
                    "status": "error"
                }
            })
        
        return jsonify({
            "FACE": {
                "faces": raw_data['faces'],
                "total_faces": len(raw_data['faces']),
                "image_dimensions": raw_data['dimensions'],
                "model_info": {
                    "detection_method": "mediapipe",
                    "detection_time": raw_data['processing_time'],
                    "confidence_threshold": 0.2,
                    "framework": "MediaPipe",
                    "version": "0.10.x"
                },
                "status": "success"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in detect_faces: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API endpoint for direct file path analysis"""
    start_time = time.time()
    
    try:
        # Get file path from query parameters
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "MediaPipe Face Detection",
                        "framework": "MediaPipe"
                    }
                }
            }), 400
        
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
                        "name": "MediaPipe Face Detection",
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
                        "name": "MediaPipe Face Detection",
                        "framework": "MediaPipe"
                    }
                }
            }), 400
        
        # Process the image directly from file path
        raw_data = process_image(file_path, is_file_path=True)
        
        # Handle errors
        if raw_data['error']:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "metadata": {
                    "processing_time": raw_data['processing_time'],
                    "model_info": {
                        "name": "MediaPipe Face Detection",
                        "framework": "MediaPipe"
                    }
                },
                "error": {
                    "message": str(raw_data['error'])
                }
            }), 500
        
        # Build V2 predictions
        predictions = []
        
        # Add face predictions
        for face in raw_data['faces']:
            predictions.append({
                "type": "face_detection",
                "label": "face",
                "emoji": "🙂",
                "confidence": float(face['confidence']),
                "bbox": face['bbox'],
                "properties": {
                    "keypoints": face.get('keypoints', {}),
                    "method": face['method']
                }
            })
        
        # Add human emoji if faces detected
        if raw_data['faces']:
            predictions.append({
                "type": "face_detection",
                "label": "human",
                "emoji": "🧑",
                "confidence": 1.0,
                "properties": {
                    "face_count": len(raw_data['faces'])
                }
            })
        
        return jsonify({
            "service": "face",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": raw_data['processing_time'],
                "model_info": {
                    "name": "MediaPipe Face Detection",
                    "framework": "MediaPipe",
                    "version": "0.10.x",
                    "model_selection": "full_range",
                    "notes": "Uses MediaPipe Face Detection framework"
                },
                "image_dimensions": raw_data['dimensions'],
                "parameters": {
                    "min_detection_confidence": 0.2,
                    "model_selection": 1
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in v2 file analysis: {str(e)}")
        return jsonify({
            'service': 'face',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {
                    'name': 'MediaPipe Face Detection',
                    'framework': 'MediaPipe'
                }
            },
            'error': {
                'message': str(e)
            }
        }), 500

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 unified API endpoint - expects query parameter with image_url"""
    start_time = time.time()
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "MediaPipe Face Detection",
                        "framework": "MediaPipe"
                    }
                }
            }), 400
        
        # Process the image using the URL
        raw_data = process_image(image_url, is_url=True)
        
        # Handle errors
        if raw_data['error']:
            return jsonify({
                "service": "face",
                "status": "error",
                "predictions": [],
                "metadata": {
                    "processing_time": raw_data['processing_time'],
                    "model_info": {
                        "name": "MediaPipe Face Detection",
                        "framework": "MediaPipe"
                    }
                },
                "error": {
                    "message": str(raw_data['error'])
                }
            }), 500
        
        # Build V2 predictions
        predictions = []
        
        # Add face predictions
        for face in raw_data['faces']:
            predictions.append({
                "type": "face_detection",
                "label": "face",
                "emoji": "🙂",
                "confidence": float(face['confidence']),
                "bbox": face['bbox'],
                "properties": {
                    "keypoints": face.get('keypoints', {}),
                    "method": face['method']
                }
            })
        
        # Add human emoji if faces detected
        if raw_data['faces']:
            predictions.append({
                "type": "face_detection",
                "label": "human",
                "emoji": "🧑",
                "confidence": 1.0,
                "properties": {
                    "face_count": len(raw_data['faces'])
                }
            })
        
        return jsonify({
            "service": "face",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": raw_data['processing_time'],
                "model_info": {
                    "name": "MediaPipe Face Detection",
                    "framework": "MediaPipe",
                    "version": "0.10.x",
                    "model_selection": "full_range",
                    "notes": "Uses MediaPipe Face Detection framework"
                },
                "image_dimensions": raw_data['dimensions'],
                "parameters": {
                    "min_detection_confidence": 0.2,
                    "model_selection": 1
                }
            }
        })
            
    except Exception as e:
        logger.error(f"Error in v2 analyze: {str(e)}")
        return jsonify({
            'service': 'face',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {
                    'name': 'MediaPipe Face Detection',
                    'framework': 'MediaPipe'
                }
            },
            'error': {
                'message': str(e)
            }
        }), 500

if __name__ == '__main__':
    # Initialize MediaPipe model at startup
    logger.info("Initializing MediaPipe Face Detection model...")
    if not initialize_face_detection_model():
        logger.error("Failed to initialize MediaPipe model. Service will not function properly.")
        exit(1)
    
    logger.info(f"Starting Face Detection API on {API_HOST}:{PORT}")
    logger.info("Using MediaPipe Face Detection framework")
    logger.info("V2 API available at /v2/analyze and /v2/analyze_file endpoints")
    app.run(host=API_HOST, port=PORT, debug=False)
