#!/usr/bin/env python3
"""
Standalone Pose Estimation Service using MediaPipe
Provides dedicated human pose analysis with enhanced classification and features
Extracted from face service for independent scaling and specialized pose capabilities
"""

import os
import io
import time
import logging
import tempfile
import uuid
import json
import random
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
from pose_analyzer import PoseAnalyzer

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

# Pose analysis settings from environment (with defaults)
POSE_MIN_DETECTION_CONFIDENCE = float(os.getenv('POSE_MIN_DETECTION_CONFIDENCE', '0.5'))
POSE_MIN_TRACKING_CONFIDENCE = float(os.getenv('POSE_MIN_TRACKING_CONFIDENCE', '0.5'))
POSE_MODEL_COMPLEXITY = int(os.getenv('POSE_MODEL_COMPLEXITY', '2'))
ENABLE_SEGMENTATION = os.getenv('ENABLE_SEGMENTATION', 'true').lower() == 'true'

# Global emoji mappings - loaded from API on startup
emoji_mappings = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global pose analyzer - initialize once at startup
pose_analyzer = None

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(8 * 1024 * 1024)))  # 8MB

CONFIDENCE_DECIMAL_PLACES = 3

# Pose classifications with enhanced types
POSE_CLASSIFICATIONS = {
    "standing": {"emoji": "🧍", "description": "Person standing upright"},
    "sitting": {"emoji": "🪑", "description": "Person sitting down"},
    "lying": {"emoji": "🛌", "description": "Person lying down"},
    "walking": {"emoji": "🚶", "description": "Person in walking motion"},
    "running": {"emoji": "🏃", "description": "Person in running motion"},
    "dancing": {"emoji": "💃", "description": "Person in dance pose"},
    "exercising": {"emoji": "🏋️", "description": "Person exercising"},
    "waving": {"emoji": "👋", "description": "Person waving"},
    "pointing": {"emoji": "👉", "description": "Person pointing"},
    "unknown": {"emoji": "🧑", "description": "Pose not classified"}
}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_pose_analyzer():
    """Initialize pose analyzer once at startup"""
    global pose_analyzer
    try:
        logger.info("Initializing MediaPipe Pose Analyzer...")
        
        pose_analyzer = PoseAnalyzer(
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=POSE_MODEL_COMPLEXITY,
            enable_segmentation=ENABLE_SEGMENTATION
        )
        
        logger.info("✅ Pose Analyzer initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing Pose Analyzer: {str(e)}")
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
        logger.warning(f"⚠️ Could not load emoji mappings: {e}. Using built-in pose emojis.")
        emoji_mappings = {}

def get_emoji(pose_type):
    """Get emoji for pose type, with fallback to built-in classifications"""
    # Try API emoji mappings first
    emoji = emoji_mappings.get(pose_type.lower(), "")
    if emoji:
        return emoji
    
    # Fallback to built-in pose classifications
    return POSE_CLASSIFICATIONS.get(pose_type, {}).get("emoji", "🧑")

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image(url):
    """Download image from URL and return as bytes"""
    try:
        headers = {'User-Agent': 'MediaPipe Pose Analysis Service'}
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

def process_image(image_source, is_url=False, is_file_path=False):
    """Process image and perform pose analysis - returns structured data"""
    start_time = time.time()
    temp_path = None
    
    try:
        # Handle image source
        if is_url:
            image_bytes = download_image(image_source)
        elif is_file_path:
            # Direct file path - use analyzer directly
            pose_data = pose_analyzer.analyze_pose(image_source)
            processing_time = time.time() - start_time
            return {
                **pose_data,
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
        
        # Perform pose analysis
        pose_data = pose_analyzer.analyze_pose(temp_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            **pose_data,
            'processing_time': processing_time,
            'error': None
        }
    
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        processing_time = time.time() - start_time
        return {
            'predictions': [],
            'persons_detected': 0,
            'image_dimensions': {'width': 0, 'height': 0},
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
    global pose_analyzer
    try:
        # Check if pose analyzer is initialized
        pose_status = 'ready' if pose_analyzer is not None else 'not_initialized'
        
        return jsonify({
            'status': 'healthy' if pose_status == 'ready' else 'degraded',
            'service': 'pose',
            'capabilities': ['pose_estimation', 'pose_classification', 'body_segmentation', 'joint_analysis'],
            'models': {
                'pose_estimation': {
                    'status': pose_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Pose',
                    'landmarks': 33,
                    'complexity': POSE_MODEL_COMPLEXITY
                }
            },
            'supported_poses': list(POSE_CLASSIFICATIONS.keys()),
            'endpoints': [
                "GET /health - Health check",
                "GET /v3/analyze?url=<image_url> - Analyze pose from URL", 
                "GET /v3/analyze?file=<file_path> - Analyze pose from file",
                "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
                "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
            ],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'pose',
            'error': str(e)
        }), 500

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3():
    """Unified V3 API endpoint for pose analysis"""
    start_time = time.time()
    
    try:
        # Handle POST file upload
        if request.method == 'POST':
            # Check for file upload
            if 'file' not in request.files:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file provided in POST request"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file selected"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            # Validate file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            # Process uploaded file
            try:
                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    file.save(temp_path)
                
                # Process with pose analyzer
                result_data = process_image(temp_path, is_file_path=True)
                
                # Handle processing errors
                if result_data['error']:
                    return jsonify({
                        "service": "pose",
                        "status": "error",
                        "predictions": [],
                        "metadata": {
                            "processing_time": round(result_data['processing_time'], 3),
                            "model_info": {"framework": "MediaPipe Pose"}
                        },
                        "error": {"message": str(result_data['error'])}
                    }), 500
                
                # Build successful response
                enhanced_predictions = []
                for prediction in result_data['predictions']:
                    is_shiny, shiny_roll = check_shiny()
                    
                    enhanced_prediction = {
                        "properties": {
                            "landmarks": prediction['landmarks'],
                            "pose_analysis": prediction['pose_analysis']
                        }
                    }
                    
                    # Add shiny flag for rare detections
                    if is_shiny:
                        enhanced_prediction["shiny"] = True
                        logger.info(f"✨ SHINY POSE LANDMARKS DETECTED! Roll: {shiny_roll} ✨")
                    
                    enhanced_predictions.append(enhanced_prediction)
                
                return jsonify({
                    "service": "pose",
                    "status": "success",
                    "predictions": enhanced_predictions,
                    "metadata": {
                        "processing_time": round(result_data['processing_time'], 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                })
                
            except Exception as e:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process uploaded image: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 500
            finally:
                # Clean up temporary file
                try:
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Could not clean up temp file: {e}")
        
        # Handle GET requests
        # Get parameters from query string
        url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input - exactly one parameter required
        if not url and not file_path:
            return jsonify({
                "service": "pose",
                "status": "error", 
                "predictions": [],
                "error": {"message": "Must provide either 'url' or 'file' parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "MediaPipe Pose"
                    }
                }
            }), 400
        
        if url and file_path:
            return jsonify({
                "service": "pose",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "MediaPipe Pose"
                    }
                }
            }), 400
        
        # Handle URL input
        if url:
            result_data = process_image(url, is_url=True)
        
        # Handle file path input
        elif file_path:
            # Validate file path
            if not os.path.exists(file_path):
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file_path}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "MediaPipe Pose"
                        }
                    }
                }), 404
            
            if not allowed_file(file_path):
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "File type not allowed"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "MediaPipe Pose"
                        }
                    }
                }), 400
            
            result_data = process_image(file_path, is_file_path=True)
        
        # Handle processing errors
        if result_data['error']:
            return jsonify({
                "service": "pose",
                "status": "error",
                "predictions": [],
                "metadata": {
                    "processing_time": round(result_data['processing_time'], 3),
                    "model_info": {
                        "framework": "MediaPipe Pose"
                    }
                },
                "error": {
                    "message": str(result_data['error'])
                }
            }), 500
        
        # Build V3 response with enhanced pose predictions
        enhanced_predictions = []
        
        for prediction in result_data['predictions']:
            is_shiny, shiny_roll = check_shiny()
            
            enhanced_prediction = {
                "properties": {
                    "landmarks": prediction['landmarks'],
                    "pose_analysis": prediction['pose_analysis']
                }
            }
            
            # Add shiny flag for rare detections
            if is_shiny:
                enhanced_prediction["shiny"] = True
                logger.info(f"✨ SHINY POSE LANDMARKS DETECTED! Roll: {shiny_roll} ✨")
            
            enhanced_predictions.append(enhanced_prediction)
        
        return jsonify({
            "service": "pose",
            "status": "success",
            "predictions": enhanced_predictions,
            "metadata": {
                "processing_time": round(result_data['processing_time'], 3),
                "model_info": {
                    "framework": "MediaPipe Pose"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in V3 pose analysis: {str(e)}")
        return jsonify({
            'service': 'pose',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {
                    'framework': 'MediaPipe Pose'
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
    # Initialize pose analyzer at startup
    logger.info("Initializing Pose Analysis Service...")
    if not initialize_pose_analyzer():
        logger.error("Failed to initialize Pose Analyzer. Service will not function properly.")
        exit(1)
    
    # Load emoji mappings on startup
    load_emoji_mappings()
    
    # Determine host based on private mode (like other services)
    host = "127.0.0.1" if PRIVATE_MODE else "0.0.0.0"
    
    logger.info(f"Starting Pose Analysis Service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE_MODE}")
    logger.info(f"Model complexity: {POSE_MODEL_COMPLEXITY}, Segmentation: {ENABLE_SEGMENTATION}")
    logger.info("Dedicated pose estimation service with enhanced pose classification")
    logger.info("V3 API available at /v3/analyze endpoint with V2 backward compatibility")
    app.run(host=host, port=PORT, debug=False)
