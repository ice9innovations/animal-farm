#!/usr/bin/env python3
"""
YOLOv11 Object Detection REST API Service
Provides object detection using Ultralytics YOLOv11 model with custom Object365 training.
"""

import json
import requests
import os
import logging
import random
import time
from typing import List, Dict, Any
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from yolo_analyzer import YoloAnalyzer

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables as strings first
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
CONFIDENCE_THRESHOLD_STR = os.getenv('CONFIDENCE_THRESHOLD')
DATASET = os.getenv('DATASET')
MODEL_PATH = os.getenv('MODEL_PATH')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'YOLO')

# Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Convert to appropriate types after validation
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR) if CONFIDENCE_THRESHOLD_STR else 0.25

# Configuration
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
IOU_THRESHOLD = 0.3  # IoU threshold for NMS
MAX_DETECTIONS = 100  # Maximum number of detections per image

# Global analyzer instance
yolo_analyzer = None

def check_shiny() -> tuple[bool, int]:
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_yolo11_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized YOLOv11 response with object detections"""
    detections = data.get('detections', [])
    
    # Create unified prediction format
    predictions = []
    for detection in detections:
        bbox = detection.get('bbox', {})
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "label": detection.get('class_name', ''),
            "confidence": float(detection.get('confidence', 0)),
            "bbox": {
                "x": bbox.get('x', 0),
                "y": bbox.get('y', 0),
                "width": bbox.get('width', 0),
                "height": bbox.get('height', 0)
            }
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {detection.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
        # Add emoji if present
        if detection.get('emoji'):
            prediction["emoji"] = detection['emoji']
        
        predictions.append(prediction)
    
    return {
        "service": SERVICE_NAME if SERVICE_NAME else "yolo",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": yolo_analyzer.get_model_info() if yolo_analyzer else {}
        }
    }

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image (in-memory processing)"""
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError("Downloaded file too large")
        
        # Return PIL Image directly from bytes (no temp files)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image (in-memory processing)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not is_allowed_file(file_path):
        raise ValueError("File type not allowed")
    
    try:
        # Load directly into memory, no temp files
        image = Image.open(file_path).convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def initialize_yolo_analyzer() -> bool:
    """Initialize YOLO analyzer once at startup - fail fast"""
    global yolo_analyzer
    try:
        logger.info("Initializing YOLO Analyzer...")
        
        yolo_analyzer = YoloAnalyzer(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            max_detections=MAX_DETECTIONS,
            service_name=SERVICE_NAME,
            dataset=DATASET
        )
        
        # Initialize the model
        if not yolo_analyzer.initialize():
            logger.error("❌ Failed to initialize YOLO Analyzer")
            return False
            
        logger.info("✅ YOLO Analyzer initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing YOLO Analyzer: {str(e)}")
        return False

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("YOLOv11 service: CORS enabled for direct browser communication")

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "status": "error"}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "status": "error"}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test if YOLOv11 analyzer is actually working
    try:
        if not yolo_analyzer:
            raise ValueError("Analyzer not loaded")
        
        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_result = yolo_analyzer.analyze_from_array(test_image)
        
        if not test_result.get('success'):
            raise ValueError(f"Analyzer test failed: {test_result.get('error')}")
        
        analyzer_status = "loaded"
        status = "healthy"
        
    except Exception as e:
        analyzer_status = f"error: {str(e)}"
        status = "unhealthy"
        
        return jsonify({
            "status": status,
            "reason": f"YOLOv11 analyzer error: {str(e)}",
            "service": "YOLOv11 Object Detection"
        }), 503
    
    model_info = yolo_analyzer.get_model_info() if yolo_analyzer else {}
    
    return jsonify({
        "status": status,
        "service": "YOLOv11 Object Detection",
        "analyzer": {
            "status": analyzer_status,
            **model_info
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "custom_model": bool(MODEL_PATH and os.path.exists(MODEL_PATH)),
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /v3/analyze - V3 compatibility",
            "GET /classes - Get supported classes",
            "GET /v2/analyze - V2 compatibility (deprecated)",
            "GET /v2/analyze_file - V2 compatibility (deprecated)"
        ]
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    if yolo_analyzer:
        try:
            class_names = yolo_analyzer.get_supported_classes()
            return jsonify({
                "classes": class_names,
                "total_classes": len(class_names),
                "source": "Custom Model" if (MODEL_PATH and os.path.exists(MODEL_PATH)) else "Standard YOLO"
            })
        except Exception as e:
            logger.warning(f"Error getting class names from analyzer: {e}")
    
    return jsonify({
        "classes": [],
        "total_classes": 0,
        "source": "Analyzer not loaded or classes unavailable"
    })

# V2 Compatibility Routes - Translate parameters and call analyze
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation: image_url -> url
        new_args = {'url': image_url}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let new analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        # Parameter translation: file_path -> file
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing (pure in-memory)"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": SERVICE_NAME if SERVICE_NAME else "yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        # Step 1: Get image into memory from any source (NO FILE SYSTEM OPERATIONS)
        if request.method == 'POST' and 'file' in request.files:
            # Handle file upload - pure in-memory processing
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
            # Validate file size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
            
            try:
                # Pure in-memory processing - no temp files
                file_data = uploaded_file.read()
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process uploaded image: {str(e)}", 500)
        
        else:
            # Handle URL or file parameter
            url = request.args.get('url')
            file_path = request.args.get('file')
            
            if not url and not file_path:
                return error_response("Must provide either 'url' or 'file' parameter, or POST a file")
            
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")
            
            if url:
                # Download from URL directly into memory
                try:
                    image = download_image_from_url(url)
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                try:
                    image = validate_image_file(file_path)
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image using the analyzer (unified processing path)
        if not yolo_analyzer:
            return error_response("YOLO analyzer not initialized", 500)
        
        processing_result = yolo_analyzer.analyze_from_array(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_yolo11_response(
            processing_result,
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls analyze function directly"""
    return analyze()


if __name__ == '__main__':
    # Initialize analyzer
    logger.info("Starting YOLOv11 service...")
    logger.info(f"Looking for model at: {MODEL_PATH}")
    
    analyzer_loaded = initialize_yolo_analyzer()
    
    if not analyzer_loaded:
        logger.error("Failed to load YOLOv11 analyzer. Service will run but detection will fail.")
        logger.error("Please ensure YOLOv11 models are available or install ultralytics: pip install ultralytics")
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting YOLOv11 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Analyzer loaded: {analyzer_loaded}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Custom model: {bool(MODEL_PATH and os.path.exists(MODEL_PATH))}")
    logger.info("🚀 In-memory processing enabled - no temp files created")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
