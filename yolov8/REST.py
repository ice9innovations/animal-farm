#!/usr/bin/env python3
"""
YOLOv8 Object Detection REST API Service
Provides object detection using Ultralytics YOLOv8 model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import json
import requests
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import numpy as np

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')  # Must be set in .env
API_PORT = int(os.getenv('API_PORT'))  # Must be set in .env
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '2.0'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTTP port for endpoint
PORT = int(os.getenv('PORT', '7773'))


# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
PRIVATE = os.getenv('PRIVATE', 'False').lower() == 'true'
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
MAX_DETECTIONS = 100  # Maximum number of detections per image

load_dotenv()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Environment variables validation
for var in ['DISCORD_TOKEN', 'DISCORD_GUILD', 'DISCORD_CHANNEL']:
    if not os.getenv(var):
        logger.warning(f"Environment variable {var} not set")

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL', '').split(',') if os.getenv('DISCORD_CHANNEL') else []

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

logger.info(f"Using device: {device}")

# Global variables for model and data
model = None

# Load emoji mappings from central API
emoji_mappings = {}

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    logger.info(f"🔄 YOLO: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    logger.info(f"✅ YOLO: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

# Load emoji mappings on startup
load_emoji_mappings()

def get_emoji(word: str) -> str:
    """Get emoji using direct mapping lookup"""
    if not word:
        return None
    
    word_clean = word.lower().strip()
    return emoji_mappings.get(word_clean)

# COCO class names (YOLOv8 uses COCO dataset classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def initialize_yolo_model(model_path: str = None) -> bool:
    """Initialize YOLOv8 model with FP16 optimization"""
    global model
    try:
        # Try different model sizes in order of preference (largest to smallest)
        model_candidates = [
            model_path,
            'yolov8x.pt',  # Extra Large - most accurate
            'yolov8l.pt',  # Large
            'yolov8m.pt',  # Medium
            'yolov8s.pt',  # Small
            'yolov8n.pt',  # Nano - fastest
        ]
        
        for model_file in model_candidates:
            if model_file is None:
                continue
                
            try:
                logger.info(f"Attempting to load YOLO model: {model_file}")
                model = YOLO(model_file)
                
                # FP16 disabled for stability - causing model load failures on some systems
                # TODO: Implement proper FP16 support for YOLO models
                precision = "FP32"
                logger.info(f"Using {precision} for YOLO model stability")
                
                # Test the model with a dummy prediction
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = model.predict(dummy_image, verbose=False, device=device)
                
                logger.info(f"YOLO model loaded successfully: {model_file}")
                logger.info(f"Model device: {model.device}")
                logger.info(f"Model precision: {precision}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
                continue
                
        logger.error("No YOLO model could be loaded")
        return False
        
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        return False


def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name using local emoji service (optimized - no HTTP requests)"""
    clean_name = class_name.lower().strip()
    
    try:
        # Use simple local emoji lookup
        emoji = get_emoji(clean_name)
        if emoji:
            logger.debug(f"Local emoji lookup: '{clean_name}' → {emoji}")
            return emoji
        
        logger.debug(f"Local emoji lookup: no emoji found for '{clean_name}'")
        return None
        
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{clean_name}': {e}")
        return None

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_path: str) -> bool:
    """Validate file size"""
    try:
        return os.path.getsize(file_path) <= MAX_FILE_SIZE
    except OSError:
        return False

def process_yolo_results(results) -> List[Dict[str, Any]]:
    """Process YOLO model results into structured format"""
    detections = []
    
    if not results or len(results) == 0:
        return detections
        
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return detections
        
    for box in result.boxes:
        try:
            # Get bounding box coordinates
            coords = box.xyxy[0].tolist()
            coords = [round(x) for x in coords]
            x1, y1, x2, y2 = coords
            
            # Get class information
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id] if class_id < len(result.names) else f"class_{class_id}"
            confidence = round(box.conf[0].item(), 3)
            
            # Only include detections above confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                # Look up emoji from central API
                try:
                    emoji = lookup_emoji(class_name)
                except RuntimeError as e:
                    logger.error(f"Emoji lookup failed for '{class_name}': {e}")
                    raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "emoji": emoji
                }
                
                detections.append(detection)
                
        except Exception as e:
            logger.warning(f"Error processing detection: {e}")
            continue
            
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Deduplicate by class name - keep only highest confidence per class
    seen_classes = {}
    deduplicated = []
    
    for detection in detections:
        class_name = detection.get('class_name', '')
        if class_name:
            if class_name not in seen_classes:
                seen_classes[class_name] = detection
                deduplicated.append(detection)
                logger.debug(f"YOLO: Keeping highest confidence {class_name}: {detection['confidence']}")
            else:
                logger.debug(f"YOLO: Skipping duplicate {class_name}: {detection['confidence']}")
    
    # Limit number of detections
    return deduplicated[:MAX_DETECTIONS]

def detect_objects(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Detect objects in image using YOLOv8"""
    if not model:
        return {"error": "Model not loaded", "status": "error"}
        
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {"error": "Image file not found", "status": "error"}
            
        if not validate_file_size(image_path):
            return {"error": "File too large", "status": "error"}
            
        # Run YOLO detection with FP16 optimization
        logger.info(f"Running YOLO detection on: {image_path}")
        
        # Use standard FP32 inference for stability
        results = model.predict(
            image_path,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=device
        )
        
        # Process results
        detections = process_yolo_results(results)
        
        logger.info(f"Detected {len(detections)} objects")
        
        # Get image dimensions for context
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception:
            image_width = image_height = None
            
        # Build response
        response = {
            "YOLO": {
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                } if image_width and image_height else None,
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                    "device": str(model.device) if hasattr(model, 'device') else device
                },
                "status": "success"
            }
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            try:
                if os.path.exists(image_path) and image_path.startswith(UPLOAD_FOLDER):
                    os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {image_path}: {e}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error detecting objects in {image_path}: {e}")
        return {"error": f"Detection failed: {str(e)}", "status": "error"}

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("YOLO service: CORS enabled for direct browser communication")

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
    model_status = "loaded" if model else "not_loaded"
    model_info = {}
    
    if model:
        try:
            model_info = {
                "device": str(model.device) if hasattr(model, 'device') else device,
                "model_name": str(model.model.model[-1].__class__.__name__) if hasattr(model, 'model') else "unknown"
            }
        except Exception:
            pass
            
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "model_info": model_info,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "supported_classes": len(COCO_CLASSES)
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES)
    })

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API endpoint for direct file path analysis"""
    import time
    start_time = time.time()
    
    try:
        # Get file path from query parameters
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                "service": "yolo",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "yolo",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        if not is_allowed_file(file_path):
            return jsonify({
                "service": "yolo",
                "status": "error",
                "predictions": [],
                "error": {"message": "File type not allowed"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Detect objects directly from file (no cleanup needed - we don't own the file)
        result = detect_objects(file_path, cleanup=False)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "yolo",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        yolo_data = result.get('YOLO', {})
        detections = yolo_data.get('detections', [])
        image_dims = yolo_data.get('image_dimensions', {})
        
        # Create unified prediction format
        predictions = []
        for detection in detections:
            bbox = detection.get('bbox', {})
            prediction = {
                "type": "object_detection",
                "label": detection.get('class_name', ''),
                "confidence": float(detection.get('confidence', 0)),  # Already normalized 0-1
                "bbox": {
                    "x": bbox.get('x1', 0),
                    "y": bbox.get('y1', 0),
                    "width": bbox.get('width', 0),
                    "height": bbox.get('height', 0)
                }
            }
            
            # Add emoji if present
            if detection.get('emoji'):
                prediction["emoji"] = detection['emoji']
            
            predictions.append(prediction)
        
        return jsonify({
            "service": "yolo",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "YOLOv8",
                    "framework": "Ultralytics"
                },
                "image_dimensions": image_dims,
                "parameters": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V2 file analysis error: {e}")
        return jsonify({
            "service": "yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API endpoint with unified response format"""
    import time
    start_time = time.time()
    filepath = None
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "yolo",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Download and process image (reuse existing logic)
        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
            
            # Download image
            filename = uuid.uuid4().hex + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError("URL does not point to an image")
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if not validate_file_size(filepath):
                os.remove(filepath)
                raise ValueError("Downloaded file too large")
            
            # Detect objects using existing function
            result = detect_objects(filepath)
            
            # Clear filepath after successful processing (detect_objects handles cleanup)
            filepath = None
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "yolo",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Detection failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            yolo_data = result.get('YOLO', {})
            detections = yolo_data.get('detections', [])
            image_dims = yolo_data.get('image_dimensions', {})
            
            # Create unified prediction format
            predictions = []
            for detection in detections:
                bbox = detection.get('bbox', {})
                prediction = {
                    "type": "object_detection",
                    "label": detection.get('class_name', ''),
                    "confidence": float(detection.get('confidence', 0)),  # Already normalized 0-1
                    "bbox": {
                        "x": bbox.get('x1', 0),
                        "y": bbox.get('y1', 0),
                        "width": bbox.get('width', 0),
                        "height": bbox.get('height', 0)
                    }
                }
                
                # Add emoji if present
                if detection.get('emoji'):
                    prediction["emoji"] = detection['emoji']
                
                predictions.append(prediction)
            
            # Get model info
            model_info = yolo_data.get('model_info', {})
            
            return jsonify({
                "service": "yolo",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "YOLOv8",
                        "framework": "Ultralytics"
                    },
                    "image_dimensions": image_dims,
                    "parameters": {
                        "confidence_threshold": CONFIDENCE_THRESHOLD,
                        "iou_threshold": IOU_THRESHOLD
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image URL {image_url}: {e}")
            return jsonify({
                "service": "yolo",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        logger.error(f"V2 API error: {e}")
        return jsonify({
            "service": "yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500
    finally:
        # Safe cleanup
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Handle URL parameter
        url = request.args.get('url') or request.args.get('img')
        path = request.args.get('path')
        
        if url:
            filepath = None
            try:
                # Validate URL
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    return jsonify({"error": "Invalid URL", "status": "error"}), 400
                    
                # Download image
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    return jsonify({"error": "URL does not point to an image", "status": "error"}), 400
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                # Validate downloaded file
                if not validate_file_size(filepath):
                    os.remove(filepath)
                    return jsonify({"error": "Downloaded file too large", "status": "error"}), 400
                    
                result = detect_objects(filepath)
                
                # Clear filepath after successful processing (detect_objects handles cleanup)
                filepath = None
                
                return jsonify(result)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading image from URL {url}: {e}")
                return jsonify({"error": "Failed to download image", "status": "error"}), 400
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                return jsonify({"error": "Error processing image", "status": "error"}), 500
            finally:
                # Safe cleanup
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup file {filepath}: {e}")
                
        elif path:
            # Handle local path (only if not in private mode)
            if PRIVATE:
                return jsonify({"error": "Path access disabled in private mode", "status": "error"}), 403
                
            if not os.path.exists(path):
                return jsonify({"error": "File not found", "status": "error"}), 404
                
            if not is_allowed_file(path):
                return jsonify({"error": "File type not allowed", "status": "error"}), 400
                
            result = detect_objects(path)
            return jsonify(result)
            
        else:
            # Return HTML form
            try:
                with open('form.html', 'r') as file:
                    html = file.read()
            except FileNotFoundError:
                html = f'''<!DOCTYPE html>
<html>
<head><title>YOLOv8 Object Detection</title></head>
<body>
<h1>YOLOv8 Object Detection Service</h1>
<form enctype="multipart/form-data" action="" method="POST">
    <input type="hidden" name="MAX_FILE_SIZE" value="{MAX_FILE_SIZE}" />
    <p>Upload an image file:</p>
    <input name="uploadedfile" type="file" accept="image/*" required /><br /><br />
    <input type="submit" value="Detect Objects" />
</form>
<p>Supported formats: {', '.join(ALLOWED_EXTENSIONS)}</p>
<p>Max file size: {MAX_FILE_SIZE // (1024*1024)}MB</p>
<p>Detects {len(COCO_CLASSES)} object classes</p>
</body>
</html>'''
            return html
            
    elif request.method == 'POST':
        filepath = None
        try:
            if 'uploadedfile' not in request.files:
                return jsonify({"error": "No file uploaded", "status": "error"}), 400
                
            file = request.files['uploadedfile']
            if file.filename == '':
                return jsonify({"error": "No file selected", "status": "error"}), 400
                
            if not is_allowed_file(file.filename):
                return jsonify({"error": "File type not allowed", "status": "error"}), 400
                
            # Save uploaded file
            filename = uuid.uuid4().hex + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Validate file size
            if not validate_file_size(filepath):
                os.remove(filepath)
                return jsonify({"error": "File too large", "status": "error"}), 400
                
            result = detect_objects(filepath)
            
            # Clear filepath after successful processing (detect_objects handles cleanup)
            filepath = None
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return jsonify({"error": "Error processing upload", "status": "error"}), 500
        finally:
            # Safe cleanup
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {filepath}: {e}")

if __name__ == '__main__':
    # Initialize model
    logger.info("Starting YOLOv8 service...")
    
    model_loaded = initialize_yolo_model()
    
    
    if not model_loaded:
        logger.error("Failed to load YOLO model. Service will run but detection will fail.")
        logger.error("Please ensure YOLOv8 models are available or install ultralytics: pip install ultralytics")
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting YOLOv8 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Supported classes: {len(COCO_CLASSES)}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
