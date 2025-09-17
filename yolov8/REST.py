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
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import numpy as np

from dotenv import load_dotenv

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
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
IOU_THRESHOLD = 0.3  # IoU threshold for NMS (lowered from 0.45 for better duplicate detection)
MAX_DETECTIONS = 100  # Maximum number of detections per image

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dead Discord code removed - no longer used

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
    """Load emoji mappings from GitHub, fall back to local cache"""
    global emoji_mappings
    
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 YOLO: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            emoji_mappings = response.json()
            logger.info(f"✅ YOLO: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  YOLO: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
            logger.info(f"✅ YOLO: Successfully loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.error(f"❌ YOLO: Failed to load local emoji mappings: {local_error}")
        raise Exception(f"Both GitHub and local emoji mappings failed: GitHub download disabled or failed, Local cache={local_error}")

# Load emoji mappings on startup
load_emoji_mappings()

def get_emoji(word: str) -> str:
    """Get emoji using direct mapping lookup with underscore normalization"""
    if not word:
        return None
    
    # Normalize word format: lowercase with underscores (consistent with ollama-api)
    word_clean = word.lower().strip().replace(' ', '_')
    return emoji_mappings.get(word_clean)

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

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

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Extract coordinates
    x1_1, y1_1 = box1['x'], box1['y']
    x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
    
    x1_2, y1_2 = box2['x'], box2['y']
    x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height'] 
    union = area1 + area2 - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0.0

def apply_iou_filtering(detections: List[Dict], iou_threshold: float = IOU_THRESHOLD) -> List[Dict]:
    """Apply IoU-based filtering to merge overlapping detections of the same class"""
    if not detections:
        return detections
    
    # Group detections by class
    class_groups = {}
    for detection in detections:
        class_name = detection.get('class_name', '')
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(detection)
    
    filtered_detections = []
    
    # Process each class separately
    for class_name, class_detections in class_groups.items():
        if len(class_detections) == 1:
            # Only one detection for this class, keep it
            filtered_detections.extend(class_detections)
            continue
        
        # For multiple detections of the same class, apply IoU filtering
        keep_indices = []
        
        for i, det1 in enumerate(class_detections):
            should_keep = True
            
            for j in keep_indices:
                det2 = class_detections[j]
                bbox1 = det1.get('bbox', {})
                bbox2 = det2.get('bbox', {})
                
                # Calculate IoU if both have valid bboxes
                if all(k in bbox1 for k in ['x', 'y', 'width', 'height']) and \
                   all(k in bbox2 for k in ['x', 'y', 'width', 'height']):
                    iou = calculate_iou(bbox1, bbox2)
                    
                    if iou > iou_threshold:
                        # High overlap detected
                        if det1['confidence'] <= det2['confidence']:
                            # Current detection has lower confidence, don't keep it
                            should_keep = False
                            logger.debug(f"YOLO IoU filter: Removing {class_name} "
                                       f"conf={det1['confidence']:.3f} (IoU={iou:.3f} with "
                                       f"conf={det2['confidence']:.3f})")
                            break
                        else:
                            # Current detection has higher confidence, remove the previous one
                            keep_indices.remove(j)
                            logger.debug(f"YOLO IoU filter: Replacing {class_name} "
                                       f"conf={det2['confidence']:.3f} with "
                                       f"conf={det1['confidence']:.3f} (IoU={iou:.3f})")
            
            if should_keep:
                keep_indices.append(i)
        
        # Add the kept detections
        for i in keep_indices:
            filtered_detections.append(class_detections[i])
        
        logger.debug(f"YOLO IoU filter: {class_name} {len(class_detections)} → {len(keep_indices)} detections")
    
    return filtered_detections

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
        # Use simple local emoji lookup (get_emoji handles underscore normalization)
        emoji = get_emoji(clean_name)
        if emoji:
            # Show the normalized name in debug output
            normalized_name = clean_name.replace(' ', '_')
            logger.debug(f"Local emoji lookup: '{clean_name}' → '{normalized_name}' → {emoji}")
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

def create_yolo_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized YOLO response with object detections"""
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

    # Sort predictions by confidence (highest first)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "service": "yolo",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "Ultralytics YOLOv8"
            }
        }
    }

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
                        "x": x1,
                        "y": y1,
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
    
    # Apply IoU-based filtering to merge overlapping detections of the same class
    filtered_detections = apply_iou_filtering(detections)
    
    # Limit number of detections
    return filtered_detections[:MAX_DETECTIONS]

def process_image_for_yolo(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns YOLOv8 detection data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing with PIL Image support
    """
    start_time = time.time()
    
    try:
        if not model:
            raise ValueError("Model not loaded")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Get image dimensions
        image_width, image_height = image.size
            
        # Run YOLO detection on PIL Image directly
        logger.debug(f"Running YOLO detection on PIL Image ({image_width}x{image_height})")
        
        # Convert PIL Image to numpy array for YOLO
        import numpy as np
        image_array = np.array(image)
        
        # Use standard FP32 inference for stability
        results = model.predict(
            image_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=device
        )
        
        # Process results
        detections = process_yolo_results(results)
        
        logger.debug(f"Detected {len(detections)} objects")
        
        processing_time = round(time.time() - start_time, 3)
        
        return {
            "success": True,
            "data": {
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                },
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                    "device": str(model.device) if hasattr(model, 'device') else device
                }
            },
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        logger.error(f"Error during YOLO detection: {e}")
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "processing_time": processing_time
        }

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
    # Test if YOLOv8 model is actually working
    try:
        if not model:
            raise ValueError("Model not loaded")
        
        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_result = process_image_for_yolo(test_image)
        
        if not test_result.get('success'):
            raise ValueError(f"Model test failed: {test_result.get('error')}")
        
        model_status = "loaded"
        status = "healthy"
        
    except Exception as e:
        model_status = f"error: {str(e)}"
        status = "unhealthy"
        
        return jsonify({
            "status": status,
            "reason": f"YOLOv8 model error: {str(e)}",
            "service": "YOLOv8 Object Detection"
        }), 503
    
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
        "status": status,
        "service": "YOLOv8 Object Detection",
        "model": {
            "status": model_status,
            **model_info
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "supported_classes": len(COCO_CLASSES),
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
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES)
    })

# Original V2 endpoints removed - now handled by compatibility routes above

# V2 Compatibility Routes - Translate parameters and call V3
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
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        # Step 1: Get image into memory from any source
        if request.method == 'POST' and 'file' in request.files:
            # Handle file upload
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            # Validate file size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
            
            try:
                from io import BytesIO
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
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        return error_response("Invalid URL format")
                    
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        return error_response("URL does not point to an image")
                    
                    if len(response.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")
                    
                    from io import BytesIO
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                
                if not is_allowed_file(file_path):
                    return error_response("File type not allowed")
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_yolo(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_yolo_response(
            processing_result["data"],
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls new analyze function directly"""
    return analyze()


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
