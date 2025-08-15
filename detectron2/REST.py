#!/usr/bin/env python3
"""
Detectron2 Object Detection and Instance Segmentation REST API Service
Coordination service for Detectron2 framework integration.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import json
import requests
import uuid
import logging
import threading
import random
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from PIL import Image

# Simple emoji lookup - no complex dependencies needed

# Load environment variables first
load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')

# Validate critical environment variables
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT_STR:
    raise ValueError("API_TIMEOUT environment variable is required")

# Convert to appropriate types after validation
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR)

# Detectron2 imports (assumes Detectron2 repo is installed)
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# Local predictor module (exists in Detectron2 repo)
from predictor import VisualizationDemo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# No more mirror-stage dependencies - using direct emoji file loading

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 160

# Performance optimization flags
USE_HALF_PRECISION = True  # Testing FP16 for VRAM optimization

# Configuration file paths (adjust based on Detectron2 installation)
#CONFIG_FILES = [
#    "./mask_rcnn_R_50_FPN_3x.yaml",  # Local config file
#    "./Base-RCNN-FPN.yaml",          # Alternative local config
#    "/home/sd/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
#    "/opt/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
#    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#]

CONFIG_FILE = "/home/sd/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

load_dotenv()

# Load emoji mappings from local JSON file
emoji_mappings = {}

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    logger.info(f"🔄 Detectron2: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    logger.info(f"✅ Detectron2: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

def get_emoji(word: str) -> str:
    """Simple emoji lookup - lowercase and replace spaces with underscores"""
    if not word:
        return None
    
    # Basic normalization: lowercase and replace spaces with underscores  
    clean_word = word.lower().strip().replace(' ', '_')
    
    # Try exact match
    if clean_word in emoji_mappings:
        return emoji_mappings[clean_word]
    
    # Try singular form for common plurals
    if clean_word.endswith('s') and len(clean_word) > 3:
        singular = clean_word[:-1]
        if singular in emoji_mappings:
            return emoji_mappings[singular]
    
    return None

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

# Load emoji mappings on startup
load_emoji_mappings()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Global variables
demo = None
coco_classes = []
# Thread lock for model inference
model_lock = threading.Lock()

# IoU threshold for filtering overlapping detections  
# Lowered from 0.45 to 0.3 to catch more overlapping detections
IOU_THRESHOLD = 0.3

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Extract coordinates - now using consistent x,y,width,height format
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
    
    # Calculate areas using width/height from bbox
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
                            logger.debug(f"Detectron2 IoU filter: Removing {class_name} "
                                       f"conf={det1['confidence']:.3f} (IoU={iou:.3f} with "
                                       f"conf={det2['confidence']:.3f})")
                            break
                        else:
                            # Current detection has higher confidence, remove the previous one
                            keep_indices.remove(j)
                            logger.debug(f"Detectron2 IoU filter: Replacing {class_name} "
                                       f"conf={det2['confidence']:.3f} with "
                                       f"conf={det1['confidence']:.3f} (IoU={iou:.3f})")
            
            if should_keep:
                keep_indices.append(i)
        
        # Add the kept detections
        for i in keep_indices:
            filtered_detections.append(class_detections[i])
        
        logger.debug(f"Detectron2 IoU filter: {class_name} {len(class_detections)} → {len(keep_indices)} detections")
    
    return filtered_detections

def resize_image_for_inference(image, max_size=512):
    """Resize image to speed up inference"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image

def find_config_file() -> Optional[str]:
    """Check for required Detectron2 config file"""
    if os.path.exists(CONFIG_FILE):
        logger.info(f"Found config file: {CONFIG_FILE}")
        return CONFIG_FILE
    else:
        logger.error(f"Required config file not found: {CONFIG_FILE}")
        return None

def setup_cfg(config_file: str, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Any:
    """Setup Detectron2 configuration"""
    try:
        cfg = get_cfg()
        
        # Load config file
        cfg.merge_from_file(config_file)
        
        # Set model weights - use local file if available, otherwise download
        local_model_path = "./model_final_280758.pkl"
        if os.path.exists(local_model_path):
            cfg.MODEL.WEIGHTS = local_model_path
            logger.info(f"Using cached local model: {local_model_path}")
        else:
            cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
            logger.info("Downloading model weights (this may take time on first run)")

        # Set confidence thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        
        # Performance optimizations
        cfg.TEST.DETECTIONS_PER_IMAGE = 20           # Limit to 20 detections max
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Reduce batch size
        
        # Set device - require CUDA
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for Detectron2.")
        cfg.MODEL.DEVICE = "cuda"
        logger.info(f"Using device: {cfg.MODEL.DEVICE}")
 
        cfg.freeze()
        return cfg
    except Exception as e:
        logger.error(f"Failed to setup config: {e}")
        return None

def load_coco_classes() -> bool:
    """Load COCO class names"""
    global coco_classes
    try:
        with open('coco_classes.txt', 'r') as f:
            coco_classes = f.read().splitlines()
        logger.info(f"Loaded {len(coco_classes)} COCO classes")
        return True
    except Exception as e:
        logger.error(f"Failed to load COCO classes: {e}")
        return False


def initialize_detectron2() -> bool:
    """Initialize Detectron2 model"""
    global demo
    try:
        # Find config file
        config_file = find_config_file()
        if not config_file:
            logger.error("No Detectron2 config file found")
            return False
            
        # Setup configuration
        cfg = setup_cfg(config_file)
        if cfg is None:
            return False
            
        # Setup logger
        setup_logger(name="fvcore")
        
        # Initialize demo
        demo = VisualizationDemo(cfg)
        
        # FP16 optimization enabled
        logger.info("FP16 half precision enabled via autocast - 50% VRAM reduction expected!")
        
        logger.info("Detectron2 demo initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Detectron2: {e}")
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

def process_detections(predictions: Dict[str, Any], scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict[str, Any]]:
    """Process Detectron2 predictions into structured format"""
    detections = []
    
    if not predictions or "instances" not in predictions:
        logger.warning("No instances in predictions")
        return detections
        
    instances = predictions["instances"]
    
    # Access detectron2 instances attributes properly
    pred_classes = instances.pred_classes if hasattr(instances, 'pred_classes') else None
    scores = instances.scores if hasattr(instances, 'scores') else None
    
    if pred_classes is None or scores is None:
        logger.warning("Missing pred_classes or scores")
        return detections
    
    
    # Get bounding boxes if available
    boxes = getattr(instances, "pred_boxes", None)
    
    for i in range(len(pred_classes)):
        # Extract class ID and confidence with specific error handling
        try:
            class_id = int(pred_classes[i].item())
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Detection {i}: Failed to extract class_id: {e}")
            continue
            
        try:
            confidence = float(scores[i].item())
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Detection {i}: Failed to extract confidence: {e}")
            continue
            
        # Get class name with +1 offset: COCO classes file has "background" at index 0,
        # but Detectron2 models only predict object classes (0-79), so model ID 0 = "person" at index 1
        if 0 <= class_id + 1 < len(coco_classes):
            class_name = coco_classes[class_id + 1]
        else:
            class_name = f"class_{class_id}"
            
        # Only include detections above confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            # Look up emoji (fails loudly if Mirror Stage unavailable/fails)
            try:
                emoji = lookup_emoji(class_name)
            except RuntimeError as e:
                logger.error(f"Emoji lookup failed for '{class_name}': {e}")
                raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "emoji": emoji
            }
            
            # Extract bounding box - try multiple approaches to always get spatial info
            bbox_extracted = False
            if boxes is not None and i < len(boxes):
                # Method 1: Standard tensor extraction
                try:
                    box = boxes[i].tensor.cpu().numpy()[0]
                    
                    if len(box) >= 4:  # Accept any format with at least 4 coordinates
                        x1_scaled = round(float(box[0]) * scale_x)
                        y1_scaled = round(float(box[1]) * scale_y)
                        x2_scaled = round(float(box[2]) * scale_x)
                        y2_scaled = round(float(box[3]) * scale_y)
                        
                        # Ensure coordinates are in correct order and bounds
                        x1_scaled = max(0, min(x1_scaled, x2_scaled))
                        y1_scaled = max(0, min(y1_scaled, y2_scaled))
                        x2_scaled = max(x1_scaled + 1, x2_scaled)  # Ensure width > 0
                        y2_scaled = max(y1_scaled + 1, y2_scaled)  # Ensure height > 0
                        
                        detection["bbox"] = {
                            "x": x1_scaled,
                            "y": y1_scaled,
                            "width": x2_scaled - x1_scaled,
                            "height": y2_scaled - y1_scaled
                        }
                        bbox_extracted = True
                        
                except Exception as e:
                    logger.debug(f"Detection {i}: Standard bbox extraction failed: {e}")
                
                # Method 2: Try alternative tensor access if Method 1 failed
                if not bbox_extracted:
                    try:
                        # Try different tensor access patterns
                        if hasattr(boxes[i], 'tensor'):
                            tensor_data = boxes[i].tensor.cpu().numpy()
                            if len(tensor_data.shape) > 1:
                                box = tensor_data.flatten()[:4]  # Take first 4 values
                            else:
                                box = tensor_data[:4]
                        else:
                            # Direct numpy array access
                            box = boxes[i].cpu().numpy()[:4]
                        
                        if len(box) >= 4:
                            x1_scaled = round(float(box[0]) * scale_x)
                            y1_scaled = round(float(box[1]) * scale_y) 
                            x2_scaled = round(float(box[2]) * scale_x)
                            y2_scaled = round(float(box[3]) * scale_y)
                            
                            x1_scaled = max(0, min(x1_scaled, x2_scaled))
                            y1_scaled = max(0, min(y1_scaled, y2_scaled))
                            x2_scaled = max(x1_scaled + 1, x2_scaled)
                            y2_scaled = max(y1_scaled + 1, y2_scaled)
                            
                            detection["bbox"] = {
                                "x": x1_scaled,
                                "y": y1_scaled,
                                "width": x2_scaled - x1_scaled,
                                "height": y2_scaled - y1_scaled
                            }
                            bbox_extracted = True
                            
                    except Exception as e:
                        logger.debug(f"Detection {i}: Alternative bbox extraction failed: {e}")
                
                if not bbox_extracted:
                    logger.info(f"Detection {i} ({class_name}): No bounding box could be extracted - complex segmentation shape")
            
            detections.append(detection)
            
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Apply IoU-based filtering to merge overlapping detections of the same class
    filtered_detections = apply_iou_filtering(detections)
    
    return filtered_detections

def detect_objects(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Detect objects using Detectron2"""
    if not demo:
        return {"error": "Detectron2 model not loaded", "status": "error"}
        
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {"error": "Image file not found", "status": "error"}
            
        if not validate_file_size(image_path):
            return {"error": "File too large", "status": "error"}
            
        # Read image
        logger.info(f"Running Detectron2 detection on: {image_path}")
        img = read_image(image_path, format="BGR")
        
        # Store original dimensions before resizing
        original_height, original_width = img.shape[:2]
        
        # Resize image for faster inference
        img = resize_image_for_inference(img, max_size=512)
        
        # Store resized dimensions for coordinate scaling
        resized_height, resized_width = img.shape[:2]
        
        # Calculate scaling factors to convert coordinates back to original image
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        
        # Run detection with autocast for FP16 optimization (thread-safe)
        start_time = time.time()
        import torch
        
        # Acquire lock for thread-safe model inference
        with model_lock:
            with torch.amp.autocast('cuda'):
                predictions, visualized_output = demo.run_on_image(img)
                precision_used = "FP16"
        
        detection_time = time.time() - start_time
        
        # Process results with coordinate scaling
        detections = process_detections(predictions, scale_x, scale_y)
        
        logger.info(f"Detected {len(detections)} objects in {detection_time:.2f}s")
        
        # Get image dimensions
        try:
            with Image.open(image_path) as pil_img:
                image_width, image_height = pil_img.size
        except Exception:
            image_width = image_height = None
            
        # Build response
        response = {
            "DETECTRON": {
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                } if image_width and image_height else None,
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "detection_time": round(detection_time, 3),
                    "framework": "Detectron2",
                    "precision": precision_used
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
print("Detectron2 service: CORS enabled for direct browser communication")

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
    model_status = "loaded" if demo else "not_loaded"
    config_file = find_config_file()
    
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "config_file": config_file,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "coco_classes_loaded": len(coco_classes),
        "framework": "Detectron2"
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    return jsonify({
        "classes": coco_classes,
        "total_classes": len(coco_classes),
        "framework": "Detectron2"
    })


# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    import time
    from flask import request
    
    # Get V2 parameter
    image_url = request.args.get('image_url')
    
    if image_url:
        # Create new request args with V3 parameter name
        new_args = request.args.copy()
        new_args = new_args.to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Create a mock request object for V3
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # No parameters - let V3 handle the error
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    import time
    from flask import request
    
    # Get V2 parameter
    file_path = request.args.get('file_path')
    
    if file_path:
        # Create new request args with V3 parameter name
        new_args = {'file': file_path}
        
        # Create a mock request object for V3
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # No parameters - let V3 handle the error
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    import time
    start_time = time.time()
    
    try:
        # Get input parameters - support both url and file
        url = request.args.get('url')
        file = request.args.get('file')
        
        # Validate input - exactly one parameter must be provided
        if not url and not file:
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Must provide either url or file parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        if url and file:
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both url and file parameters"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Handle URL input
        if url:
            filepath = None
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError("Invalid URL format")
                
                # Download image
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError("URL does not point to an image")
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if not validate_file_size(filepath):
                    os.remove(filepath)
                    filepath = None
                    raise ValueError("Downloaded file too large")
                
                # Detect objects using existing function
                result = detect_objects(filepath)
                filepath = None  # detect_objects handles cleanup
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                return jsonify({
                    "service": "detectron2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process URL: {str(e)}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            finally:
                # Ensure cleanup of temporary file
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up temporary file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup file {filepath}: {e}")
        
        # Handle file input
        if file:
            # Validate file path
            if not os.path.exists(file):
                return jsonify({
                    "service": "detectron2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 404
            
            if not is_allowed_file(file):
                return jsonify({
                    "service": "detectron2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "File type not allowed"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            # Detect objects directly from file (no cleanup needed - we don't own the file)
            result = detect_objects(file, cleanup=False)
        
        # Process results (common for both URL and file)
        if result.get('status') == 'error':
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v3 format
        detectron_data = result.get('DETECTRON', {})
        detections = detectron_data.get('detections', [])
        
        # Create unified prediction format
        predictions = []
        for detection in detections:
            bbox = detection.get('bbox', {})
            is_shiny, shiny_roll = check_shiny()
            
            prediction = {
                "label": detection.get('class_name', ''),
                "confidence": round(float(detection.get('confidence', 0)), 3)
            }
            
            # Add shiny flag only for shiny detections
            if is_shiny:
                prediction["shiny"] = True
                logger.info(f"✨ SHINY {detection.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
            
            # Add bbox if present
            if bbox:
                prediction["bbox"] = {
                    "x": bbox.get('x', 0),
                    "y": bbox.get('y', 0),
                    "width": bbox.get('width', 0),
                    "height": bbox.get('height', 0)
                }
            
            # Add emoji if present
            if detection.get('emoji'):
                prediction["emoji"] = detection['emoji']
            
            predictions.append(prediction)
        
        return jsonify({
            "service": "detectron2",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "Facebook AI Research"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V3 API error: {e}")
        return jsonify({
            "service": "detectron2",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500


if __name__ == '__main__':
    # Set multiprocessing start method for Detectron2
    mp.set_start_method("spawn", force=True)
    
    # Initialize services
    logger.info("Starting Detectron2 service...")
    
    coco_loaded = load_coco_classes()
    model_loaded = initialize_detectron2()
    
    # Using local emoji file - no external dependencies
    
    if not model_loaded:
        logger.error("Failed to load Detectron2 model. Service will run but detection will fail.")
        logger.error("Please ensure Detectron2 is installed and config files are available.")
        
    if not coco_loaded:
        logger.error("Failed to load COCO classes. Detection may fail.")
        
    
    # Always use 0.0.0.0 to allow external connections
    host = "0.0.0.0"
    
    logger.info(f"Starting Detectron2 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"COCO classes loaded: {coco_loaded}")
    logger.info("Emoji lookup: Local file mode")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
