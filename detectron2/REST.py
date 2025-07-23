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
API_HOST = os.getenv('API_HOST')  # Must be set in .env
API_PORT = int(os.getenv('API_PORT'))  # Must be set in .env
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '2.0'))

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
PRIVATE = os.getenv('PRIVATE', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', '7771'))
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

# Load emoji mappings on startup
load_emoji_mappings()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Environment variables validation
for var in ['DISCORD_TOKEN', 'DISCORD_GUILD', 'DISCORD_CHANNEL']:
    if not os.getenv(var):
        logger.warning(f"Environment variable {var} not set")

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL', '').split(',') if os.getenv('DISCORD_CHANNEL') else []

# Global variables
demo = None
coco_classes = []
# Thread lock for model inference
model_lock = threading.Lock()

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
                            "x1": x1_scaled,
                            "y1": y1_scaled,
                            "x2": x2_scaled,
                            "y2": y2_scaled,
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
                                "x1": x1_scaled,
                                "y1": y1_scaled,
                                "x2": x2_scaled,
                                "y2": y2_scaled,
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
    
    # Deduplicate by class name - keep only highest confidence per class
    seen_classes = {}
    deduplicated = []
    
    for detection in detections:
        class_name = detection.get('class_name', '')
        if class_name:
            if class_name not in seen_classes:
                seen_classes[class_name] = detection
                deduplicated.append(detection)
    
    return deduplicated

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
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        if not is_allowed_file(file_path):
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": "File type not allowed"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Detect objects directly from file (no cleanup needed - we don't own the file)
        result = detect_objects(file_path, cleanup=False)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        detectron_data = result.get('DETECTRON', {})
        detections = detectron_data.get('detections', [])
        image_dims = detectron_data.get('image_dimensions', {})
        
        # Create unified prediction format
        predictions = []
        for detection in detections:
            bbox = detection.get('bbox', {})
            prediction = {
                "type": "object_detection",
                "label": detection.get('class_name', ''),
                "confidence": round(float(detection.get('confidence', 0)), 3)  # Normalize to 0-1
            }
            
            # Add bbox if present - convert from x1,y1,x2,y2 to x,y,width,height format
            if bbox:
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)
                
                prediction["bbox"] = {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
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
                    "name": "Detectron2",
                    "framework": "Facebook AI Research"
                },
                "image_dimensions": image_dims,
                "parameters": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V2 file analysis error: {e}")
        return jsonify({
            "service": "detectron2",
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
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "detectron2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Download and process image (reuse existing logic)
        filepath = None
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
                filepath = None  # Mark as already removed
                raise ValueError("Downloaded file too large")
            
            # Detect objects using existing function
            result = detect_objects(filepath)
            filepath = None  # detect_objects handles cleanup
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "detectron2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Detection failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            detectron_data = result.get('DETECTRON', {})
            detections = detectron_data.get('detections', [])
            image_dims = detectron_data.get('image_dimensions', {})
            
            # Create unified prediction format
            predictions = []
            for detection in detections:
                bbox = detection.get('bbox', {})
                prediction = {
                    "type": "object_detection",
                    "label": detection.get('class_name', ''),
                    "confidence": round(float(detection.get('confidence', 0)), 3)  # Normalize to 0-1
                }
                
                # Add bbox if present - convert from x1,y1,x2,y2 to x,y,width,height format
                if bbox:
                    x1 = bbox.get('x1', 0)
                    y1 = bbox.get('y1', 0)
                    x2 = bbox.get('x2', 0)
                    y2 = bbox.get('y2', 0)
                    
                    prediction["bbox"] = {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
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
                        "name": "Detectron2",
                        "framework": "Facebook AI Research"
                    },
                    "image_dimensions": image_dims,
                    "parameters": {
                        "confidence_threshold": CONFIDENCE_THRESHOLD
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image URL {image_url}: {e}")
            return jsonify({
                "service": "detectron2",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
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
        
    except Exception as e:
        logger.error(f"V2 API error: {e}")
        return jsonify({
            "service": "detectron2",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

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
                    filepath = None  # Mark as already removed
                    return jsonify({"error": "Downloaded file too large", "status": "error"}), 400
                    
                result = detect_objects(filepath)
                filepath = None  # detect_objects handles cleanup
                return jsonify(result)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading image from URL {url}: {e}")
                return jsonify({"error": "Failed to download image", "status": "error"}), 400
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                return jsonify({"error": "Error processing image", "status": "error"}), 500
            finally:
                # Ensure cleanup of temporary file
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up temporary file: {filepath}")
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
<head><title>Detectron2 Object Detection</title></head>
<body>
<h1>Detectron2 Object Detection & Instance Segmentation</h1>
<form enctype="multipart/form-data" action="" method="POST">
    <input type="hidden" name="MAX_FILE_SIZE" value="{MAX_FILE_SIZE}" />
    <p>Upload an image file:</p>
    <input name="uploadedfile" type="file" accept="image/*" required /><br /><br />
    <input type="submit" value="Detect Objects" />
</form>
<p>Supported formats: {', '.join(ALLOWED_EXTENSIONS)}</p>
<p>Max file size: {MAX_FILE_SIZE // (1024*1024)}MB</p>
<p>Detects objects and instance segmentation using Detectron2</p>
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
                filepath = None  # Mark as already removed
                return jsonify({"error": "File too large", "status": "error"}), 400
                
            result = detect_objects(filepath)
            filepath = None  # detect_objects handles cleanup
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return jsonify({"error": "Error processing upload", "status": "error"}), 500
        finally:
            # Ensure cleanup of temporary file
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temporary file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {filepath}: {e}")

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
