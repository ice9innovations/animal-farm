#!/usr/bin/env python3
"""
RT-DETR Object Detection REST API Service
Provides real-time object detection using RT-DETR (Real-Time Detection Transformer).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import json
import requests
import os
import sys
import uuid
import logging
import threading
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import numpy as np

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Load environment variables as strings first
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')

# Validate critical environment variables
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT_STR:
    raise ValueError("API_TIMEOUT environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Convert to appropriate types after validation
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR)
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add rtdetr_pytorch to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rtdetr_pytorch'))
from src.core import YAMLConfig

# Mirror Stage removed - using local emoji service

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
# Configuration constants
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
MAX_DETECTIONS = 100  # Maximum number of detections per image

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# COCO class names (same as your other services)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Mirror Stage emoji lookup (now handled via HTTP)
request_count = 0  # Track requests for periodic cleanup

app = Flask(__name__)
CORS(app)

# Global model variable
model = None
device = None

# IoU threshold for filtering overlapping detections  
IOU_THRESHOLD = 0.3  # Lowered from 0.45 for better duplicate detection

def calculate_iou_rtdetr(box1, box2) -> float:
    """Calculate IoU for RT-DETR format boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
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
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def apply_iou_filtering_rtdetr_scaled(objects, iou_threshold=IOU_THRESHOLD):
    """Apply IoU filtering to scaled detection objects in final coordinate space"""
    if not objects:
        return objects
    
    # Group detections by class
    class_groups = {}
    for obj in objects:
        class_name = obj.get('object', '')
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(obj)
    
    filtered_objects = []
    
    # Process each class separately
    for class_name, class_objects in class_groups.items():
        if len(class_objects) == 1:
            # Only one detection for this class, keep it
            filtered_objects.extend(class_objects)
            continue
        
        # For multiple detections, apply IoU filtering
        class_keep = []
        
        # Sort by confidence (highest first)
        class_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        for obj in class_objects:
            should_keep = True
            
            for kept_obj in class_keep:
                # Both objects have bbox in [x1, y1, x2, y2] format
                box1 = obj['bbox']
                box2 = kept_obj['bbox']
                iou = calculate_iou_rtdetr(box1, box2)
                
                if iou > iou_threshold:
                    # High overlap - don't keep the lower confidence one
                    should_keep = False
                    logger.debug(f"RT-DETR IoU filter (scaled): Removing {class_name} "
                               f"conf={obj['confidence']:.3f} (IoU={iou:.3f} with "
                               f"conf={kept_obj['confidence']:.3f})")
                    break
            
            if should_keep:
                class_keep.append(obj)
        
        filtered_objects.extend(class_keep)
        logger.debug(f"RT-DETR IoU filter (scaled): {class_name} {len(class_objects)} → {len(class_keep)} detections")
    
    return filtered_objects

def apply_iou_filtering_rtdetr(labels, boxes, scores, class_names, iou_threshold=IOU_THRESHOLD):
    """Apply IoU filtering for RT-DETR numpy arrays"""
    if len(labels) == 0:
        return labels, boxes, scores
    
    # Group detections by class
    class_groups = {}
    for i in range(len(labels)):
        label_id = int(labels[i])
        if 0 <= label_id < len(class_names):
            class_name = class_names[label_id]
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(i)
    
    keep_indices = []
    
    # Process each class separately
    for class_name, indices in class_groups.items():
        if len(indices) == 1:
            # Only one detection for this class, keep it
            keep_indices.extend(indices)
            continue
        
        # For multiple detections, apply IoU filtering
        class_keep = []
        
        # Sort by confidence (highest first)
        indices.sort(key=lambda x: scores[x], reverse=True)
        
        for i in indices:
            should_keep = True
            
            for j in class_keep:
                iou = calculate_iou_rtdetr(boxes[i], boxes[j])
                
                if iou > iou_threshold:
                    # High overlap - don't keep the lower confidence one
                    should_keep = False
                    logger.debug(f"RT-DETR IoU filter: Removing {class_name} "
                               f"conf={scores[i]:.3f} (IoU={iou:.3f} with "
                               f"conf={scores[j]:.3f})")
                    break
            
            if should_keep:
                class_keep.append(i)
        
        keep_indices.extend(class_keep)
        logger.debug(f"RT-DETR IoU filter: {class_name} {len(indices)} → {len(class_keep)} detections")
    
    # Return filtered arrays
    if keep_indices:
        keep_indices = sorted(keep_indices)
        return labels[keep_indices], boxes[keep_indices], scores[keep_indices]
    else:
        return np.array([]), np.array([]), np.array([])

# Initialize local emoji service (replaces HTTP requests to centralized service)
# Load emoji mappings from central API
emoji_mappings = {}

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    logger.info(f"🔄 RT-DETR: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    logger.info(f"✅ RT-DETR: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

def get_emoji(concept: str):
    """Get emoji for a single concept"""
    if not concept:
        return None
    concept_clean = concept.lower().strip()
    return emoji_mappings.get(concept_clean)

# Load emoji mappings on startup
load_emoji_mappings()

class RTDETRModel(nn.Module):
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
            cfg.model.load_state_dict(state)
        
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def load_model():
    """Load RT-DETR model"""
    global model, device
    
    try:
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Log GPU info
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        
        logger.info(f"Using device: {device}")
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        config_path = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
        checkpoint_path = 'rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth'
        
        # Load model with error handling
        try:
            model = RTDETRModel(config_path, checkpoint_path)
            
            # Test model on dummy input before moving to device
            dummy_input = torch.randn(1, 3, 640, 640)
            dummy_size = torch.tensor([[640, 640]])
            with torch.no_grad():
                _ = model(dummy_input, dummy_size)
            logger.info("Model test on CPU successful")
            
            # Move to device
            model = model.to(device)
            model.eval()
            
            # Test on device
            if device.type == 'cuda':
                dummy_input = dummy_input.to(device)
                dummy_size = dummy_size.to(device)
                with torch.no_grad():
                    _ = model(dummy_input, dummy_size)
                torch.cuda.synchronize()
                logger.info("Model test on GPU successful")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            if device.type == 'cuda':
                logger.info("Falling back to CPU")
                device = torch.device('cpu')
                model = RTDETRModel(config_path, checkpoint_path).to(device)
                model.eval()
        
        logger.info("RT-DETR model loaded successfully")
        
        # Using local emoji service
        logger.info("Emoji lookup: Local file mode")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load RT-DETR model: {e}")
        return False

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name using local emoji service (optimized - no HTTP requests)"""
    clean_name = class_name.lower().strip()
    
    try:
        # Use local emoji service instead of HTTP requests
        emoji = get_emoji(clean_name)
        if emoji:
            logger.debug(f"Local emoji service: '{clean_name}' → {emoji}")
            return emoji
        
        logger.debug(f"Local emoji service: no emoji found for '{clean_name}'")
        return None
        
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{clean_name}': {e}")
        return None

def download_image(url: str) -> Optional[str]:
    """Download image from URL and return local path"""
    filepath = None
    
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            return None
            
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            logger.warning(f"URL does not contain an image: {content_type}")
            return None
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            logger.warning(f"Image too large: {content_length} bytes")
            return None
        
        # Save to file
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_FILE_SIZE:
                        logger.warning("Image too large during download")
                        return None
        
        return filepath
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None
    finally:
        # Cleanup on error (success case returns filepath for caller to manage)
        if filepath and not os.path.exists(filepath):
            # File was partially created but removed during error - ensure cleanup
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass

def detect_objects(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Perform object detection on image"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        
        # Transform image
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(image)[None].to(device)
        
        # IMPORTANT: RT-DETR expects the size of the transformed image, not the original
        # Since we resize to 640x640, we need to pass that size
        orig_size = torch.tensor([640, 640])[None].to(device)
        
        # Run inference with minimal CUDA overhead
        with torch.no_grad():
            try:
                outputs = model(im_data, orig_size)
                
                # Clear GPU cache periodically to prevent memory buildup
                if device.type == 'cuda' and torch.cuda.memory_allocated() > 1024**3:  # 1GB
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error during inference: {e}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU inference failed: {e}")
                else:
                    raise
            
        
        labels, boxes, scores = outputs
        
        # Move to CPU and explicitly delete GPU references
        labels_cpu = labels.cpu().numpy()
        boxes_cpu = boxes.cpu().numpy() 
        scores_cpu = scores.cpu().numpy()
        
        # Delete GPU tensors explicitly
        del labels, boxes, scores, outputs, im_data, orig_size
        
        # Use CPU versions
        labels = labels_cpu
        boxes = boxes_cpu
        scores = scores_cpu
        
        # Periodic GPU sync to prevent state corruption (every 10 requests)
        global request_count
        request_count += 1
        if device.type == 'cuda' and request_count % 10 == 0:
            torch.cuda.synchronize()
            logger.debug(f"GPU sync at request {request_count}")
        
        # Filter by confidence only (IoU filtering will happen after scaling)
        if len(scores) > 0:
            valid_indices = scores > CONFIDENCE_THRESHOLD
            labels = labels[valid_indices]
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
        
        # Format results with Mirror Stage emoji enrichment
        objects = []
        
        # Build results first without emojis (fast path)
        for i in range(len(labels)):
            label_id = int(labels[i])
            if 0 <= label_id < len(COCO_CLASSES):
                object_name = COCO_CLASSES[label_id]
                confidence = round(float(scores[i]), 3)
                box = boxes[i].tolist()
                
                # Scale bounding box coordinates from 640x640 back to original image size
                # RT-DETR returns boxes in [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box
                x1 = (x1 / 640) * w
                y1 = (y1 / 640) * h
                x2 = (x2 / 640) * w
                y2 = (y2 / 640) * h
                scaled_box = [x1, y1, x2, y2]
                
                objects.append({
                    'object': object_name,
                    'confidence': confidence,
                    'bbox': scaled_box,
                    'emoji': ''  # Will be filled asynchronously
                })
        
        # Apply IoU filtering in final coordinate space (after scaling)
        objects = apply_iou_filtering_rtdetr_scaled(objects)
        
        # Collect unique object names from filtered results for emoji lookup
        unique_objects = list(set(obj['object'] for obj in objects))
        
        # Look up emojis for each unique object
        if unique_objects:
            try:
                emoji_mappings = {}
                for obj_name in unique_objects:
                    emoji = lookup_emoji(obj_name)
                    if emoji:
                        emoji_mappings[obj_name] = emoji
                
                # Update objects with emojis
                for obj in objects:
                    obj['emoji'] = emoji_mappings.get(obj['object'], '')
            except Exception as e:
                logger.debug(f"Emoji lookup failed: {e}")
                # Objects already have empty emoji strings, so this is graceful
        
        result = {
            'success': True,
            'objects': objects,
            'image_size': {'width': w, 'height': h},
            'model': 'RT-DETR',
            'total_detections': len(objects)
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            try:
                if os.path.exists(image_path) and image_path.startswith(UPLOAD_FOLDER):
                    os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {image_path}: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        
        # Cleanup on error (only for temporary files)
        if cleanup:
            try:
                if os.path.exists(image_path) and image_path.startswith(UPLOAD_FOLDER):
                    os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {image_path}: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'objects': []
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_data = {
        'status': 'healthy',
        'model': 'RT-DETR R50',
        'device': str(device) if device else 'unknown',
        'model_loaded': model is not None,
        'emoji_service': 'local_file'
    }
    
    
    return jsonify(health_data)

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
        new_args = request.args.copy().to_dict()
        new_args['file'] = file_path
        del new_args['file_path']
        
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
                "service": "rtdetr",
                "status": "error",
                "predictions": [],
                "error": {"message": "Must provide either url or file parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        if url and file:
            return jsonify({
                "service": "rtdetr",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both url and file parameters"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Handle URL input
        if url:
            filepath = None
            try:
                # Validate URL
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    return jsonify({
                        "service": "rtdetr",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": "Invalid URL format"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                # Download image
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    return jsonify({
                        "service": "rtdetr",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": "URL does not point to an image"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    os.remove(filepath)
                    return jsonify({
                        "service": "rtdetr",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": "Downloaded file too large"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                # Detect objects using existing function
                result = detect_objects(filepath)
                filepath = None  # detect_objects handles cleanup
                
            except requests.exceptions.RequestException as e:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "service": "rtdetr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Failed to download image"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            except Exception as e:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "service": "rtdetr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Error processing image"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
        
        # Handle file input
        if file:
            # Validate file path
            if not os.path.exists(file):
                return jsonify({
                    "service": "rtdetr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 404
            
            if not allowed_file(file):
                return jsonify({
                    "service": "rtdetr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "File type not allowed"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            # Detect objects directly from file (no cleanup needed - we don't own the file)
            result = detect_objects(file, cleanup=False)
        
        # Process results (common for both URL and file)
        if not result.get('success', False):
            return jsonify({
                "service": "rtdetr",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v3 format - extract objects and create predictions
        objects = result.get('objects', [])
        
        # Create unified prediction format
        predictions = []
        for obj in objects:
            bbox = obj.get('bbox', [])
            prediction = {
                "label": obj.get('object', ''),
                "confidence": round(float(obj.get('confidence', 0)), 3)
            }
            
            # Add bbox if present
            if len(bbox) >= 4:
                prediction["bbox"] = {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1])
                }
            
            # Add emoji if present
            if obj.get('emoji'):
                prediction["emoji"] = obj['emoji']
            
            predictions.append(prediction)
        
        # Return unified v3 format
        return jsonify({
            "service": "rtdetr",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "RT-DETR PyTorch"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V3 API error: {e}")
        return jsonify({
            "service": "rtdetr",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500


if __name__ == '__main__':
    logger.info("Starting RT-DETR Object Detection Service...")
    
    if load_model():
        logger.info(f"Service starting on port {PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        logger.error("Failed to start service: model loading failed")
        sys.exit(1)
