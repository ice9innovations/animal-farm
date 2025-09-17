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
import random
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import numpy as np

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

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
    """Load emoji mappings from GitHub, fall back to local cache"""
    global emoji_mappings
    
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 RT-DETR: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            emoji_mappings = response.json()
            logger.info(f"✅ RT-DETR: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  RT-DETR: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
            logger.info(f"✅ RT-DETR: Successfully loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.error(f"❌ RT-DETR: Failed to load local emoji mappings: {local_error}")
        raise Exception(f"Both GitHub and local emoji mappings failed: GitHub download disabled or failed, Local cache={local_error}")

def get_emoji(concept: str):
    """Get emoji for a single concept with underscore normalization"""
    if not concept:
        return None
    # Normalize word format: lowercase with underscores (consistent with ollama-api)
    concept_clean = concept.lower().strip().replace(' ', '_')
    return emoji_mappings.get(concept_clean)

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

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
        # Use local emoji service (get_emoji handles underscore normalization)
        emoji = get_emoji(clean_name)
        if emoji:
            # Show the normalized name in debug output
            normalized_name = clean_name.replace(' ', '_')
            logger.debug(f"Local emoji service: '{clean_name}' → '{normalized_name}' → {emoji}")
            return emoji
        
        logger.debug(f"Local emoji service: no emoji found for '{clean_name}'")
        return None
        
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{clean_name}': {e}")
        return None

def create_rtdetr_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized RT-DETR response with object detections"""
    objects = data.get('objects', [])
    
    # Create unified prediction format
    predictions = []
    for obj in objects:
        bbox = obj.get('bbox', [])
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "label": obj.get('object', ''),
            "confidence": round(float(obj.get('confidence', 0)), 3)
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {obj.get('object', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
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

    # Sort predictions by confidence (highest first)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "service": "rtdetr",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "RT-DETR PyTorch"
            }
        }
    }

def process_image_for_rtdetr(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns RT-DETR detection data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing with PIL Image support
    """
    start_time = time.time()
    
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
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
        
        processing_time = round(time.time() - start_time, 3)
        
        return {
            'success': True,
            'data': {
                'objects': objects,
                'image_size': {'width': w, 'height': h},
                'model': 'RT-DETR',
                'total_detections': len(objects)
            },
            'processing_time': processing_time
        }
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        logger.error(f"Error during object detection: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test if RT-DETR model is actually working
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_result = process_image_for_rtdetr(test_image)
        
        if not test_result.get('success'):
            raise ValueError(f"Model test failed: {test_result.get('error')}")
        
        model_status = "loaded"
        status = "healthy"
        
    except Exception as e:
        model_status = f"error: {str(e)}"
        status = "unhealthy"
        
        return jsonify({
            "status": status,
            "reason": f"RT-DETR model error: {str(e)}",
            "service": "RT-DETR Object Detection"
        }), 503
    
    return jsonify({
        "status": status,
        "service": "RT-DETR Object Detection",
        "model": {
            "name": "RT-DETR R50",
            "status": model_status,
            "device": str(device) if device else "unknown"
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "emoji_service": "local_file",
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /v3/analyze - V3 compatibility",
            "GET /v2/analyze - V2 compatibility (deprecated)",
            "GET /v2/analyze_file - V2 compatibility (deprecated)"
        ]
    })

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
            "service": "rtdetr",
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
                
                if not allowed_file(file_path):
                    return error_response("File type not allowed")
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_rtdetr(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_rtdetr_response(
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
    logger.info("Starting RT-DETR Object Detection Service...")
    
    if load_model():
        logger.info(f"Service starting on port {PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        logger.error("Failed to start service: model loading failed")
        sys.exit(1)
