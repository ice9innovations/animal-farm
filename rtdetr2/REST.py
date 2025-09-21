#!/usr/bin/env python3
"""
RT-DETRv2 Object Detection REST API Service
Provides real-time object detection using RT-DETRv2 (Real-Time Detection Transformer v2).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops
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
MODEL_SIZE_STR = os.getenv('MODEL_SIZE', 'rtdetrv2_s')

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
MODEL_SIZE = MODEL_SIZE_STR  # rtdetrv2_s, rtdetrv2_m_r34, rtdetrv2_m_r50, rtdetrv2_l, rtdetrv2_x

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add RT-DETRv2 to path
rtdetrv2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RT-DETRv2')
sys.path.insert(0, rtdetrv2_path)

# Configuration
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_DETECTIONS = 100  # Maximum number of detections per image

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

# Note: Manual IoU filtering removed - now handled by proper NMS in postprocessor
# IOU_THRESHOLD = 0.3  # Lowered from 0.45 for better duplicate detection

# Old manual IoU filtering functions - no longer needed since NMS is handled properly in postprocessor
# def calculate_iou_rtdetrv2(box1, box2) -> float: ...
# def apply_iou_filtering_rtdetrv2_scaled(objects, iou_threshold=0.3): ...

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
            logger.info(f"🔄 RT-DETRv2: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()

            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)

            emoji_mappings = response.json()
            logger.info(f"✅ RT-DETRv2: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  RT-DETRv2: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")

    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
            logger.info(f"✅ RT-DETRv2: Successfully loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.error(f"❌ RT-DETRv2: Failed to load local emoji mappings: {local_error}")
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

class RTDETRPostProcessorWithNMS(nn.Module):
    """Custom RT-DETR postprocessor that adds proper NMS"""
    def __init__(self, num_classes=80, num_top_queries=300, nms_threshold=0.5, score_threshold=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

    def forward(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # Convert boxes from cxcywh to xyxy format
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Get scores using focal loss approach
        scores = F.sigmoid(logits)

        batch_size = scores.shape[0]
        results = []

        for i in range(batch_size):
            # Get scores and boxes for this image
            img_scores = scores[i]  # [num_queries, num_classes]
            img_boxes = bbox_pred[i]  # [num_queries, 4]

            # Find best class for each detection
            max_scores, labels = img_scores.max(dim=-1)  # [num_queries]

            # Filter by confidence threshold
            valid_mask = max_scores > self.score_threshold
            valid_scores = max_scores[valid_mask]
            valid_labels = labels[valid_mask]
            valid_boxes = img_boxes[valid_mask]

            if len(valid_scores) == 0:
                # No valid detections
                results.append((
                    torch.zeros(0, dtype=torch.long, device=scores.device),
                    torch.zeros(0, 4, device=scores.device),
                    torch.zeros(0, device=scores.device)
                ))
                continue

            # Apply NMS (class-agnostic for simplicity)
            keep_indices = torchvision.ops.nms(valid_boxes, valid_scores, self.nms_threshold)

            # Keep only top detections after NMS
            if len(keep_indices) > self.num_top_queries:
                # Sort by score and keep top queries
                sorted_indices = torch.argsort(valid_scores[keep_indices], descending=True)
                keep_indices = keep_indices[sorted_indices[:self.num_top_queries]]

            final_labels = valid_labels[keep_indices]
            final_boxes = valid_boxes[keep_indices]
            final_scores = valid_scores[keep_indices]

            results.append((final_labels, final_boxes, final_scores))

        # For single image inference, just return the first result
        if batch_size == 1:
            return results[0]
        else:
            # For batch inference, pad to same length
            max_detections = max(len(r[0]) for r in results)
            padded_results = []
            for labels, boxes, scores in results:
                if len(labels) < max_detections:
                    pad_size = max_detections - len(labels)
                    labels = torch.cat([labels, torch.zeros(pad_size, dtype=torch.long, device=labels.device)])
                    boxes = torch.cat([boxes, torch.zeros(pad_size, 4, device=boxes.device)])
                    scores = torch.cat([scores, torch.zeros(pad_size, device=scores.device)])
                padded_results.append((labels, boxes, scores))

            labels = torch.stack([r[0] for r in padded_results])
            boxes = torch.stack([r[1] for r in padded_results])
            scores = torch.stack([r[2] for r in padded_results])
            return labels, boxes, scores

def load_model():
    """Load RT-DETRv2 model using torch.hub"""
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

        # Load RT-DETRv2 model using torch.hub
        logger.info(f"Loading RT-DETRv2 model: {MODEL_SIZE}")

        # Use torch.hub to load the model
        base_model = torch.hub.load(rtdetrv2_path, MODEL_SIZE, source='local', pretrained=True)

        # Replace the postprocessor with our NMS-enabled version
        logger.info("Replacing postprocessor with NMS-enabled version")

        class ModelWithNMS(nn.Module):
            def __init__(self, base_model, device):
                super().__init__()
                self.model = base_model.model
                self.postprocessor = RTDETRPostProcessorWithNMS(
                    num_classes=80,
                    num_top_queries=100,  # Reduced from 300
                    nms_threshold=0.5,
                    score_threshold=CONFIDENCE_THRESHOLD
                )

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        model = ModelWithNMS(base_model, device)

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

        logger.info(f"RT-DETRv2 model ({MODEL_SIZE}) loaded successfully")

        # Using local emoji service
        logger.info("Emoji lookup: Local file mode")

        return True
    except Exception as e:
        logger.error(f"Failed to load RT-DETRv2 model: {e}")
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

def create_rtdetrv2_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized RT-DETRv2 response with object detections"""
    objects = data.get('objects', [])

    # Create unified prediction format
    predictions = []
    for obj in objects:
        bbox = obj.get('bbox', [])
        is_shiny, shiny_roll = check_shiny()

        prediction = {
            "label": obj.get('object', '').replace(' ', '_'),
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
        "service": "rtdetrv2",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "RT-DETRv2 PyTorch",
                "model_size": MODEL_SIZE
            }
        }
    }

def process_image_for_rtdetrv2(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns RT-DETRv2 detection data
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

        # Transform image (RT-DETRv2 uses standard 640x640)
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(image)[None].to(device)

        # RT-DETRv2 expects the original image size for postprocessing
        orig_size = torch.tensor([w, h])[None].to(device)

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

        # Format results with emoji enrichment
        objects = []

        # Build results first without emojis (fast path)
        for i in range(len(labels)):
            label_id = int(labels[i])
            if 0 <= label_id < len(COCO_CLASSES):
                object_name = COCO_CLASSES[label_id]
                confidence = round(float(scores[i]), 3)
                box = boxes[i].tolist()

                # RT-DETRv2 returns boxes already scaled to original image size in [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box
                scaled_box = [x1, y1, x2, y2]

                objects.append({
                    'object': object_name,
                    'confidence': confidence,
                    'bbox': scaled_box,
                    'emoji': ''  # Will be filled asynchronously
                })

        # Note: Removed manual IoU filtering - should be handled by proper NMS in postprocessor

        # Collect unique object names from filtered results for emoji lookup
        unique_objects = list(set(obj['object'] for obj in objects))

        # Look up emojis for each unique object
        if unique_objects:
            try:
                emoji_mappings_local = {}
                for obj_name in unique_objects:
                    emoji = lookup_emoji(obj_name)
                    if emoji:
                        emoji_mappings_local[obj_name] = emoji

                # Update objects with emojis
                for obj in objects:
                    obj['emoji'] = emoji_mappings_local.get(obj['object'], '')
            except Exception as e:
                logger.debug(f"Emoji lookup failed: {e}")
                # Objects already have empty emoji strings, so this is graceful

        processing_time = round(time.time() - start_time, 3)

        return {
            'success': True,
            'data': {
                'objects': objects,
                'image_size': {'width': w, 'height': h},
                'model': f'RT-DETRv2 ({MODEL_SIZE})',
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
    # Test if RT-DETRv2 model is actually working
    try:
        if model is None:
            raise ValueError("Model not loaded")

        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_result = process_image_for_rtdetrv2(test_image)

        if not test_result.get('success'):
            raise ValueError(f"Model test failed: {test_result.get('error')}")

        model_status = "loaded"
        status = "healthy"

    except Exception as e:
        model_status = f"error: {str(e)}"
        status = "unhealthy"

        return jsonify({
            "status": status,
            "reason": f"RT-DETRv2 model error: {str(e)}",
            "service": "RT-DETRv2 Object Detection"
        }), 503

    return jsonify({
        "status": status,
        "service": "RT-DETRv2 Object Detection",
        "model": {
            "name": f"RT-DETRv2 ({MODEL_SIZE})",
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
            "service": "rtdetrv2",
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
        processing_result = process_image_for_rtdetrv2(image)

        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        # Step 4: Create response
        response = create_rtdetrv2_response(
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
    logger.info("Starting RT-DETRv2 Object Detection Service...")

    if load_model():
        logger.info(f"Service starting on port {PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        logger.error("Failed to start service: model loading failed")
        sys.exit(1)