from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import sys
import numpy as np
import os
import os.path
import requests
import json
import re
import uuid
import logging
import random
import pickle
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')
API_PORT = os.getenv('API_PORT')
API_TIMEOUT = os.getenv('API_TIMEOUT')

# Validate critical environment variables
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT:
    raise ValueError("API_TIMEOUT environment variable is required")

# Convert to appropriate types after validation
API_PORT = int(API_PORT)
API_TIMEOUT = float(API_TIMEOUT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Prediction filtering configuration - easily adjustable
DEFAULT_CONFIDENCE_THRESHOLD = 0.00  #  only return reasonably confident predictions
DEFAULT_MAX_PREDICTIONS = 100         # Safety cap on number of predictions

CONFIDENCE_THRESHOLD_STR = os.getenv('CLIP_CONFIDENCE_THRESHOLD')
MAX_PREDICTIONS_STR = os.getenv('CLIP_MAX_PREDICTIONS')

# Validate CLIP configuration
if not CONFIDENCE_THRESHOLD_STR:
    raise ValueError("CLIP_CONFIDENCE_THRESHOLD environment variable is required")
if not MAX_PREDICTIONS_STR:
    raise ValueError("CLIP_MAX_PREDICTIONS environment variable is required")

# Convert to appropriate types
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR)
MAX_PREDICTIONS = int(MAX_PREDICTIONS_STR)

# Label file configuration - automatically loads all .txt files from labels folder
LABELS_FOLDER = './labels'

# CLIP model configuration - Options: ViT-B/32, ViT-L/14, ViT-L/14@336px
CLIP_MODEL = 'ViT-L/14'  # Larger model for better discrimination (~6-8GB VRAM)
#CLIP_MODEL = 'ViT-B/32'  # Smaller model (~4-6GB VRAM)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Text embedding cache
text_embedding_cache = {}
TEXT_EMBEDDINGS_CACHE_FILE = './text_embeddings_cache.pkl'


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")

logger.info(f"Using device: {device}")

# Load emoji mappings from central API
emoji_mappings = {}

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    logger.info(f"🔄 CLIP: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    logger.info(f"✅ CLIP: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

def get_emoji(concept: str) -> Optional[str]:
    """Get emoji for a single concept"""
    if not concept:
        return None
    concept_clean = concept.lower().strip().replace(' ', '_')
    return emoji_mappings.get(concept_clean)

def load_text_embeddings_cache():
    """Load text embeddings from cache file if it exists"""
    global text_embedding_cache
    
    if os.path.exists(TEXT_EMBEDDINGS_CACHE_FILE):
        try:
            with open(TEXT_EMBEDDINGS_CACHE_FILE, 'rb') as f:
                text_embedding_cache = pickle.load(f)
            logger.info(f"Loaded {len(text_embedding_cache)} text embeddings from cache file")
        except Exception as e:
            logger.error(f"Failed to load text embeddings cache: {e}")
            text_embedding_cache = {}
    else:
        logger.info("No text embeddings cache file found, will create one after first computation")

def save_text_embeddings_cache():
    """Save text embeddings to cache file"""
    try:
        with open(TEXT_EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(text_embedding_cache, f)
        logger.info(f"Saved {len(text_embedding_cache)} text embeddings to cache file")
    except Exception as e:
        logger.error(f"Failed to save text embeddings cache: {e}")

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

# Load emoji mappings on startup
load_emoji_mappings()

def load_labels_from_files() -> List[str]:
    """Load classification labels from all .txt files in the labels/ folder"""
    all_labels = []
    
    # Automatically discover all .txt files in the labels folder
    try:
        txt_files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')]
        txt_files.sort()  # Consistent ordering
        logger.info(f"Found {len(txt_files)} label files: {txt_files}")
    except OSError as e:
        logger.error(f"Could not read labels folder {LABELS_FOLDER}: {e}")
        return []
    
    for filename in txt_files:
        filepath = os.path.join(LABELS_FOLDER, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, ignore empty lines and comments
                file_labels = [
                    line.strip() 
                    for line in f.readlines() 
                    if line.strip() and not line.strip().startswith('#')
                ]
                all_labels.extend(file_labels)
                logger.info(f"Loaded {len(file_labels)} labels from {filepath}")
        except Exception as e:
            logger.error(f"Error loading labels from {filepath}: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in all_labels:
        if label.lower() not in seen:
            seen.add(label.lower())
            unique_labels.append(label)
    
    logger.info(f"Total unique labels loaded: {len(unique_labels)}")
    return unique_labels


# Global variables for model and data
model = None
preprocess = None
labels = None
label_tensor = None
label_features = None  # CACHED TEXT FEATURES

def initialize_clip_model() -> bool:
    """Initialize CLIP model and labels with FP16 optimization and text embedding caching"""
    global model, preprocess, labels, label_tensor, label_features
    try:
        logger.info(f"Loading CLIP model: {CLIP_MODEL}...")
        model, preprocess = clip.load(CLIP_MODEL, device=device)
        
        # Apply FP16 optimization for VRAM savings and speed boost
        if device == "cuda":
            model = model.half()
            logger.info(f"Applied FP16 quantization to {CLIP_MODEL} - 50% VRAM reduction achieved!")
            logger.info(f"Expected VRAM usage: ~4.3GB (down from ~8.7GB)")
        
        logger.info(f"CLIP model {CLIP_MODEL} loaded successfully")
        
        # Load cached embeddings
        load_text_embeddings_cache()
        
        # Load labels from files
        labels = load_labels_from_files()
        
        if not labels:
            logger.error("No labels loaded from files!")
            return False
        
        # Create text descriptions
        labels_desc = [f"a picture of a {label}" for label in labels]
        
        # Check which embeddings need to be computed
        labels_to_compute = []
        cached_features = []
        
        for i, (label, desc) in enumerate(zip(labels, labels_desc)):
            cache_key = desc.lower()  # Use description as cache key
            if cache_key in text_embedding_cache:
                # Convert cached numpy array back to tensor
                cached_tensor = torch.from_numpy(text_embedding_cache[cache_key]).to(device)
                if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                    cached_tensor = cached_tensor.half()
                cached_features.append(cached_tensor)
            else:
                labels_to_compute.append((i, label, desc))
                cached_features.append(None)  # Placeholder
        
        # Compute missing embeddings if any
        if labels_to_compute:
            logger.info(f"Computing embeddings for {len(labels_to_compute)} new labels...")
            
            # Tokenize only the missing labels
            missing_descs = [desc for _, _, desc in labels_to_compute]
            label_tensor = clip.tokenize(missing_descs).to(device)
            
            # Compute text features for missing labels
            with torch.no_grad():
                if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                    with torch.cuda.amp.autocast():
                        computed_features = model.encode_text(label_tensor)
                else:
                    computed_features = model.encode_text(label_tensor)
            
            # Cache the computed features and fill in the placeholders
            for i, (original_idx, label, desc) in enumerate(labels_to_compute):
                feature_tensor = computed_features[i]
                cached_features[original_idx] = feature_tensor
                
                # Cache as numpy array for persistence
                cache_key = desc.lower()
                text_embedding_cache[cache_key] = feature_tensor.cpu().numpy()
            
            # Save updated cache
            save_text_embeddings_cache()
        else:
            logger.info(f"All {len(labels)} label embeddings loaded from cache")
        
        # Stack all features into final tensor
        label_features = torch.stack(cached_features)
        
        logger.info(f"Pre-computed text features for {len(labels)} labels with caching - memory leak fixed!")
        logger.info(f"Initialized {len(labels)} classification labels from files")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize CLIP model: {e}")
        return False


def lookup_emoji(tag: str, score: float) -> List[Dict[str, Any]]:
    """Look up emoji for a given tag with score using local emoji service (optimized - no HTTP requests)"""
    # Convert tag to string - let get_emoji() handle all normalization
    original_tag = str(tag).strip()
    
    try:
        # Use local emoji service instead of HTTP requests
        emoji = get_emoji(original_tag)
        if emoji:
            logger.info(f"Local emoji service: '{original_tag}' → {emoji} (confidence: {score:.3f})")
            return [{
                "keyword": original_tag,
                "emoji": emoji,
                "confidence": round(float(score), 3)
            }]
        
        logger.debug(f"Local emoji service: no emoji found for '{original_tag}'")
        return []
        
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{original_tag}': {e}")
        return []

def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    """Preprocess image for CLIP model"""
    try:
        image = Image.open(image_path).convert("RGB")
        return preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def compute_similarity(image_path: str) -> Optional[torch.Tensor]:
    """Compute similarity between image and text labels"""
    logger.info("Checking model and label features...")
    if model is None:
        logger.error("Model not initialized")
        return None
    if label_features is None:
        logger.error("Label features not initialized")
        return None
        
    try:
        with torch.no_grad():
            logger.info("Preprocessing image...")
            image_tensor = preprocess_image(image_path)
            if image_tensor is None:
                return None
                
            # Convert image tensor to half precision if model is FP16
            if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                image_tensor = image_tensor.half()
                
            logger.info("Encoding image features...")
            # Use autocast for FP16 inference stability - ONLY encode image (text features cached)
            if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(image_tensor)
            else:
                image_features = model.encode_image(image_tensor)
            
            # Use pre-computed cached label_features (no more text encoding per request!)
                
            logger.info("Computing similarity...")
            similarity = (image_features @ label_features.T).softmax(dim=-1)
            logger.info("Similarity computed successfully")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return similarity
        
    except Exception as e:
        logger.error(f"Error computing similarity for {image_path}: {e}")
        return None

def compute_caption_similarity(image_path: str, caption: str) -> Optional[float]:
    """Compute similarity between image and arbitrary caption text with caching"""
    logger.info(f"Computing caption similarity for: '{caption}'")
    
    if model is None:
        logger.error("Model not initialized")
        return None
        
    try:
        with torch.no_grad():
            # Preprocess image
            logger.info("Preprocessing image...")
            image_tensor = preprocess_image(image_path)
            if image_tensor is None:
                return None
                
            # Convert image tensor to half precision if model is FP16
            if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                image_tensor = image_tensor.half()
                
            # Check cache for text features first
            cache_key = caption.lower().strip()
            if cache_key in text_embedding_cache:
                logger.info(f"Using cached text features for caption: '{caption}'")
                text_features = torch.from_numpy(text_embedding_cache[cache_key]).to(device)
                if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                    text_features = text_features.half()
                # Ensure proper shape (add batch dimension if needed)
                if len(text_features.shape) == 1:
                    text_features = text_features.unsqueeze(0)
            else:
                # Tokenize caption text - try raw caption for better discrimination
                logger.info("Tokenizing and encoding caption...")
                logger.info(f"Using raw caption: '{caption}'")
                text_tokens = clip.tokenize([caption]).to(device)
                
                # Encode text with autocast for FP16 stability
                if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                    with torch.cuda.amp.autocast():
                        text_features = model.encode_text(text_tokens)
                else:
                    text_features = model.encode_text(text_tokens)
                
                # Cache the text features
                text_embedding_cache[cache_key] = text_features.cpu().numpy()
                # Save cache periodically (but not every single request to avoid I/O overhead)
                if len(text_embedding_cache) % 10 == 0:  # Save every 10 new entries
                    save_text_embeddings_cache()
                
            logger.info("Encoding image features...")
            # Encode image with autocast for FP16 stability
            if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(image_tensor)
            else:
                image_features = model.encode_image(image_tensor)
            
            # Normalize features (important for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            logger.info("Computing similarity...")
            similarity = (image_features @ text_features.T).item()
            
            logger.info(f"Caption similarity computed: {similarity:.3f}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return similarity
        
    except Exception as e:
        logger.error(f"Error computing caption similarity for {image_path}: {e}")
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

def handle_image_input(url: str = None, file: str = None) -> Dict[str, Any]:
    """
    Handle image input from either URL or file path
    Returns: {"success": bool, "filepath": str, "cleanup": bool, "error": str}
    """
    # Validate input - exactly one parameter must be provided
    if not url and not file:
        return {"success": False, "error": "Must provide either url or file parameter"}
    
    if url and file:
        return {"success": False, "error": "Cannot provide both url and file parameters"}
    
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
                raise ValueError("Downloaded file too large")
            
            return {"success": True, "filepath": filepath, "cleanup": True, "error": None}
            
        except Exception as e:
            logger.error(f"Error processing image URL {url}: {e}")
            # Cleanup on error
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file {filepath}: {cleanup_error}")
            
            return {"success": False, "error": f"Failed to process image URL: {str(e)}"}
    
    # Handle file path input
    elif file:
        # Validate file path
        if not os.path.exists(file):
            return {"success": False, "error": f"File not found: {file}"}
        
        if not is_allowed_file(file):
            return {"success": False, "error": "File type not allowed"}
        
        if not validate_file_size(file):
            return {"success": False, "error": "File too large"}
        
        return {"success": True, "filepath": file, "cleanup": False, "error": None}


def classify_image(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Classify image using CLIP model - simplified to match working version"""
    if not model or not labels:
        return {"error": "Model not initialized", "status": "error"}
        
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {"error": "Image file not found", "status": "error"}
            
        if not validate_file_size(image_path):
            return {"error": "File too large", "status": "error"}
            
        # Use the exact same logic as the working version
        logger.info("Computing similarities...")
        similarity_scores = compute_similarity(image_path)
        if similarity_scores is None:
            return {"error": "Failed to compute similarities", "status": "error"}
            
        logger.info(f"Getting predictions above threshold {CONFIDENCE_THRESHOLD}...")
        
        # Get all scores and indices, sorted by confidence
        scores_sorted, indices_sorted = similarity_scores[0].sort(descending=True)
        
        predictions = []
        all_emoji_matches = []
        
        for i, (score, idx) in enumerate(zip(scores_sorted, indices_sorted)):
            confidence = score.item()
            
            # Stop if below threshold
            if confidence < CONFIDENCE_THRESHOLD:
                break
                
            # Stop if we've hit the max limit
            if len(predictions) >= MAX_PREDICTIONS:
                break
            
            label = labels[idx]
            predictions.append({"label": label, "confidence": round(confidence, 3)})
            
            # Look up emoji for each prediction (fails loudly only if Mirror Stage unavailable)
            try:
                emoji_match = lookup_emoji(label, confidence)
                if emoji_match:
                    all_emoji_matches.extend(emoji_match)
                else:
                    logger.debug(f"No emoji found for label '{label}'")
            except RuntimeError as e:
                logger.error(f"Mirror Stage service failure for '{label}': {e}")
                raise RuntimeError(f"Classification failed due to Mirror Stage service failure: {e}")
            
            logger.info(f"Prediction {i+1}: {label} ({confidence:.3f})")
        
        logger.info(f"Returned {len(predictions)} predictions above threshold {CONFIDENCE_THRESHOLD}")
        
        # Return simple format for V2 processing
        response = {"predictions": predictions, "emoji_matches": all_emoji_matches, "status": "success"}
        
        # Cleanup (only for temporary files)
        if cleanup:
            try:
                if os.path.exists(image_path) and image_path.startswith(UPLOAD_FOLDER):
                    os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {image_path}: {e}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error classifying image {image_path}: {e}")
        return {"error": f"Classification failed: {str(e)}", "status": "error"}

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("CLIP service: CORS enabled for direct browser communication")

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
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "device": str(device),
        "num_labels": len(labels) if labels else 0
    })

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    import time
    start_time = time.time()
    
    try:
        # Get input parameters - support both url and file
        url = request.args.get('url')
        file = request.args.get('file')
        
        # Handle image input using shared helper
        image_result = handle_image_input(url=url, file=file)
        
        if not image_result["success"]:
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": image_result["error"]},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        filepath = image_result["filepath"]
        cleanup = image_result["cleanup"]
        
        try:
            # Classify using existing function
            result = classify_image(filepath, cleanup=cleanup)
            
        except Exception as e:
            logger.error(f"Error classifying image {filepath}: {e}")
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": f"Classification failed: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        finally:
            # Ensure cleanup of temporary file if needed
            if cleanup and filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temporary file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {filepath}: {e}")
        
        # Common processing for both input types
        if result.get('status') == 'error':
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Classification failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v3 format (same as v2 format for CLIP)
        raw_predictions = result.get('predictions', [])
        emoji_matches = result.get('emoji_matches', [])
        
        # Create emoji map for quick lookup
        emoji_map = {}
        for match in emoji_matches:
            emoji_map[match["keyword"]] = match["emoji"]
        
        # Create unified prediction format
        predictions = []
        for item in raw_predictions:
            is_shiny, shiny_roll = check_shiny()
            
            prediction = {
                "label": item.get('label', ''),
                "confidence": round(float(item.get('confidence', 0)), 3)
            }
            
            # Add shiny flag only for shiny detections
            if is_shiny:
                prediction["shiny"] = True
                logger.info(f"✨ SHINY {item.get('label', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
            
            # Add emoji if found
            label = item.get('label', '')
            if label in emoji_map:
                prediction["emoji"] = emoji_map[label]
            
            predictions.append(prediction)
        
        return jsonify({
            "service": "clip",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "OpenAI"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V3 unified API error: {e}")
        return jsonify({
            "service": "clip",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v3/score', methods=['GET'])
def score_caption_v3():
    """Score caption similarity against image using CLIP"""
    import time
    start_time = time.time()
    
    try:
        # Get input parameters - same pattern as /v3/analyze
        caption = request.args.get('caption')
        url = request.args.get('url')
        file = request.args.get('file')
        
        # Validate caption parameter
        if not caption or not isinstance(caption, str) or not caption.strip():
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": "Must provide non-empty caption string"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        caption = caption.strip()
        
        # Handle image input using shared helper
        image_result = handle_image_input(url=url, file=file)
        
        if not image_result["success"]:
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": image_result["error"]},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        filepath = image_result["filepath"]
        cleanup = image_result["cleanup"]
        
        try:
            # Check model availability
            if model is None:
                return jsonify({
                    "service": "clip",
                    "status": "error",
                    "similarity_score": None,
                    "error": {"message": "CLIP model not initialized"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Compute similarity score
            similarity_score = compute_caption_similarity(filepath, caption)
            
            if similarity_score is None:
                return jsonify({
                    "service": "clip",
                    "status": "error",
                    "similarity_score": None,
                    "error": {"message": "Failed to compute similarity score"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Determine image source for response
            image_source = "url" if url else "file"
            
            return jsonify({
                "service": "clip",
                "status": "success",
                "similarity_score": round(float(similarity_score), 3),
                "caption": caption,
                "image_source": image_source,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "OpenAI",
                        "model": CLIP_MODEL
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error computing caption similarity: {e}")
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": f"Similarity computation failed: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        finally:
            # Ensure cleanup of temporary file if needed
            if cleanup and filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temporary file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {filepath}: {e}")
        
    except Exception as e:
        logger.error(f"V3 caption scoring error: {e}")
        return jsonify({
            "service": "clip",
            "status": "error",
            "similarity_score": None,
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze_file/', methods=['GET'])
@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
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

@app.route('/v2/analyze/', methods=['GET'])
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
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


if __name__ == '__main__':
    # Initialize model and emoji data
    logger.info("Starting CLIP service...")
    
    model_loaded = initialize_clip_model()
    
    if not model_loaded:
        logger.error("Failed to load CLIP model. Service will run but classification will fail.")
        logger.error("Please ensure CLIP is installed: pip install clip-by-openai")
        
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting CLIP service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"CLIP model: {CLIP_MODEL}")
    logger.info(f"Model loaded: {model_loaded}")
    if labels:
        logger.info(f"Classification labels: {len(labels)}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
