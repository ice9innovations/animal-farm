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
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')  # Must be set in .env
API_PORT = int(os.getenv('API_PORT'))  # Must be set in .env
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '2.0'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
PRIVATE = os.getenv('PRIVATE', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', '80'))

# Prediction filtering configuration - easily adjustable
DEFAULT_CONFIDENCE_THRESHOLD = 0.01  # 5% - only return reasonably confident predictions
DEFAULT_MAX_PREDICTIONS = 10         # Safety cap on number of predictions

CONFIDENCE_THRESHOLD = float(os.getenv('CLIP_CONFIDENCE_THRESHOLD', str(DEFAULT_CONFIDENCE_THRESHOLD)))
MAX_PREDICTIONS = int(os.getenv('CLIP_MAX_PREDICTIONS', str(DEFAULT_MAX_PREDICTIONS)))

# Label file configuration - easily add/remove categories
LABELS_FOLDER = './labels'
LABEL_FILES = [
    'labels_coco.txt'         # Standard COCO dataset objects
    #'labels_extra.txt',      # Additional useful labels  
    #'labels_people.txt',     # People, professions, and roles
    #'labels_animals.txt',    # Animals with clear emojis
    #'labels_foods.txt',      # Foods & drinks with emojis
    #'labels_weather.txt',    # Weather & nature with emojis
    #'labels_emotions.txt',   # Facial expressions & emotions
    #'labels_transport.txt',  # Vehicles & transportation
    #'labels_sports.txt',     # Sports & activities
    #'labels_music.txt',      # Musical instruments & music
    #'labels_objects.txt',    # Common objects with emojis
    #'labels_symbols.txt',    # Symbols, flags, and signs
    #'labels_plants.txt',     # Plants, flowers, trees
    #'labels_objectnet.txt'   # Too many non-emoji items
]

# CLIP model configuration - Options: ViT-B/32, ViT-L/14, ViT-L/14@336px
#CLIP_MODEL = 'ViT-L/14'  # Upgraded from ViT-B/32 for better accuracy (~4-6GB VRAM)
CLIP_MODEL = 'ViT-B/32'  # Upgraded from ViT-B/32 for better accuracy (~4-6GB VRAM)

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
    concept_clean = concept.lower().strip()
    return emoji_mappings.get(concept_clean)

# Load emoji mappings on startup
load_emoji_mappings()

def load_labels_from_files() -> List[str]:
    """Load classification labels from text files in the labels/ folder"""
    all_labels = []
    
    for filename in LABEL_FILES:
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
        except FileNotFoundError:
            logger.warning(f"Label file {filepath} not found, skipping")
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
    """Initialize CLIP model and labels with FP16 optimization"""
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
        
        # Load labels from files
        labels = load_labels_from_files()
        
        if not labels:
            logger.error("No labels loaded from files!")
            return False
        
        # Create text descriptions
        labels_desc = [f"a picture of a {label}" for label in labels]
        
        # Tokenize labels (convert to half precision if model is FP16)
        label_tensor = clip.tokenize(labels_desc).to(device)
        if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
            # No need to convert text tokens to half - CLIP handles this internally
            pass
        
        # PRE-COMPUTE TEXT FEATURES ONCE (MEMORY LEAK FIX)
        logger.info("Pre-computing text features to prevent memory leak...")
        with torch.no_grad():
            if device == "cuda" and hasattr(model, 'dtype') and model.dtype == torch.float16:
                with torch.cuda.amp.autocast():
                    label_features = model.encode_text(label_tensor)
            else:
                label_features = model.encode_text(label_tensor)
        
        logger.info(f"Pre-computed text features for {len(labels)} labels - memory leak fixed!")
        logger.info(f"Initialized {len(labels)} classification labels from files")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize CLIP model: {e}")
        return False


def lookup_emoji(tag: str, score: float) -> List[Dict[str, Any]]:
    """Look up emoji for a given tag with score using local emoji service (optimized - no HTTP requests)"""
    # Convert tag to string and normalize
    original_tag = str(tag).strip().replace(",", "").lower()
    
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

def build_response(predictions: List[Dict[str, Any]], emoji_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build JSON response for CLIP results - now with multiple predictions"""
    result = {"CLIP": []}
    
    # Create a map of labels to emojis for quick lookup
    emoji_map = {}
    for match in emoji_matches:
        # The keyword in emoji_matches is the normalized form
        emoji_map[match["keyword"]] = match["emoji"]
    
    # Add each prediction as a separate vote
    for pred in predictions:
        label = pred["label"]
        normalized_label = label.replace(" ", "_")
        
        # Each prediction gets its own entry in the CLIP array
        prediction_entry = {
            "keyword": label,
            "confidence": str(pred["confidence"])
        }
        
        # Add emoji if found
        if normalized_label in emoji_map:
            prediction_entry["emoji"] = emoji_map[normalized_label]
        else:
            prediction_entry["emoji"] = ""
            
        result["CLIP"].append(prediction_entry)
    
    result["status"] = "success"
    return result

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
        
        # Build response with threshold-filtered predictions
        response = build_response(predictions, all_emoji_matches)
        
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
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        if not is_allowed_file(file_path):
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": "File type not allowed"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Classify directly from file (no cleanup needed - we don't own the file)
        result = classify_image(file_path, cleanup=False)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "clip",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Classification failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        clip_data = result.get('CLIP', [])
        
        # Create unified prediction format
        predictions = []
        for item in clip_data:
            # Ensure confidence is properly normalized to 0-1 scale
            raw_confidence = item.get('confidence', 0)
            if isinstance(raw_confidence, str):
                raw_confidence = float(raw_confidence)
            
            prediction = {
                "type": "classification",
                "label": item.get('keyword', ''),
                "confidence": round(float(raw_confidence), 3)  # Normalize to 0-1 scale
            }
            
            # Add emoji if present
            if item.get('emoji'):
                prediction["emoji"] = item['emoji']
            
            predictions.append(prediction)
        
        return jsonify({
            "service": "clip",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "CLIP",
                    "framework": "OpenAI"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V2 file analysis error: {e}")
        return jsonify({
            "service": "clip",
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
                "service": "clip",
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
            
            # Classify using existing function
            result = classify_image(filepath)
            filepath = None  # classify_image handles cleanup
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "clip",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Classification failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            clip_data = result.get('CLIP', [])
            
            # Create unified prediction format
            predictions = []
            for item in clip_data:
                # Ensure confidence is properly normalized to 0-1 scale
                raw_confidence = item.get('confidence', 0)
                if isinstance(raw_confidence, str):
                    raw_confidence = float(raw_confidence)
                
                prediction = {
                    "type": "classification",
                    "label": item.get('keyword', ''),
                    "confidence": round(float(raw_confidence), 3)  # Normalize to 0-1 scale
                }
                
                # Add emoji if present
                if item.get('emoji'):
                    prediction["emoji"] = item['emoji']
                
                predictions.append(prediction)
            
            return jsonify({
                "service": "clip",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "CLIP",
                        "framework": "OpenAI"
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image URL {image_url}: {e}")
            return jsonify({
                "service": "clip",
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
            "service": "clip",
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
                    
                result = classify_image(filepath)
                filepath = None  # classify_image handles cleanup
                if "error" in result:
                    return jsonify(result), 500
                return jsonify(result), 200
                
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
                
            result = classify_image(path)
            if "error" in result:
                return jsonify(result), 500
            return jsonify(result), 200
            
        else:
            # Return HTML form
            try:
                with open('form.html', 'r') as file:
                    html = file.read()
            except FileNotFoundError:
                html = f'''<!DOCTYPE html>
<html>
<head><title>CLIP Image Classification</title></head>
<body>
<h1>CLIP Image Classification Service</h1>
<form enctype="multipart/form-data" action="" method="POST">
    <input type="hidden" name="MAX_FILE_SIZE" value="{MAX_FILE_SIZE}" />
    <p>Upload an image file:</p>
    <input name="uploadedfile" type="file" accept="image/*" required /><br /><br />
    <input type="submit" value="Classify Image" />
</form>
<p>Supported formats: {', '.join(ALLOWED_EXTENSIONS)}</p>
<p>Max file size: {MAX_FILE_SIZE // (1024*1024)}MB</p>
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
                
            result = classify_image(filepath)
            filepath = None  # classify_image handles cleanup
            if "error" in result:
                return jsonify(result), 500
            return jsonify(result), 200
            
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
