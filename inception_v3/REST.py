#!/usr/bin/env python3
"""
Inception v3 Image Classification REST API Service
Coordination service for Google's Inception v3 model for ImageNet classification.
"""

import os
import json
import uuid
import logging
import random
import time
from typing import List, Dict, Any, Optional

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from PIL import Image
import tensorflow as tf

# Simple emoji lookup - no complex dependencies needed

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load environment variables as strings first
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')

# Validate required environment variables
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Convert to appropriate types after validation
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR) if API_TIMEOUT_STR else 2.0
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() == 'true'


# Configure TensorFlow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
    except RuntimeError as e:
        logger.error(f"GPU memory configuration error: {e}")

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 160


# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
imagenet_classes = []

# Load emoji mappings from local JSON file
emoji_mappings = {}

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    logger.info(f"🔄 Inception: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    logger.info(f"✅ Inception: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

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

# ImageNet classes are handled by TensorFlow's decode_predictions internally
# No separate loading required


def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name using local emoji service (optimized - no HTTP requests)"""
    clean_name = class_name.lower().strip().replace(" ", "_")
    
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

def initialize_inception_model() -> bool:
    """Initialize Inception v3 model using TensorFlow 2.x"""
    global model
    try:
        # Load pre-trained Inception v3 model
        logger.info("Loading Inception v3 model...")
        model = tf.keras.applications.InceptionV3(
            weights='imagenet',
            include_top=True,
            input_shape=(299, 299, 3)
        )
        logger.info("Inception v3 model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load Inception v3 model: {e}")
        return False

def preprocess_image_from_pil(image: Image.Image) -> Optional[tf.Tensor]:
    """Preprocess PIL Image for Inception v3 (no file required)"""
    try:
        # Resize PIL Image directly
        image = image.resize((299, 299), Image.Resampling.LANCZOS)
        
        # Convert PIL Image to array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing PIL image: {e}")
        return None

def process_image_for_classification(image: Image.Image) -> dict:
    """
    Main processing function - takes PIL Image, returns classification data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing
    """
    start_time = time.time()
    
    if not model:
        return {
            "success": False,
            "error": "Inception v3 model not loaded"
        }
    
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        img_array = preprocess_image_from_pil(image)
        if img_array is None:
            return {
                "success": False,
                "error": "Failed to preprocess image"
            }
        
        # Run classification
        classification_start = time.time()
        predictions = model.predict(img_array, verbose=0)
        classification_time = time.time() - classification_start
        
        # Decode predictions with higher top count for better coverage
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(
            predictions, top=100
        )[0]
        
        # Process results with deduplication by both emoji and class name
        classifications = []
        seen_emojis = set()
        seen_classes = set()
        
        for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
            if confidence >= CONFIDENCE_THRESHOLD:
                # Skip if we've already seen this class
                if class_name in seen_classes:
                    continue
                    
                try:
                    emoji = lookup_emoji(class_name)
                    
                    # Only include if we haven't seen this emoji before
                    if emoji not in seen_emojis:
                        classification = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(float(confidence), 3),
                            "rank": i + 1,
                            "emoji": emoji
                        }
                        classifications.append(classification)
                        seen_emojis.add(emoji)
                        seen_classes.add(class_name)
                except RuntimeError as e:
                    return {
                        "success": False,
                        "error": f"Classification failed due to emoji lookup failure: {e}"
                    }
        
        logger.info(f"Found {len(classifications)} classifications in {classification_time:.2f}s")
        
        # Get image dimensions from PIL Image
        image_width, image_height = image.size
        
        return {
            "success": True,
            "data": {
                "classifications": classifications,
                "total_classifications": len(classifications),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                },
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "classification_time": round(classification_time, 3),
                    "framework": "TensorFlow",
                    "model": "Inception v3"
                }
            },
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Classification processing error: {str(e)}")
        return {
            "success": False,
            "error": f"Classification failed: {str(e)}",
            "processing_time": time.time() - start_time
        }

def create_inception_response(data: dict, processing_time: float) -> dict:
    """Create standardized inception v3 response"""
    classifications = data.get('classifications', [])
    
    # Create unified prediction format
    predictions = []
    for classification in classifications:
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "confidence": round(float(classification.get('confidence', 0)), 3),
            "label": classification.get('class_name', '')
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {classification.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
        # Add emoji if present
        if classification.get('emoji'):
            prediction["emoji"] = classification['emoji']
        
        predictions.append(prediction)
    
    return {
        "service": "inception",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "TensorFlow"
            }
        }
    }

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

def classify_image(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Classify image using Inception v3"""
    if not model:
        return {"error": "Inception v3 model not loaded", "status": "error"}
        
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {"error": "Image file not found", "status": "error"}
            
        if not validate_file_size(image_path):
            return {"error": "File too large", "status": "error"}
            
        # Preprocess image
        logger.info(f"Classifying image: {image_path}")
        img_array = preprocess_image(image_path)
        if img_array is None:
            return {"error": "Failed to preprocess image", "status": "error"}
            
        # Run classification
        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        classification_time = time.time() - start_time
        
        # Decode predictions with higher top count for better coverage
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(
            predictions, top=100
        )[0]
        
        # Process results with deduplication by both emoji and class name
        classifications = []
        seen_emojis = set()
        seen_classes = set()
        
        for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
            if confidence >= CONFIDENCE_THRESHOLD:
                # Skip if we've already seen this class (ensures highest confidence per class)
                if class_name in seen_classes:
                    continue
                    
                try:
                    emoji = lookup_emoji(class_name)
                    
                    # Only include if we haven't seen this emoji before
                    # This ensures we get the highest confidence for each unique emoji
                    if emoji not in seen_emojis:
                        classification = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(float(confidence), 3),
                            "rank": i + 1,
                            "emoji": emoji
                        }
                        classifications.append(classification)
                        seen_emojis.add(emoji)
                        seen_classes.add(class_name)
                except RuntimeError as e:
                    raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")
                
        logger.info(f"Found {len(classifications)} classifications in {classification_time:.2f}s")
        
        # Get image dimensions
        try:
            with Image.open(image_path) as pil_img:
                image_width, image_height = pil_img.size
        except Exception:
            image_width = image_height = None
            
        # Build response
        response = {
            "INCEPTION": {
                "classifications": classifications,
                "total_classifications": len(classifications),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                } if image_width and image_height else None,
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "classification_time": round(classification_time, 3),
                    "framework": "TensorFlow",
                    "model": "Inception v3"
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
        logger.error(f"Error classifying image {image_path}: {e}")
        return {"error": f"Classification failed: {str(e)}", "status": "error"}


# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Inception v3 service: CORS enabled for direct browser communication")

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
    if not model:
        return jsonify({
            "status": "unhealthy",
            "reason": "Inception v3 model not loaded",
            "framework": "TensorFlow"
        }), 503
    
    return jsonify({
        "status": "healthy",
        "model_status": "loaded",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "imagenet_classes_loaded": "handled_by_tensorflow",
        "framework": "TensorFlow",
        "model": "Inception v3"
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported ImageNet classes"""
    return jsonify({
        "classes": "handled_by_tensorflow",  # Classes handled by decode_predictions
        "total_classes": 1000,
        "framework": "TensorFlow",
        "model": "Inception v3"
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "inception",
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
            
            # Validate file type
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
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
                    
                    response = requests.get(url, timeout=10, stream=True)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        return error_response("URL does not point to an image")
                    
                    # Check content length if provided
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")
                    
                    from io import BytesIO
                    image_data = BytesIO()
                    for chunk in response.iter_content(chunk_size=8192):
                        image_data.write(chunk)
                        if image_data.tell() > MAX_FILE_SIZE:
                            return error_response("Downloaded file too large")
                    
                    image_data.seek(0)
                    image = Image.open(image_data).convert('RGB')
                    
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                
                if not is_allowed_file(file_path):
                    return error_response("File type not allowed")
                
                if not validate_file_size(file_path):
                    return error_response("File too large")
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_classification(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_inception_response(
            processing_result["data"],
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls new analyze function directly"""
    return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API compatibility endpoint - translates file_path to file parameter"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API compatibility endpoint - translates image_url to url parameter"""
    image_url = request.args.get('image_url')
    
    if image_url:
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        if 'image_url' in new_args:
            del new_args['image_url']
        
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()



if __name__ == '__main__':
    # Initialize services
    logger.info("Starting Inception v3 service...")
    
    model_loaded = initialize_inception_model()
    
    if not model_loaded:
        logger.error("Failed to load Inception v3 model.")
        logger.error("Please ensure TensorFlow is installed with ImageNet weights.")
        logger.error("Service cannot function without model. Exiting.")
        exit(1)
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting Inception v3 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


