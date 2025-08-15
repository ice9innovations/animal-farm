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

def preprocess_image(image_path: str) -> Optional[tf.Tensor]:
    """Preprocess image for Inception v3"""
    try:
        # Load and preprocess image
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(299, 299)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
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
    model_status = "loaded" if model else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
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

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API compatibility endpoint - translates file_path to file parameter"""
    # Get file_path parameter and translate to file parameter for V3
    file_path = request.args.get('file_path')
    if not file_path:
        return jsonify({
            "metadata": {"processing_time": 0.001},
            "predictions": [],
            "service": "inception",
            "status": "error",
            "error": {"message": "Missing file_path parameter"}
        }), 400
    
    # Use Flask's test_request_context to call V3 endpoint internally
    with app.test_request_context(f'/v3/analyze?file={file_path}'):
        return analyze_v3()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API compatibility endpoint - translates image_url to url parameter"""
    # Get image_url parameter and translate to url parameter for V3
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({
            "metadata": {"processing_time": 0.001},
            "predictions": [],
            "service": "inception",
            "status": "error",
            "error": {"message": "Missing image_url parameter"}
        }), 400
    
    # Use Flask's test_request_context to call V3 endpoint internally
    with app.test_request_context(f'/v3/analyze?url={image_url}'):
        return analyze_v3()

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """V3 API unified endpoint - accepts both URL and file path"""
    import time
    start_time = time.time()
    
    try:
        # Get parameters
        image_url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input parameters
        if not image_url and not file_path:
            return jsonify({
                "metadata": {"processing_time": round(time.time() - start_time, 3)},
                "predictions": [],
                "service": "inception",
                "status": "error",
                "error": {"message": "Either 'url' or 'file' parameter is required"}
            }), 400
            
        if image_url and file_path:
            return jsonify({
                "metadata": {"processing_time": round(time.time() - start_time, 3)},
                "predictions": [],
                "service": "inception", 
                "status": "error",
                "error": {"message": "Cannot provide both 'url' and 'file' parameters"}
            }), 400
        
        # Handle URL input
        if image_url:
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
                
                # Validate file size
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    os.remove(filepath)
                    raise ValueError("Downloaded file too large")
                
                # Classify using existing function
                result = classify_image(filepath)
                filepath = None  # classify_image handles cleanup
                
            except Exception as e:
                logger.error(f"Error processing image URL {image_url}: {e}")
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                return jsonify({
                    "metadata": {"processing_time": round(time.time() - start_time, 3)},
                    "predictions": [],
                    "service": "inception",
                    "status": "error",
                    "error": {"message": f"Failed to process image: {str(e)}"}
                }), 500
        
        # Handle file path input
        else:  # file_path
            # Validate file path
            if not os.path.exists(file_path):
                return jsonify({
                    "metadata": {"processing_time": round(time.time() - start_time, 3)},
                    "predictions": [],
                    "service": "inception",
                    "status": "error",
                    "error": {"message": f"File not found: {file_path}"}
                }), 404
            
            if not is_allowed_file(file_path):
                return jsonify({
                    "metadata": {"processing_time": round(time.time() - start_time, 3)},
                    "predictions": [],
                    "service": "inception",
                    "status": "error",
                    "error": {"message": "File type not allowed"}
                }), 400
            
            # Classify directly from file (no cleanup needed - we don't own the file)
            result = classify_image(file_path, cleanup=False)
        
        # Handle classification errors
        if result.get('status') == 'error':
            return jsonify({
                "metadata": {"processing_time": round(time.time() - start_time, 3)},
                "predictions": [],
                "service": "inception",
                "status": "error",
                "error": {"message": result.get('error', 'Classification failed')}
            }), 500
        
        # Convert to V3 format
        inception_data = result.get('INCEPTION', {})
        classifications = inception_data.get('classifications', [])
        image_dims = inception_data.get('image_dimensions', {})
        
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
        
        return jsonify({
            "metadata": {
                "model_info": {
                    "framework": "TensorFlow"
                },
                "processing_time": round(time.time() - start_time, 3)
            },
            "predictions": predictions,
            "service": "inception",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"V3 API error: {e}")
        return jsonify({
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
            "predictions": [],
            "service": "inception",
            "status": "error",
            "error": {"message": f"Internal error: {str(e)}"}
        }), 500


if __name__ == '__main__':
    # Initialize services
    logger.info("Starting Inception v3 service...")
    
    model_loaded = initialize_inception_model()
    
    
    if not model_loaded:
        logger.error("Failed to load Inception v3 model. Service will run but classification will fail.")
        logger.error("Please ensure TensorFlow is installed with ImageNet weights.")
        
        
    
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


