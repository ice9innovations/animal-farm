#!/usr/bin/env python3
"""
Inception v3 Image Classification REST API Service
Coordination service for Google's Inception v3 model for ImageNet classification.
"""

import os
import json
import uuid
import logging
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

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')  # Must be set in .env
API_PORT = int(os.getenv('API_PORT'))  # Must be set in .env  
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '2.0'))


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
PRIVATE = os.getenv('PRIVATE', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', '7779'))
CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 160

# Environment variables validation
for var in ['DISCORD_TOKEN', 'DISCORD_GUILD', 'DISCORD_CHANNEL']:
    if not os.getenv(var):
        logger.warning(f"Environment variable {var} not set")

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL', '').split(',') if os.getenv('DISCORD_CHANNEL') else []

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
    """V2 API endpoint for direct file path analysis"""
    import time
    start_time = time.time()
    
    try:
        # Get file path from query parameters
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                "service": "inception",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "inception",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        if not is_allowed_file(file_path):
            return jsonify({
                "service": "inception",
                "status": "error",
                "predictions": [],
                "error": {"message": "File type not allowed"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Classify directly from file (no cleanup needed - we don't own the file)
        result = classify_image(file_path, cleanup=False)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "inception",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Classification failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        inception_data = result.get('INCEPTION', {})
        classifications = inception_data.get('classifications', [])
        image_dims = inception_data.get('image_dimensions', {})
        
        # Create unified prediction format
        predictions = []
        for classification in classifications:
            prediction = {
                "type": "classification",
                "label": classification.get('class_name', ''),
                "confidence": round(float(classification.get('confidence', 0)), 3)  # Normalize to 0-1
            }
            
            # Add emoji if present
            if classification.get('emoji'):
                prediction["emoji"] = classification['emoji']
            
            predictions.append(prediction)
        
        return jsonify({
            "service": "inception",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "Inception v3",
                    "framework": "TensorFlow"
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
            "service": "inception",
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
                "service": "inception",
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
            
            # Validate file size
            if os.path.getsize(filepath) > MAX_FILE_SIZE:
                os.remove(filepath)
                raise ValueError("Downloaded file too large")
            
            # Classify using existing function
            result = classify_image(filepath)
            filepath = None  # classify_image handles cleanup
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "inception",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Classification failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            inception_data = result.get('INCEPTION', {})
            classifications = inception_data.get('classifications', [])
            image_dims = inception_data.get('image_dimensions', {})
            
            # Create unified prediction format
            predictions = []
            for classification in classifications:
                prediction = {
                    "type": "classification",
                    "label": classification.get('class_name', ''),
                    "confidence": round(float(classification.get('confidence', 0)), 3)  # Normalize to 0-1
                }
                
                # Add emoji if present
                if classification.get('emoji'):
                    prediction["emoji"] = classification['emoji']
                
                predictions.append(prediction)
            
            # Get model info
            model_info = inception_data.get('model_info', {})
            
            return jsonify({
                "service": "inception",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "Inception v3",
                        "framework": "TensorFlow"
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
                "service": "inception",
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
            "service": "inception",
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
                
            result = classify_image(path)
            return jsonify(result)
            
        else:
            # Return HTML form
            try:
                with open('form.html', 'r') as file:
                    html = file.read()
            except FileNotFoundError:
                html = f'''<!DOCTYPE html>
<html>
<head><title>Inception v3 Image Classification</title></head>
<body>
<h1>Inception v3 Image Classification</h1>
<form enctype="multipart/form-data" action="" method="POST">
    <input type="hidden" name="MAX_FILE_SIZE" value="{MAX_FILE_SIZE}" />
    <p>Upload an image file:</p>
    <input name="uploadedfile" type="file" accept="image/*" required /><br /><br />
    <input type="submit" value="Classify Image" />
</form>
<p>Supported formats: {', '.join(ALLOWED_EXTENSIONS)}</p>
<p>Max file size: {MAX_FILE_SIZE // (1024*1024)}MB</p>
<p>Classifies images using Google's Inception v3 trained on ImageNet</p>
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


