import os
from dotenv import load_dotenv

load_dotenv()

# Must be set before importing TensorFlow
MODE_STR = os.getenv('MODE')
if not MODE_STR:
    raise ValueError("MODE environment variable is required (cpu or gpu)")
MODE = MODE_STR.lower()
if MODE not in ('cpu', 'gpu'):
    raise ValueError(f"MODE must be 'cpu' or 'gpu', got: {MODE_STR!r}")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if MODE == 'cpu' else '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import requests
import uuid
import time
import shutil
import random
from typing import Dict, Any

import tensorflow as tf
from absl import logging as absl_logging
import opennsfw2 as n2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

tf.get_logger().setLevel('ERROR')
absl_logging.set_verbosity(absl_logging.ERROR)

if MODE == 'gpu':
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.optimizer.set_jit(True)
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print(f"GPU optimization enabled for {len(gpus)} GPU(s)")
        else:
            print("MODE=gpu but no GPU devices found")
    except Exception as e:
        print(f"GPU setup warning: {e}")
else:
    print("Running on CPU (MODE=cpu)")

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
NSFW_THRESHOLD_STR = os.getenv('NSFW_THRESHOLD')
# MODE already validated at startup before TF import

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not NSFW_THRESHOLD_STR:
    raise ValueError("NSFW_THRESHOLD environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
NSFW_THRESHOLD = float(NSFW_THRESHOLD_STR)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

def get_nsfw_emoji(probability: float) -> str:
    """Get appropriate emoji based on NSFW probability"""
    if probability > NSFW_THRESHOLD:
        return "🔞"
    return ""

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

# Note: Old file-based functions removed - now using pure in-memory processing

def process_image_for_nsfw(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns NSFW detection data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing with OpenNSFW2 PIL Image support
    """
    start_time = time.time()
    
    try:
        # Ensure image is in RGB mode (OpenNSFW2 expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get NSFW probability using OpenNSFW2 directly with PIL Image
        # No file I/O needed - OpenNSFW2 accepts PIL Images directly!
        nsfw_probability = n2.predict_image(image)
        
        # Handle different return types from OpenNSFW2
        if isinstance(nsfw_probability, dict):
            nsfw_probability = nsfw_probability["nsfw"]
        elif not isinstance(nsfw_probability, (float, int)):
            raise ValueError(f"Unexpected result type from OpenNSFW2: {type(nsfw_probability)}")
        
        # Convert to percentage
        probability_percent = round(nsfw_probability * 100, 3)
        
        emoji = get_nsfw_emoji(probability_percent)
        detection_time = round(time.time() - start_time, 3)
        is_nsfw = probability_percent > NSFW_THRESHOLD
        
        # Calculate confidence - how confident we are in the prediction
        if is_nsfw:
            # If NSFW, confidence is the raw NSFW probability
            prediction_confidence = round(nsfw_probability, 3)
        else:
            # If safe, confidence is 1 - NSFW probability (confidence it's safe)
            prediction_confidence = round(1.0 - nsfw_probability, 3)
        
        return {
            "success": True,
            "data": {
                "confidence": prediction_confidence,
                "emoji": emoji,
                "nsfw": is_nsfw,
                "probability_percent": probability_percent,
                "raw_probability": nsfw_probability
            },
            "processing_time": detection_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"NSFW detection failed: {str(e)}",
            "processing_time": round(time.time() - start_time, 3)
        }

def create_nsfw_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized NSFW detection response"""
    is_shiny, shiny_roll = check_shiny()
    
    prediction = {
        "confidence": data["confidence"],
        "emoji": data["emoji"],
        "nsfw": data["nsfw"]
    }
    
    # Add shiny flag only for shiny detections
    if is_shiny:
        prediction["shiny"] = True
        print(f"✨ SHINY NSFW2 MODERATION DETECTED! Roll: {shiny_roll} ✨")
    
    return {
        "service": "nsfw2",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "TensorFlow"
            }
        }
    }

app = Flask(__name__)


def _normalize_health_payload(payload):
    """Ensure every /health response exposes the common Animal Farm health shape."""
    if not isinstance(payload, dict):
        return payload

    payload.setdefault("status", "healthy")
    payload.setdefault("schema_version", "health.v1")
    if not payload.get("service"):
        payload["service"] = str(globals().get("SERVICE_NAME") or __file__.replace("\\", "/").rstrip("/").split("/")[-2])

    warnings = payload.get("warnings", [])
    if warnings is None:
        warnings = []
    elif isinstance(warnings, str):
        warnings = [warnings]
    payload["warnings"] = warnings

    dependencies = payload.get("dependencies") if isinstance(payload.get("dependencies"), dict) else {}
    for key in ("llama_server", "ollama", "extraction_engines"):
        if key in payload and key not in dependencies:
            dependencies[key] = payload[key]
    payload["dependencies"] = dependencies

    model_value = payload.get("model")
    if isinstance(model_value, dict):
        model = dict(model_value)
    elif model_value is not None:
        payload.setdefault("model_name", model_value)
        model = {"name": model_value}
    else:
        model = {}

    if "models" in payload and "components" not in model:
        model["components"] = payload["models"]
    for source_key in ("detector", "analyzer", "ocr_engine"):
        source_value = payload.get(source_key)
        if isinstance(source_value, dict):
            model.setdefault("status", source_value.get("status"))
            model.setdefault("details", source_value)
    if "model_status" in payload and "status" not in model:
        model["status"] = payload["model_status"]
    if "model_loaded" in payload and "status" not in model:
        model["status"] = "loaded" if payload["model_loaded"] else "not_loaded"
    if "backend_status" in payload and "status" not in model:
        model["status"] = payload["backend_status"]
    if "device" in payload and "device" not in model:
        model["device"] = payload["device"]
    if "framework" in payload and "framework" not in model:
        model["framework"] = payload["framework"]
    if "backend" in payload and "backend" not in model:
        model["backend"] = payload["backend"]
    for threshold_key in ("threshold", "confidence_threshold", "detection_threshold", "classification_threshold"):
        if threshold_key in payload and threshold_key not in model:
            model[threshold_key] = payload[threshold_key]
    payload["model"] = model

    payload.setdefault("endpoints", [])
    return payload


@app.after_request
def _normalize_health_response(response):
    if request.path != "/health" or not response.is_json:
        return response
    payload = response.get_json(silent=True)
    normalized = _normalize_health_payload(payload)
    if normalized is not payload:
        return response
    response.set_data(app.json.dumps(normalized))
    response.content_type = "application/json"
    return response

# Enable CORS for direct browser access
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("NSFW2 service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test if OpenNSFW2 is working by creating a small test image
    try:
        import opennsfw2 as n2
        test_image = Image.new('RGB', (10, 10), color='red')
        n2.predict_image(test_image)  # This will fail if model can't load
        model_status = "loaded"
        status = "healthy"
    except Exception as e:
        model_status = f"error: {str(e)}"
        status = "unhealthy"
        return jsonify({
            "status": status,
            "reason": f"OpenNSFW2 model error: {str(e)}",
            "service": "NSFW2 Detection"
        }), 503
    
    return jsonify({
        "status": status,
        "service": "NSFW2 Detection",
        "model": {
            "status": model_status,
            "framework": "Keras/TensorFlow",
            "model_type": "OpenNSFW2",
            "threshold": NSFW_THRESHOLD
        },
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /v3/analyze - V3 compatibility",
            "GET /v2/analyze - V2 compatibility (deprecated)",
            "GET /v2/analyze_file - V2 compatibility (deprecated)"
        ]
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "nsfw2",
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
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
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
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_nsfw(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_nsfw_response(
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

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

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


if __name__ == '__main__':
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting NSFW2 Detection API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"NSFW threshold: {NSFW_THRESHOLD}%")
    print(f"Model: OpenNSFW2 (Yahoo/Flickr)")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)
