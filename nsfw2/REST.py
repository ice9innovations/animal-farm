import os

# Set CUDA environment variables BEFORE importing any ML libraries
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
os.environ['PATH'] = '/usr/local/cuda-12.2/bin:' + os.environ.get('PATH', '')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CUDA paths are set by nsfw2.sh script

import json
import requests
import uuid
import time
import shutil
from typing import Dict, Any

import tensorflow as tf
from absl import logging as absl_logging
import opennsfw2 as n2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

tf.get_logger().setLevel('ERROR')
absl_logging.set_verbosity(absl_logging.ERROR)

# Enable optimizations only if GPU is available
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print(f"GPU optimization enabled for {len(gpus)} GPU(s)")
    else:
        print("Running on CPU - GPU optimizations disabled")
except Exception as e:
    print(f"GPU setup warning: {e}")

load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
NSFW_THRESHOLD_STR = os.getenv('NSFW_THRESHOLD')

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

FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

def get_nsfw_emoji(probability: float) -> str:
    """Get appropriate emoji based on NSFW probability"""
    if probability > NSFW_THRESHOLD:
        return "🔞"
    return ""

def convert_to_jpg(filepath: str) -> str:
    """Convert image to JPG format if needed"""
    try:
        # Check if already JPG/JPEG
        if filepath.lower().endswith(('.jpg', '.jpeg')):
            return filepath
            
        # Open image with PIL and convert to JPG
        with Image.open(filepath) as img:
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create new filename with .jpg extension
            jpg_filepath = filepath.rsplit('.', 1)[0] + '.jpg'
            
            # Save as JPG
            img.save(jpg_filepath, 'JPEG', quality=95)
            
            return jpg_filepath
    except Exception as e:
        print(f"Error converting image to JPG: {e}")
        return filepath  # Return original if conversion fails

def cleanup_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Could not remove file {filepath}: {e}")

def detect_nsfw_opennsfw2(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Detect NSFW content using OpenNSFW2 model"""
    start_time = time.time()
    full_path = os.path.join(FOLDER, image_path)
    
    try:
        # Convert to JPG if needed (OpenNSFW2 handles multiple formats but JPG is most reliable)
        jpg_path = convert_to_jpg(full_path)
        
        # Get NSFW probability using OpenNSFW2
        result = n2.predict_image(jpg_path)
        
        # Handle different return types from OpenNSFW2
        if isinstance(result, dict):
            nsfw_probability = result["nsfw"]
        elif isinstance(result, (float, int)):
            nsfw_probability = result
        else:
            raise ValueError(f"Unexpected result type from OpenNSFW2: {type(result)}")
        
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
        
        response = {
            "service": "nsfw2",
            "status": "success",
            "predictions": [{
                "confidence": prediction_confidence,
                "emoji": emoji,
                "nsfw": is_nsfw
            }],
            "metadata": {
                "processing_time": detection_time,
                "model_info": {
                    "framework": "TensorFlow"
                }
            }
        }
        
        # Cleanup files
        if cleanup:
            cleanup_file(full_path)
            if jpg_path != full_path:
                cleanup_file(jpg_path)  # Clean up converted JPG if different
        
        return response
        
    except Exception as e:
        if cleanup:
            cleanup_file(full_path)
            if 'jpg_path' in locals() and jpg_path != full_path:
                cleanup_file(jpg_path)
        return {
            "service": "nsfw2",
            "status": "error",
            "predictions": [],
            "error": {"message": f"NSFW detection failed: {str(e)}"},
            "metadata": {
                "processing_time": round(time.time() - start_time, 3)
            }
        }

app = Flask(__name__)

# Enable CORS for direct browser access
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("NSFW2 service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "NSFW2 Detection",
        "model": {
            "status": "loaded",
            "framework": "Keras/TensorFlow",
            "model_type": "OpenNSFW2",
            "threshold": NSFW_THRESHOLD
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /v3/analyze?url=<image_url> - Analyze image from URL",
            "GET /v3/analyze?file=<file_path> - Analyze image from file",
            "GET /v2/analyze?image_url=<url> - V2 compatibility (deprecated)",
            "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
        ]
    })

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    start_time = time.time()
    
    try:
        # Get parameters from query string
        url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input - exactly one parameter required
        if not url and not file_path:
            return jsonify({
                "service": "nsfw2",
                "status": "error", 
                "predictions": [],
                "error": {"message": "Must provide either 'url' or 'file' parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "TensorFlow"
                    }
                }
            }), 400
        
        if url and file_path:
            return jsonify({
                "service": "nsfw2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "TensorFlow"
                    }
                }
            }), 400
        
        # Handle URL input
        if url:
            try:
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(FOLDER, filename)
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                if len(response.content) > MAX_FILE_SIZE:
                    return jsonify({
                        "service": "nsfw2",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"},
                        "metadata": {
                            "processing_time": round(time.time() - start_time, 3),
                            "model_info": {
                                "framework": "TensorFlow"
                            }
                        }
                    }), 400
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                # Analyze using OpenNSFW2
                result = detect_nsfw_opennsfw2(filename)
                return jsonify(result)
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    "service": "nsfw2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to download image: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "TensorFlow"
                        }
                    }
                }), 400
        
        # Handle file path input
        elif file_path:
            # Validate file path exists
            if not os.path.exists(file_path):
                return jsonify({
                    "service": "nsfw2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file_path}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "TensorFlow"
                        }
                    }
                }), 404
            
            # Extract filename from path for processing
            filename = os.path.basename(file_path)
            
            # Copy to uploads folder temporarily
            temp_filename = uuid.uuid4().hex + "_" + filename
            temp_filepath = os.path.join(FOLDER, temp_filename)
            
            try:
                shutil.copy2(file_path, temp_filepath)
                
                result = detect_nsfw_opennsfw2(temp_filename)
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "service": "nsfw2",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process file: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "TensorFlow"
                        }
                    }
                }), 500
        
    except Exception as e:
        return jsonify({
            "service": "nsfw2",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "TensorFlow"
                }
            }
        }), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation: image_url -> url
        new_args = {'url': image_url}
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # Let V3 handle validation errors
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()


if __name__ == '__main__':
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting NSFW2 Detection API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"NSFW threshold: {NSFW_THRESHOLD}%")
    print(f"Model: OpenNSFW2 (Yahoo/Flickr)")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)