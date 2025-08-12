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

FOLDER = './uploads'
PRIVATE = os.getenv('PRIVATE', 'False').lower() in ['true', '1', 'yes']
PORT = os.getenv('PORT', '7774')
NSFW_THRESHOLD = float(os.getenv('NSFW_THRESHOLD', '50'))  # Using 50% as starting threshold
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
        
        response = {
            "service": "nsfw2",
            "status": "success",
            "predictions": [{
                "confidence": round(nsfw_probability, 3),  # 0-1 scale
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
            "GET /?url=<image_url> - Detect NSFW from URL",
            "GET /?path=<local_path> - Detect NSFW from local file (if not private)",
            "POST / - Upload and detect NSFW in image",
            "GET /v2/analyze?image_url=<url> - V2 API endpoint"
        ]
    })

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API endpoint with unified response format"""
    start_time = time.time()
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "nsfw2",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Download and process image
        try:
            filename = uuid.uuid4().hex + ".jpg"
            filepath = os.path.join(FOLDER, filename)
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            if len(response.content) > MAX_FILE_SIZE:
                raise ValueError("Downloaded file too large")
            
            with open(filepath, "wb") as file:
                file.write(response.content)
            
            # Analyze using OpenNSFW2
            result = detect_nsfw_opennsfw2(filename)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                "service": "nsfw2",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        return jsonify({
            "service": "nsfw2",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        url = request.args.get('url') or request.args.get('img')
        path = request.args.get('path')

        if url:
            try:
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(FOLDER, filename)
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                if len(response.content) > MAX_FILE_SIZE:
                    return jsonify({
                        "error": f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                        "status": "error"
                    }), 400
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                result = detect_nsfw_opennsfw2(filename)
                return jsonify(result)
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    "error": f"Failed to download image: {str(e)}",
                    "status": "error"
                }), 400
            except Exception as e:
                return jsonify({
                    "error": f"Image processing failed: {str(e)}",
                    "status": "error"
                }), 500
                
        elif path:
            if PRIVATE:
                return jsonify({
                    "error": "Local file access disabled in private mode",
                    "status": "error"
                }), 403
            
            if not os.path.exists(os.path.join(FOLDER, path)):
                return jsonify({
                    "error": "File not found",
                    "status": "error"
                }), 404
                
            return jsonify(detect_nsfw_opennsfw2(path, cleanup=False))
            
        else:
            # Return simple form for testing
            return '''<!DOCTYPE html>
<html>
<head><title>NSFW2 Detection API</title></head>
<body>
<h2>NSFW2 Detection Service (OpenNSFW2)</h2>
<form enctype="multipart/form-data" method="POST">
    <input type="file" name="uploadedfile" accept="image/*" required><br><br>
    <input type="submit" value="Detect NSFW">
</form>
<p><strong>API Usage:</strong></p>
<ul>
    <li>GET /?url=&lt;image_url&gt; - Detect NSFW from URL</li>
    <li>POST with file upload - Detect NSFW in uploaded image</li>
    <li>GET /health - Service health check</li>
    <li>GET /v2/analyze?image_url=&lt;url&gt; - V2 unified API</li>
</ul>
</body>
</html>'''
    
    elif request.method == 'POST':
        if not request.files:
            return jsonify({
                "error": "No file uploaded",
                "status": "error"
            }), 400
        
        for field_name, file_data in request.files.items():
            if not file_data.filename:
                continue
                
            try:
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(FOLDER, filename)
                file_data.save(filepath)
                
                file_size = os.path.getsize(filepath)
                if file_size > MAX_FILE_SIZE:
                    cleanup_file(filepath)
                    return jsonify({
                        "error": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                        "status": "error"
                    }), 400
                
                result = detect_nsfw_opennsfw2(filename)
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "error": f"File processing failed: {str(e)}",
                    "status": "error"
                }), 500
        
        return jsonify({
            "error": "No valid file found in upload",
            "status": "error"
        }), 400

if __name__ == '__main__':
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting NSFW2 Detection API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"NSFW threshold: {NSFW_THRESHOLD}%")
    print(f"Model: OpenNSFW2 (Yahoo/Flickr)")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)