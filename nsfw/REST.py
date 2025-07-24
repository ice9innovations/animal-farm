import os
import json
import requests
import os
from dotenv import load_dotenv
import uuid
import time
from typing import List, Any, Tuple, Dict, Optional

import tensorflow as tf
from absl import logging as absl_logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

tf.get_logger().setLevel('ERROR')
absl_logging.set_verbosity(absl_logging.ERROR)

# Force CPU-only execution for NSFW service - GPU was causing timeout issues
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
NSFW_THRESHOLD = float(os.getenv('NSFW_THRESHOLD', '35'))
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

def preprocess_for_evaluation(image: tf.Tensor, image_size: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    """Preprocess image for evaluation with the Private Detector model"""
    image = pad_resize_image(image, [image_size, image_size])
    image = tf.cast(image, dtype)
    image -= 128
    image /= 128
    return image

def pad_resize_image(image: tf.Tensor, dims: Tuple[int, int]) -> tf.Tensor:
    """Resize image with padding to maintain aspect ratio"""
    image = tf.image.resize(image, dims, preserve_aspect_ratio=True)
    shape = tf.shape(image)
    
    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]
    
    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)
    
    paddings = tf.convert_to_tensor([
        [sy, syd - sy],
        [sx, sxd - sx],
        [0, 0]
    ])
    
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=128)
    return image

def read_image(filename: str) -> tf.Tensor:
    """Load and preprocess image for NSFW detection"""
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    # Model expects 480x480x3=691200 elements with float16 precision
    image = preprocess_for_evaluation(image, 480, tf.float16)
    image = tf.reshape(image, [1, -1])
    return image

def load_model() -> Optional[tf.Module]:
    """Load the NSFW detection model"""
    try:
        model_path = "./saved_model/"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
        return tf.saved_model.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_nsfw_emoji(probability: float) -> str:
    """Get appropriate emoji based on NSFW probability"""
    if probability > NSFW_THRESHOLD:
        return "🔞"
    return ""

def convert_to_jpg(filepath: str) -> str:
    """Convert image to JPG format - always convert since NSFW only handles JPG"""
    try:
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


model = load_model()

def detect_nsfw(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Detect NSFW content in image using Bumble's private-detector model"""
    global model
    
    if model is None:
        return {
            "error": "NSFW detection model not loaded",
            "status": "error"
        }
    
    start_time = time.time()
    full_path = os.path.join(FOLDER, image_path)
    
    try:
        image = read_image(full_path)
        predictions = model(image)
        probability = float(tf.get_static_value(predictions[0])[0])
        
        # Convert to percentage and round to 3 decimal places
        probability_percent = round(probability * 100, 3)
        
        emoji = get_nsfw_emoji(probability_percent)
        detection_time = round(time.time() - start_time, 3)
        
        result = {
            "NSFW": {
                "probability": probability_percent,
                "threshold": NSFW_THRESHOLD,
                "is_nsfw": probability_percent > NSFW_THRESHOLD,
                "emoji": emoji,
                "model_info": {
                    "framework": "TensorFlow",
                    "model": "EfficientNet-v2 (Bumble private-detector)",
                    "detection_time": detection_time,
                    "input_size": "320x320",
                    "threshold": NSFW_THRESHOLD
                },
                "status": "success"
            }
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            cleanup_file(full_path)
        return result
        
    except Exception as e:
        if cleanup:
            cleanup_file(full_path)
        return {
            "error": f"NSFW detection failed: {str(e)}",
            "status": "error"
        }


app = Flask(__name__)

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("NSFW service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        "status": "healthy",
        "service": "NSFW Detection",
        "model": {
            "status": model_status,
            "framework": "TensorFlow",
            "model_type": "EfficientNet-v2 (Bumble private-detector)",
            "threshold": NSFW_THRESHOLD
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /?url=<image_url> - Detect NSFW from URL",
            "GET /?path=<local_path> - Detect NSFW from local file (if not private)",
            "POST / - Upload and detect NSFW in image"
        ]
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
                "service": "nsfw",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "nsfw",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        # Validate file is JPG (NSFW service requires JPG format)
        if not file_path.lower().endswith(('.jpg', '.jpeg')):
            return jsonify({
                "service": "nsfw",
                "status": "error",
                "predictions": [],
                "error": {"message": "NSFW service requires JPEG format. Please provide a .jpg or .jpeg file."},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Create a temporary copy in FOLDER since detect_nsfw expects files there
        temp_filename = uuid.uuid4().hex + ".jpg"
        temp_filepath = os.path.join(FOLDER, temp_filename)
        
        # Copy the file to temp location
        import shutil
        shutil.copy2(file_path, temp_filepath)
        
        # Analyze using existing function (no cleanup - we handle the temp file)
        result = detect_nsfw(temp_filename, cleanup=False)
        
        # Clean up temporary file
        cleanup_file(temp_filepath)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "nsfw",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'NSFW detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        nsfw_data = result.get('NSFW', {})
        probability = nsfw_data.get('probability', 0)
        is_nsfw = nsfw_data.get('is_nsfw', False)
        emoji = nsfw_data.get('emoji', '')
        
        # Create unified prediction format
        predictions = []
        
        # NSFW content moderation prediction
        prediction = {
            "type": "content_moderation",
            "label": "nsfw" if is_nsfw else "safe",
            "emoji": emoji,
            "confidence": round(probability / 100.0, 3),  # Convert percentage to 0-1 scale
            "value": "nsfw" if is_nsfw else "safe",
            "properties": {
                "probability": probability,
                "threshold": NSFW_THRESHOLD,
                "is_nsfw": is_nsfw
            }
        }
        predictions.append(prediction)
        
        return jsonify({
            "service": "nsfw",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "EfficientNet-v2",
                    "framework": "TensorFlow"
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "service": "nsfw",
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
                "service": "nsfw",
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
            
            # Convert to JPG if needed
            jpg_filepath = convert_to_jpg(filepath)
            jpg_filename = os.path.basename(jpg_filepath)
            
            # Analyze using existing function
            result = detect_nsfw(jpg_filename)
            
            # Cleanup both original and converted files
            cleanup_file(filepath)  # Original download
            cleanup_file(jpg_filepath)  # Converted JPG
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "nsfw",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'NSFW detection failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            nsfw_data = result.get('NSFW', {})
            probability = nsfw_data.get('probability', 0)
            is_nsfw = nsfw_data.get('is_nsfw', False)
            emoji = nsfw_data.get('emoji', '')
            
            # Create unified prediction format
            predictions = []
            
            # NSFW content moderation prediction
            prediction = {
                "type": "content_moderation",
                "label": "nsfw" if is_nsfw else "safe",
                "emoji": emoji,
                "confidence": round(probability / 100.0, 3),  # Convert percentage to 0-1 scale
                "value": "nsfw" if is_nsfw else "safe",
                "properties": {
                    "probability": probability,
                    "threshold": NSFW_THRESHOLD,
                    "is_nsfw": is_nsfw
                }
            }
            predictions.append(prediction)
            
            return jsonify({
                "service": "nsfw",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "EfficientNet-v2",
                        "framework": "TensorFlow"
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "service": "nsfw",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        return jsonify({
            "service": "nsfw",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        url = request.args.get('url') or request.args.get('img')  # Accept both 'url' and 'img' parameters
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
                
                result = jsonify(detect_nsfw(filename))
                # Cleanup file after getting result
                cleanup_file(filepath)
                return result
                
            except requests.exceptions.RequestException as e:
                # Cleanup on error
                if 'filepath' in locals() and os.path.exists(filepath):
                    cleanup_file(filepath)
                return jsonify({
                    "error": f"Failed to download image: {str(e)}",
                    "status": "error"
                }), 400
            except Exception as e:
                # Cleanup on error
                if 'filepath' in locals() and os.path.exists(filepath):
                    cleanup_file(filepath)
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
                
            return jsonify(detect_nsfw(path))
            
        else:
            try:
                with open('form.html', 'r') as file:
                    return file.read()
            except FileNotFoundError:
                return '''<!DOCTYPE html>
<html>
<head><title>NSFW Detection API</title></head>
<body>
<h2>NSFW Detection Service</h2>
<form enctype="multipart/form-data" method="POST">
    <input type="file" name="uploadedfile" accept="image/*" required><br><br>
    <input type="submit" value="Detect NSFW">
</form>
<p><strong>API Usage:</strong></p>
<ul>
    <li>GET /?url=&lt;image_url&gt; - Detect NSFW from URL</li>
    <li>POST with file upload - Detect NSFW in uploaded image</li>
    <li>GET /health - Service health check</li>
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
                
                result = jsonify(detect_nsfw(filename))
                # Cleanup file after getting result
                cleanup_file(filepath)
                return result
                
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
    if model is None:
        print("WARNING: NSFW detection model not loaded. Please ensure ./saved_model/ directory exists.")
        print("Download instructions: https://github.com/bumble-tech/private-detector")
    
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting NSFW Detection API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"NSFW threshold: {NSFW_THRESHOLD}%")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)


