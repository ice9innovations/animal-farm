import json
import requests
import os
import uuid
import time
import re
from typing import Dict, Any, Optional, List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
import nltk

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT_STR:
    raise ValueError("API_TIMEOUT environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR)

FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Global emoji mappings and MWE tokenizer - loaded from API on startup
emoji_mappings = {}
emoji_tokenizer = None

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

# Initialize PaddleOCR (run once)
print("Initializing PaddleOCR...")
ocr_engine = PaddleOCR(lang='en')
print("PaddleOCR initialized successfully")

def cleanup_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Could not remove file {filepath}: {e}")

def load_mwe_mappings():
    """Load fresh MWE mappings from central API and convert to tuples"""
    mwe_url = f"http://{API_HOST}:{API_PORT}/mwe.txt"
    try:
        print(f"🔄 OCR: Loading fresh multi-word expressions (MWE) mappings from {mwe_url}")
        response = requests.get(mwe_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        mwe_text = response.text.splitlines()
        
        # Convert to tuples for MWETokenizer
        mwe_tuples = []
        for line in mwe_text:
            if line.strip():
                mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
        
        return mwe_tuples
    except Exception as e:
        print(f"❌ OCR: Failed to load multi-word expressions (MWE) mappings from {mwe_url}: {e}")
        return []

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    global emoji_mappings
    
    api_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
    print(f"🔄 OCR: Loading fresh emoji mappings from {api_url}")
    
    response = requests.get(api_url, timeout=API_TIMEOUT)
    response.raise_for_status()
    emoji_mappings = response.json()
    
    print(f"✅ OCR: Loaded fresh emoji mappings from API ({len(emoji_mappings)} entries)")

def get_emoji(label):
    """Get emoji for a given label from emoji mappings"""
    return emoji_mappings.get(label, None)

def get_emojis_for_words(words: List[str]) -> List[Dict[str, str]]:
    """Get emoji mappings for a list of words"""
    if not words:
        return []
    
    mappings = []
    for word in words:
        emoji = get_emoji(word.lower())
        if emoji:
            mappings.append({"word": word, "emoji": emoji})
    
    print(f"OCR: Found {len(mappings)} emoji mappings from {len(words)} words")
    return mappings

def extract_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words from text using NLTK with MWE detection"""
    if not text or not text.strip():
        return []
    
    try:
        # Tokenize text with MWE detection
        word_tokens = []
        for token in text.split():
            # Remove common punctuation
            token = token.strip('.,!?;:"()[]{}').lower()
            if token:
                word_tokens.append(token)
        
        # Use MWE tokenizer if available
        if emoji_tokenizer:
            tokens = emoji_tokenizer.tokenize(word_tokens)
        else:
            tokens = word_tokens
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Filter meaningful tokens
        meaningful_words = []
        for token in tokens:
            # Handle MWE tokens (joined with _)
            if '_' in token:
                # Multi-word expression - keep as is
                meaningful_words.append(token.replace('_', ' '))
            elif token.isalpha() and len(token) >= 3 and token not in stop_words:
                # Single word - apply normal filtering
                meaningful_words.append(token)
        
        # Remove duplicates while preserving order
        unique_words = []
        seen = set()
        for word in meaningful_words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        return unique_words[:20]  # Limit to first 20 meaningful words
        
    except Exception as e:
        print(f"Error extracting meaningful words: {e}")
        # Fallback to simple regex extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(dict.fromkeys(words))[:20]  # Remove duplicates

def extract_text_paddleocr(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Extract text from image using PaddleOCR"""
    start_time = time.time()
    
    try:
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Run PaddleOCR
        ocr_result = ocr_engine.ocr(image_array, cls=True)
        
        # Extract text, confidence scores, and bounding boxes
        texts = []
        confidences = []
        text_regions = []
        
        if ocr_result and ocr_result[0]:  # PaddleOCR returns nested list
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    bbox_coords = line[0]  # 4-point polygon coordinates
                    text_info = line[1]   # [text, confidence]
                    
                    if len(text_info) >= 2:
                        text_content = text_info[0]
                        confidence = text_info[1]
                        
                        texts.append(text_content)
                        confidences.append(confidence)
                        
                        # Convert 4-point polygon to bounding box
                        if bbox_coords and len(bbox_coords) >= 4:
                            # bbox_coords = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            x_coords = [point[0] for point in bbox_coords]
                            y_coords = [point[1] for point in bbox_coords]
                            
                            x_min = min(x_coords)
                            y_min = min(y_coords)
                            x_max = max(x_coords)
                            y_max = max(y_coords)
                            
                            text_regions.append({
                                "text": text_content,
                                "confidence": round(confidence, 3),
                                "bbox": {
                                    "x": int(x_min),
                                    "y": int(y_min),
                                    "width": int(x_max - x_min),
                                    "height": int(y_max - y_min)
                                }
                            })
        
        # Combine all text
        raw_text = ' '.join(texts)
        
        # Clean text (better spacing, normalization)
        cleaned_text = raw_text
        # Remove spaces before punctuation (common OCR artifact)
        cleaned_text = re.sub(r'\s+([.!?,:;])', r'\1', cleaned_text)
        # Add single space after punctuation if not already there
        cleaned_text = re.sub(r'([.!?,:;])(?!\s)', r'\1 ', cleaned_text)
        # Fix multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        # Remove leading/trailing spaces
        cleaned_text = cleaned_text.strip()
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine if meaningful text was found
        has_text = bool(cleaned_text and len(cleaned_text.strip()) >= 3)
        
        processing_time = round(time.time() - start_time, 3)
        
        result = {
            "text": cleaned_text,
            "raw_text": raw_text,
            "has_text": has_text,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "engine": "PaddleOCR",
            "status": "success",
            "text_regions": text_regions
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            cleanup_file(image_path)
        
        return result
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        
        # Cleanup on error (only for temporary files)  
        if cleanup:
            cleanup_file(image_path)
        
        return {
            "error": f"OCR processing failed: {str(e)}",
            "processing_time": processing_time,
            "status": "error"
        }

app = Flask(__name__)

# Enable CORS for direct browser access
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("PaddleOCR service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "PaddleOCR",
        "ocr_engine": {
            "available": True,
            "version": "PaddleOCR 2.x",
            "languages": ["English", "Chinese", "80+ others"],
            "gpu_enabled": True
        },
        "features": {
            "text_angle_classification": True,
            "multilingual_support": True,
            "gpu_acceleration": True,
            "high_accuracy": True
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
                "service": "ocr",
                "status": "error", 
                "predictions": [],
                "error": {"message": "Must provide either 'url' or 'file' parameter"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "PaddleOCR"
                    }
                }
            }), 400
        
        if url and file_path:
            return jsonify({
                "service": "ocr",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "PaddleOCR"
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
                        "service": "ocr",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"},
                        "metadata": {
                            "processing_time": round(time.time() - start_time, 3),
                            "model_info": {
                                "framework": "PaddleOCR"
                            }
                        }
                    }), 400
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                # Extract text using PaddleOCR (cleanup=True will handle file removal)
                result = extract_text_paddleocr(filepath)
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    "service": "ocr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to download image: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "PaddleOCR"
                        }
                    }
                }), 400
        
        # Handle file path input
        elif file_path:
            # Validate file path exists
            if not os.path.exists(file_path):
                return jsonify({
                    "service": "ocr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file_path}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {
                            "framework": "PaddleOCR"
                        }
                    }
                }), 404
            
            # Extract text directly from file (no cleanup needed - we don't own the file)
            result = extract_text_paddleocr(file_path, cleanup=False)
        
        # Process result for both URL and file path cases
        if result.get('status') == 'error':
            return jsonify({
                "service": "ocr",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'OCR processing failed')},
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "framework": "PaddleOCR"
                    }
                }
            }), 500
        
        # Create unified prediction format
        predictions = []
        
        text = result.get('text', '')
        raw_text = result.get('raw_text', '')
        has_text = result.get('has_text', False)
        confidence = result.get('confidence', 0.0)
        
        # Text extraction prediction (main OCR result with text regions and emoji mappings)
        text_regions = result.get('text_regions', [])
        
        # Get emoji mappings for meaningful words in the text
        emoji_mappings_list = []
        if has_text and text:
            print(f"OCR: Extracted text: '{text[:100]}...'")
            meaningful_words = extract_meaningful_words(text)
            print(f"OCR: Found {len(meaningful_words)} meaningful words: {meaningful_words[:10]}")
            
            if meaningful_words:
                emoji_mappings_list = get_emojis_for_words(meaningful_words)
                print(f"OCR: Generated {len(emoji_mappings_list)} emoji mappings from text content")
        
        text_prediction = {
            "text": text,
            "emoji": get_emoji("ocr") or "📝",
            "confidence": round(confidence, 3),
            "has_text": has_text,
            "text_regions": text_regions,
            "emoji_mappings": emoji_mappings_list
        }
        predictions.append(text_prediction)
        
        return jsonify({
            "service": "ocr",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "PaddleOCR"
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "service": "ocr",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "PaddleOCR"
                }
            }
        }), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        # Parameter translation: file_path -> file
        new_args = {'file': file_path}
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # Let V3 handle validation errors
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
    # Load emoji mappings and MWE patterns on startup
    load_emoji_mappings()
    
    # Initialize MWE tokenizer
    mwe_mappings = load_mwe_mappings()
    if mwe_mappings:
        emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')
        print(f"✅ OCR: Initialized MWE tokenizer with {len(mwe_mappings)} multi-word expressions")
    else:
        print("⚠️  OCR: No MWE mappings available - using basic tokenization")
    
    app.run(host='0.0.0.0', port=int(PORT), debug=False)