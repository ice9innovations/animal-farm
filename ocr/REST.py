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
from nltk.tokenize import word_tokenize
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

FOLDER = './uploads'
PRIVATE = os.getenv('PRIVATE', 'False').lower() in ['true', '1', 'yes']
API_PORT = os.getenv('API_PORT', '7775')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

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

def get_emojis_for_text(text: str) -> List[Dict[str, str]]:
    """Get emoji mappings for words in text using centralized service"""
    if not text or not text.strip():
        return []
    
    print(f"OCR: About to call centralized service with text: '{text[:100]}...'")
    
    try:
        # Call centralized emoji service (Mirror Stage)
        response = requests.post(
            'http://localhost:7776/emoji',
            json={'text': text},
            timeout=5
        )
        
        print(f"OCR: Centralized service responded with status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            mappings_dict = data.get('mappings', {})
            
            # Convert mappings dict to list format expected by the voting system
            mappings = [
                {"word": word, "emoji": emoji}
                for word, emoji in mappings_dict.items()
            ]
            
            print(f"OCR: Centralized service found {len(mappings)} mappings: {mappings}")
            return mappings
        else:
            print(f"OCR: Centralized service returned status {response.status_code}")
            
    except Exception as e:
        print(f"OCR: Failed to call centralized emoji service: {e}")
    
    print("OCR: No emojis found")
    return []

def extract_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words from text using NLTK stopwords"""
    if not text:
        return []
    
    try:
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Filter out stopwords and keep only alphabetic words of 3+ characters
        meaningful_words = [
            word for word in tokens 
            if word.isalpha() and len(word) >= 3 and word not in stop_words
        ]
        
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
        
        # Clean text (remove extra whitespace, normalize)
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
        
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
            "POST /v2/analyze - Extract text and emojis from image URL"
        ]
    })

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API endpoint for direct file path analysis"""
    start_time = time.time()
    
    try:
        # Get file path from query parameters
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                "service": "ocr",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "ocr",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        # Extract text directly from file (no cleanup needed - we don't own the file)
        result = extract_text_paddleocr(file_path, cleanup=False)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "ocr",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'OCR processing failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Create unified prediction format
        predictions = []
        
        text = result.get('text', '')
        raw_text = result.get('raw_text', '')
        has_text = result.get('has_text', False)
        confidence = result.get('confidence', 0.0)
        
        # Text extraction prediction (main OCR result with text regions)
        text_regions = result.get('text_regions', [])
        text_prediction = {
            "type": "text_extraction",
            "text": text,
            "emoji": "💬" if has_text else "",
            "confidence": round(confidence, 3),
            "properties": {
                "has_text": has_text,
                "raw_text": raw_text,
                "cleaned_text": text,
                "engine": "PaddleOCR",
                "processing_method": "gpu_accelerated",
                "text_regions": text_regions
            }
        }
        predictions.append(text_prediction)
        
        # Extract meaningful words and get emojis for them
        if has_text and text:
            print(f"OCR: Extracted text: '{text[:100]}...'")
            meaningful_words = extract_meaningful_words(text)
            print(f"OCR: Found {len(meaningful_words)} meaningful words: {meaningful_words[:10]}")
            
            if meaningful_words:
                # Get emoji mappings for the meaningful words
                word_text = ' '.join(meaningful_words)
                emoji_mappings = get_emojis_for_text(word_text)
                
                if emoji_mappings:
                    # Create emoji mappings prediction
                    emoji_prediction = {
                        "type": "emoji_mappings",
                        "confidence": 1.0,
                        "properties": {
                            "mappings": emoji_mappings,
                            "source": "ocr_content_analysis"
                        }
                    }
                    predictions.append(emoji_prediction)
                    print(f"OCR: Generated {len(emoji_mappings)} emoji mappings from text content")
        
        return jsonify({
            "service": "ocr",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "PaddleOCR",
                    "framework": "PaddlePaddle",
                    "version": "2.x"
                },
                "parameters": {
                    "use_gpu": True,
                    "language": "en",
                    "angle_classification": True
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "service": "ocr",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API endpoint with unified response format"""
    start_time = time.time()
    filepath = None  # Initialize for proper cleanup
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "ocr",
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
            
            # Extract text using PaddleOCR (cleanup=True will handle file removal)
            result = extract_text_paddleocr(filepath)
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "ocr",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'OCR processing failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Create unified prediction format
            predictions = []
            
            text = result.get('text', '')
            raw_text = result.get('raw_text', '')
            has_text = result.get('has_text', False)
            confidence = result.get('confidence', 0.0)
            
            # Text extraction prediction (main OCR result with text regions)
            text_regions = result.get('text_regions', [])
            text_prediction = {
                "type": "text_extraction",
                "text": text,
                "emoji": "💬" if has_text else "",
                "confidence": round(confidence, 3),
                "properties": {
                    "has_text": has_text,
                    "raw_text": raw_text,
                    "cleaned_text": text,
                    "engine": "PaddleOCR",
                    "processing_method": "gpu_accelerated",
                    "text_regions": text_regions
                }
            }
            predictions.append(text_prediction)
            
            # Extract meaningful words and get emojis for them
            if has_text and text:
                print(f"OCR: Extracted text: '{text[:100]}...'")
                meaningful_words = extract_meaningful_words(text)
                print(f"OCR: Found {len(meaningful_words)} meaningful words: {meaningful_words[:10]}")
                
                if meaningful_words:
                    # Get emoji mappings for the meaningful words
                    word_text = ' '.join(meaningful_words)
                    emoji_mappings = get_emojis_for_text(word_text)
                    
                    if emoji_mappings:
                        # Create emoji mappings prediction
                        emoji_prediction = {
                            "type": "emoji_mappings",
                            "confidence": 1.0,
                            "properties": {
                                "mappings": emoji_mappings,
                                "source": "ocr_content_analysis"
                            }
                        }
                        predictions.append(emoji_prediction)
                        print(f"OCR: Generated {len(emoji_mappings)} emoji mappings from text content")
            
            return jsonify({
                "service": "ocr",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "PaddleOCR",
                        "framework": "PaddlePaddle",
                        "version": "2.x"
                    },
                    "parameters": {
                        "use_gpu": True,
                        "language": "en",
                        "angle_classification": True
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "service": "ocr",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        return jsonify({
            "service": "ocr",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500
    
    finally:
        # Ensure cleanup happens regardless of success or failure
        if filepath:
            cleanup_file(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(API_PORT), debug=False)