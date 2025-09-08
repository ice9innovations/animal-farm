#!/usr/bin/env python3
"""
BLIP2 REST API Service - Caption generation using LAVIS BLIP2

Provides REST endpoints for image caption generation using BLIP2 model.
Compatible with the Animal Farm pipeline API format.
"""

import json
import requests
import os
import sys
import logging
import random
import time
import nltk
from nltk.tokenize import MWETokenizer
from typing import List, Optional, Dict, Any
from io import BytesIO

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Configuration for GitHub raw file downloads (optional - fallback to local config)
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from blip2_analyzer import Blip2Analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
PRIVATE_STR = os.getenv('PRIVATE', 'false')
PORT_STR = os.getenv('PORT', '7777')

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load emoji mappings from local JSON file
emoji_mappings = {}
emoji_tokenizer = None

# Global BLIP2 analyzer - initialize once at startup
blip2_analyzer = None

# Priority overrides for critical ambiguities
PRIORITY_OVERRIDES = {
    'glass': '🥛',        # drinking glass, not eyewear
    'glasses': '👓',      # eyewear (explicit to avoid fallback)
    'wood': '🌲',         # base material form
    'wooden': '🌲',       # morphological variant
    'metal': '🔧',        # base material form  
    'metallic': '🔧',     # morphological variant
}

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        
        try:
            logger.info(f"🔄 BLIP2: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 BLIP2: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  BLIP2: Failed to cache emoji mappings: {cache_error}")
            
            logger.info("✅ BLIP2: Successfully loaded emoji mappings from GitHub")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  BLIP2: Failed to load emoji mappings from GitHub: {e}")
            logger.info("🔄 BLIP2: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 BLIP2: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file
    try:
        logger.info(f"🔄 BLIP2: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("✅ BLIP2: Successfully loaded emoji mappings from local cache")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ BLIP2: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in LAVIS directory: {e}")

def load_mwe_mappings():
    """Load MWE mappings from GitHub raw files with local caching and convert to tuples"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'mwe.txt')
    mwe_text = []
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
        
        try:
            logger.info(f"🔄 BLIP2: Loading fresh multi-word expressions (MWE) mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            mwe_text = response.text.splitlines()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"💾 BLIP2: Cached MWE mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  BLIP2: Failed to cache MWE mappings: {cache_error}")
            
            logger.info("✅ BLIP2: Successfully loaded MWE mappings from GitHub")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  BLIP2: Failed to load MWE mappings from GitHub: {e}")
            logger.info("🔄 BLIP2: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 BLIP2: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file if GitHub failed or AUTO_UPDATE is disabled
    if not mwe_text:
        try:
            logger.info(f"🔄 BLIP2: Loading MWE mappings from local cache: {local_cache_path}")
            with open(local_cache_path, 'r', encoding='utf-8') as f:
                mwe_text = f.read().splitlines()
            logger.info("✅ BLIP2: Successfully loaded MWE mappings from local cache")
        except FileNotFoundError as e:
            logger.error(f"❌ BLIP2: Failed to load local MWE mappings from {local_cache_path}: {e}")
            if AUTO_UPDATE:
                raise Exception(f"Failed to load MWE mappings from both GitHub and local cache: {e}")
            else:
                raise Exception(f"Failed to load MWE mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide mwe.txt in LAVIS directory: {e}")

    # Convert to tuples for MWETokenizer
    mwe_tuples = []
    for line in mwe_text:
        if line.strip():  # Skip empty lines
            # Convert underscore format to word tuples (e.g., "street_sign" -> ("street", "sign"))
            mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
    
    return mwe_tuples

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def initialize_blip2_analyzer() -> bool:
    """Initialize BLIP2 analyzer once at startup - fail fast"""
    global blip2_analyzer
    try:
        logger.info("Initializing BLIP2 Analyzer...")
        
        blip2_analyzer = Blip2Analyzer(
            model_name="blip_caption",
            model_type="large_coco"  # Using large model
        )
        
        # Initialize the model
        if not blip2_analyzer.initialize():
            logger.error("❌ Failed to initialize BLIP2 Analyzer")
            return False
            
        logger.info("✅ BLIP2 Analyzer initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing BLIP2 Analyzer: {str(e)}")
        return False

# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()

# Initialize MWE tokenizer with the loaded mappings (already converted to tuples)
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')

def get_emoji_for_word(word: str) -> str:
    """Get emoji for a single word with morphological variations"""
    if not word:
        return None

    word_clean = word.lower().strip()

    if word_clean in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[word_clean]

    # Try exact match in main mappings
    if word_clean in emoji_mappings:
        return emoji_mappings[word_clean]

    # Try singular form for common plurals
    if word_clean.endswith('s') and len(word_clean) > 3:
        singular = word_clean[:-1]
        if singular in emoji_mappings:
            return emoji_mappings[singular]
    
    return None

def lookup_text_for_emojis(text: str) -> Dict[str, Any]:
    """Look up emojis for text with tokenization - returns same format as utils"""
    if not text or not text.strip():
        return {"mappings": {}, "found_emojis": []}
    
    try:
        # Tokenize text with MWE detection
        word_tokens = []
        for token in text.split():
            # Remove common punctuation
            token = token.strip('.,!?;:"()[]{}')
            if token:
                word_tokens.append(token)
        
        tokens = emoji_tokenizer.tokenize(word_tokens)
        
        # Look up emojis for each token
        mappings = {}
        found_emojis = []
        
        for token in tokens:
            emoji = get_emoji_for_word(token)
            if emoji:
                mappings[token] = emoji
                if emoji not in found_emojis:
                    found_emojis.append(emoji)
        
        return {"mappings": mappings, "found_emojis": found_emojis}
        
    except Exception as e:
        logger.error(f"BLIP2: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}

def get_emojis_and_mappings_for_caption(caption: str) -> tuple[List[str], Dict[str, str]]:
    """Extract emojis and word mappings from caption using local emoji service (optimized - no HTTP requests)"""
    logger.debug(f"BLIP2: get_emojis_and_mappings_for_caption called with: '{caption[:100]}...'")
    if not caption:
        return [], {}
    
    logger.debug("BLIP2: Using local emoji service (no HTTP requests)...")
    
    try:
        # Use local emoji lookup instead of HTTP requests
        result = lookup_text_for_emojis(caption)
        
        emojis = result["found_emojis"]
        mappings = result["mappings"]
        
        logger.debug(f"BLIP2: Found {len(mappings)} mappings: {mappings}")
        logger.debug(f"BLIP2: Returning {len(emojis)} emojis: {emojis}")
        logger.debug(f"BLIP2: Returning {len(mappings)} mappings: {mappings}")
        
        return emojis[:3], mappings  # Limit to 3 emojis as before
        
    except Exception as e:
        logger.error(f"BLIP2: Failed to use local emoji service: {e}")
        
    # Return empty lists if service fails
    logger.debug("BLIP2: No emojis found")
    return [], {}

def create_blip2_response(caption: str, processing_time: float, word_mappings: Dict[str, str] = None) -> Dict[str, Any]:
    """Create standardized BLIP2 response with metadata"""
    is_shiny, shiny_roll = check_shiny()
    
    # Create prediction in Animal Farm format
    prediction = {
        "text": caption
    }
    
    # Add emoji mappings in BLIP format if provided
    if word_mappings:
        emoji_mappings = []
        for word, emoji in word_mappings.items():
            emoji_mappings.append({
                "emoji": emoji,
                "word": word
            })
        prediction["emoji_mappings"] = emoji_mappings
    
    # Add shiny flag for rare detections
    if is_shiny:
        prediction["shiny"] = True
        logger.info(f"✨ SHINY CAPTION GENERATED! Roll: {shiny_roll} ✨")
    
    return {
        "service": "blip2",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {"framework": "BLIP2 (Bootstrapping Language-Image Pre-training v2)"}
        }
    }

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        import requests
        headers = {'User-Agent': 'BLIP2 Caption Generation Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        # Return PIL Image directly from bytes
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_for_caption(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns caption data
    This is the core business logic, separated from HTTP concerns
    """
    try:
        # Generate caption using analyzer
        result = blip2_analyzer.analyze_caption_from_array(image)
        
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Caption generation failed')
            }
        
        # Get caption and emoji mappings
        caption = result.get('caption', '')
        emojis, word_mappings = get_emojis_and_mappings_for_caption(caption)
        
        return {
            "success": True,
            "caption": caption,
            "emojis": emojis,
            "word_mappings": word_mappings
        }
        
    except Exception as e:
        logger.error(f"Error processing image for caption: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("BLIP2 service: CORS enabled for direct browser communication")

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
    try:
        model_status = "loaded" if blip2_analyzer and blip2_analyzer.model else "not_loaded"
        device_str = str(blip2_analyzer.device) if blip2_analyzer else "unknown"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "device": device_str
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"error": "Health check failed", "status": "error"}), 500

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - redirect to new analyze endpoint"""
    if request.method == 'POST':
        return analyze()
    else:
        with app.test_request_context('/analyze', query_string=request.args):
            return analyze()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "blip2",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        image = None
        
        # Step 1: Get image into memory from any source
        if request.method == 'POST' and 'file' in request.files:
            # Handle file upload - direct to memory
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
            # Read directly into memory
            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")
            
            image = Image.open(BytesIO(file_data)).convert('RGB')
        
        else:
            # Handle URL or file parameter
            url = request.args.get('url')
            file = request.args.get('file')
            
            if not url and not file:
                return error_response("Must provide either url or file parameter, or POST a file")
            
            if url and file:
                return error_response("Cannot provide both url and file parameters")
            
            if url:
                try:
                    image = download_image_from_url(url)
                except Exception as e:
                    return error_response(str(e))
                
            else:  # file parameter
                try:
                    image = validate_image_file(file)
                except Exception as e:
                    return error_response(str(e))
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_caption(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_blip2_response(
            processing_result["caption"],
            time.time() - start_time,
            processing_result.get("word_mappings", {})
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

if __name__ == '__main__':
    # Initialize model
    logger.info("Starting BLIP2 service...")
    
    model_loaded = initialize_blip2_analyzer()
    
    if not model_loaded:
        logger.error("Failed to initialize BLIP2 analyzer. Service cannot function.")
        exit(1)
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting BLIP2 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )