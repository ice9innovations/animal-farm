import json
import requests
import os
import sys
import uuid
import logging
import nltk
from nltk.tokenize import MWETokenizer
from typing import List, Optional, Dict, Any

# Add current directory to Python path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')
API_PORT = os.getenv('API_PORT')
API_TIMEOUT = os.getenv('API_TIMEOUT')

# Validate critical environment variables
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT:
    raise ValueError("API_TIMEOUT environment variable is required")

# Convert to appropriate types after validation
API_PORT = int(API_PORT)
API_TIMEOUT = float(API_TIMEOUT)

import torch
from urllib.parse import urlparse, parse_qs
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# No more mirror-stage dependencies - using direct emoji file loading

# Configuration
IMAGE_SIZE = 384
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load emoji mappings from local JSON file
emoji_mappings = {}
emoji_tokenizer = None

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
    """Load fresh emoji mappings from central API"""
    emoji_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"

    try:
        logger.info(f"🔄 BLIP: Loading fresh emoji mappings from {emoji_url}")
        response = requests.get(emoji_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ BLIP: Failed to load emoji mappings from {emoji_url}: {e}")
        raise # re-raise to crash the service

def load_mwe_mappings():
    """Load fresh MWE mappings from central API and convert to tuples"""
    mwe_url = f"http://{API_HOST}:{API_PORT}/mwe.txt"
    try:
        logger.info(f"🔄 BLIP: Loading fresh multi-word expressions (MWE) mappings from {mwe_url}")
        response = requests.get(mwe_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        mwe_text = response.text.splitlines()

        # Convert to tuples for MWETokenizer
        mwe_tuples = []
        for line in mwe_text:
            if line.strip():  # Skip empty lines
                # Convert underscore format to word tuples (e.g., "street_sign" -> ("street", "sign"))
                mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
        
        return mwe_tuples
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ BLIP: Failed to load multi-word expressions (MWE) mappings from {mwe_url}: {e}")
        raise # re-raise to crash the service

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
        logger.error(f"BLIP: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")

logger.info(f"Using device: {device}")

# Model configuration
MODEL_URLS = {
    'base_14M': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth',
    'base': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth',
    'large': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
    'base_capfilt_large': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
}

model = None

def load_model() -> bool:
    """Load BLIP model with error handling"""
    global model
    try:
        # Try to import BLIP model
        try:
            from models.blip import blip_decoder
        except ImportError:
            logger.error("BLIP models not found. Please install BLIP or ensure models directory exists.")
            return False
            
        # Load the required model
        model_path = "./model_base_capfilt_large.pth"
        
        if not os.path.exists(model_path):
            logger.error(f"BLIP model not found: {model_path}")
            logger.error("Please download the required model: model_base_capfilt_large.pth")
            return False
            
        logger.info(f"Loading BLIP model from {model_path}")
        # Use base ViT for all models (the 'large' refers to caption filtering, not ViT size)
        model = blip_decoder(pretrained=model_path, image_size=IMAGE_SIZE, vit='base')
        model.eval()
        model = model.to(device)
        
        # Temporarily disable FP16 for BLIP - causing tensor type mismatch errors  
        # Similar to YOLO, BLIP architecture has FP16 incompatibility issues
        logger.info("Using FP32 for BLIP model stability")
        
        logger.info("BLIP model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load BLIP model: {e}")
        return False


def get_emojis_and_mappings_for_caption(caption: str) -> tuple[List[str], Dict[str, str]]:
    """Extract emojis and word mappings from caption using local emoji service (optimized - no HTTP requests)"""
    logger.debug(f"BLIP: get_emojis_and_mappings_for_caption called with: '{caption[:100]}...'")
    if not caption:
        return [], {}
    
    logger.debug("BLIP: Using local emoji service (no HTTP requests)...")
    
    try:
        # Use local emoji lookup instead of HTTP requests
        result = lookup_text_for_emojis(caption)
        
        emojis = result["found_emojis"]
        mappings = result["mappings"]
        
        logger.debug(f"BLIP: Found {len(mappings)} mappings: {mappings}")
        logger.debug(f"BLIP: Returning {len(emojis)} emojis: {emojis}")
        logger.debug(f"BLIP: Returning {len(mappings)} mappings: {mappings}")
        
        return emojis[:3], mappings  # Limit to 3 emojis as before
        
    except Exception as e:
        logger.error(f"BLIP: Failed to use local emoji service: {e}")
        
    # Return empty lists if service fails
    logger.debug("BLIP: No emojis found")
    return [], {}

# Removed old mirror-stage lookup function - using local file now

def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    """Preprocess image for BLIP model"""
    try:
        logger.info(f"Preprocessing image: {image_path}")
        
        # Open and validate image
        raw_image = Image.open(image_path)
        
        # Ensure RGB format (handle different image modes)
        if raw_image.mode != 'RGB':
            logger.info(f"Converting image from {raw_image.mode} to RGB")
            raw_image = raw_image.convert('RGB')
        
        # Log original image dimensions
        logger.info(f"Original image size: {raw_image.size}")
        
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image = transform(raw_image)
        
        # Log tensor shape before adding batch dimension
        logger.info(f"Image tensor shape after transform: {image.shape}")
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)
        
        # Convert to half precision if model is FP16
        if device.type == 'cuda' and hasattr(model, 'dtype') and model.dtype == torch.float16:
            image = image.half()
        
        # Log final tensor shape
        logger.info(f"Final image tensor shape: {image.shape}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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

def generate_caption(image_path: str, cleanup: bool = True) -> Dict[str, Any]:
    """Generate caption for image using BLIP model"""
    if not model:
        return {"error": "Model not loaded", "status": "error"}
        
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {"error": "Image file not found", "status": "error"}
            
        if not validate_file_size(image_path):
            return {"error": "File too large", "status": "error"}
            
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return {"error": "Failed to preprocess image", "status": "error"}
            
        # Generate caption with FP16 optimization
        with torch.no_grad():
            # Use autocast for FP16 inference if model is FP16
            use_autocast = (device.type == 'cuda' and hasattr(model, 'dtype') and 
                          model.dtype == torch.float16)
            # Ensure proper tensor format
            if image_tensor.dim() != 4:
                logger.error(f"Invalid image tensor dimensions: {image_tensor.shape}")
                return {"error": "Invalid image format", "status": "error"}
                
            # Try with num_beams=1 first (primary fix)
            try:
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        caption = model.generate(
                            image_tensor, 
                            sample=False, 
                            num_beams=1,  # Fix for tensor size mismatch issue
                            max_length=20, 
                            min_length=5
                        )
                else:
                    caption = model.generate(
                        image_tensor, 
                        sample=False, 
                        num_beams=1,  # Fix for tensor size mismatch issue
                        max_length=20, 
                        min_length=5
                    )
            except RuntimeError as e:
                if "size of tensor" in str(e):
                    logger.warning("Beam search failed, trying with sample=True")
                    # Fallback: use sampling instead of beam search
                    if use_autocast:
                        with torch.cuda.amp.autocast():
                            caption = model.generate(
                                image_tensor, 
                                sample=True, 
                                num_beams=1,
                                max_length=20, 
                                min_length=5
                            )
                    else:
                        caption = model.generate(
                            image_tensor, 
                            sample=True, 
                            num_beams=1,
                            max_length=20, 
                            min_length=5
                        )
                else:
                    raise e
            
        caption_text = caption[0] if caption else "No caption generated"
        logger.info(f"Generated caption: {caption_text}")
        
        # Return simple caption for V2 endpoint processing
        result = {"caption": caption_text, "status": "success"}
        
        # Cleanup (only for temporary files)
        if cleanup:
            try:
                if os.path.exists(image_path) and image_path.startswith(UPLOAD_FOLDER):
                    os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {image_path}: {e}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating caption for {image_path}: {e}")
        return {"error": f"Caption generation failed: {str(e)}", "status": "error"}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("BLIP service: CORS enabled for direct browser communication")

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
        "device": str(device)
    })

# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    import time
    from flask import request
    
    # Get V2 parameter
    image_url = request.args.get('image_url')
    
    if image_url:
        # Create new request args with V3 parameter name
        new_args = request.args.copy()
        new_args = new_args.to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Create a mock request object for V3
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # No parameters - let V3 handle the error
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v2/analyze_file', methods=['GET']) 
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    import time
    from flask import request
    
    # Get V2 parameter
    file_path = request.args.get('file_path')
    
    if file_path:
        # Create new request args with V3 parameter name
        new_args = {'file': file_path}
        
        # Create a mock request object for V3
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # No parameters - let V3 handle the error
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v3/analyze', methods=['GET'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    import time
    start_time = time.time()
    
    try:
        # Get input parameters - support both url and file
        url = request.args.get('url')
        file = request.args.get('file')
        
        # Validate input - exactly one parameter must be provided
        if not url and not file:
            return jsonify({
                "service": "blip",
                "status": "error",
                "predictions": [],
                "error": {"message": "Must provide either url or file parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        if url and file:
            return jsonify({
                "service": "blip",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both url and file parameters"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Handle URL input
        if url:
            filepath = None
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError("Invalid URL format")
                
                # Download image
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError("URL does not point to an image")
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if not validate_file_size(filepath):
                    os.remove(filepath)
                    filepath = None  # Mark as already removed
                    raise ValueError("Downloaded file too large")
                
                # Generate caption using existing function (with cleanup)
                result = generate_caption(filepath, cleanup=True)
                filepath = None  # generate_caption handles cleanup
                
            except Exception as e:
                logger.error(f"Error processing image URL {url}: {e}")
                return jsonify({
                    "service": "blip",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process image URL: {str(e)}"},
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
        
        # Handle file path input
        elif file:
            # Validate file path
            if not os.path.exists(file):
                return jsonify({
                    "service": "blip",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 404
            
            if not is_allowed_file(file):
                return jsonify({
                    "service": "blip",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "File type not allowed"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            # Generate caption directly from file (no cleanup - we don't own the file)
            result = generate_caption(file, cleanup=False)
        
        # Common processing for both input types
        if result.get('status') == 'error':
            return jsonify({
                "service": "blip",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'Caption generation failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        caption = result.get('caption', '')
        
        # Get proper emoji mappings
        emojis, word_mappings = get_emojis_and_mappings_for_caption(caption)
        
        # Create unified prediction format
        predictions = []
        if caption:
            # Build emoji mappings from word mappings
            emoji_mappings = []
            if word_mappings:
                for word, emoji in word_mappings.items():
                    emoji_mappings.append({
                        "word": word,
                        "emoji": emoji
                    })
            
            predictions.append({
                "text": caption,
                "emoji_mappings": emoji_mappings
            })
        
        return jsonify({
            "service": "blip",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "Salesforce"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V2 unified API error: {e}")
        return jsonify({
            "service": "blip",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500


if __name__ == '__main__':
    # Initialize model and emoji data
    logger.info("Starting BLIP service...")
    
    model_loaded = load_model()
    
    # Using local emoji file - no external dependencies
    
    if not model_loaded:
        logger.error("Failed to load BLIP model. Service will run but caption generation will fail.")
        logger.error("Please ensure BLIP models are downloaded and available.")
        
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting BLIP service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info("Emoji lookup: Local file mode")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
