import asyncio
import base64
import json
import requests
import os
import uuid
import time
import logging
import nltk
from nltk.tokenize import MWETokenizer
from typing import Dict, Any, Optional, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ollama import AsyncClient
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Configuration for emoji downloads (required)
API_HOST = os.getenv('API_HOST')  # Must be set in .env
API_PORT = int(os.getenv('API_PORT'))  # Must be set in .env
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '2.0'))

FOLDER = './uploads'
PRIVATE = os.getenv('PRIVATE', 'False').lower() in ['true', '1', 'yes']
PORT = int(os.getenv('PORT', '7782'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
DEFAULT_TEXT_MODEL = os.getenv('DEFAULT_TEXT_MODEL', 'llama2')
#DEFAULT_VISION_MODEL = os.getenv('DEFAULT_VISION_MODEL', 'llava')
DEFAULT_VISION_MODEL = os.getenv('DEFAULT_VISION_MODEL', 'llava-llama3')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
MAX_RESPONSE_LENGTH = 4000  # Reasonable limit for responses

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

# Common Ollama models - these will be dynamically fetched from Ollama
COMMON_MODELS = {
    "text": ["llama2", "llama2:13b", "mistral", "mixtral", "codellama", "dolphin-mistral", "phi"],
    "vision": ["llava", "llava:13b", "bakllava"],
    "code": ["codellama", "codellama:13b", "deepseek-coder:6.7b"]
}

# Load emoji mappings from central API
emoji_mappings = {}
emoji_tokenizer = None

def load_emoji_mappings():
    """Load fresh emoji mappings from central API"""
    emoji_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"

    try:
        logger.info(f"🔄 LLaMa: Loading fresh emoji mappings from {emoji_url}")
        response = requests.get(emoji_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ LLaMa: Failed to load emoji mappings from {emoji_url}: {e}")
        raise # re-raise to crash the service

def load_mwe_mappings():
    """Load fresh MWE mappings from central API and convert to tuples"""
    mwe_url = f"http://{API_HOST}:{API_PORT}/mwe.txt"
    try:
        logger.info(f"🔄 LLaMa: Loading fresh multi-word expressions (MWE) mappings from {mwe_url}")
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
        logger.error(f"❌ LLaMa: Failed to load multi-word expressions (MWE) mappings from {mwe_url}: {e}")
        raise # re-raise to crash the service

# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()

# Initialize MWE tokenizer with the loaded mappings (already converted to tuples)
emoji_tokenizer = MWETokenizer(mwe_mappings)

# Priority overrides for critical ambiguities - defined once at module level
PRIORITY_OVERRIDES = {
    'glass': '🥛',        # drinking glass, not eyewear
    'glasses': '👓',      # eyewear (explicit to avoid fallback)
    'wood': '🌲',         # base material form
    'wooden': '🌲',       # morphological variant
    'metal': '🔧',        # base material form  
    'metallic': '🔧',     # morphological variant
}

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
        # Strip punctuation from whole text first, then split
        clean_text = text.strip('.,!?;:"()[]{}')
        word_tokens = []
        for token in clean_text.split():
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
        logger.error(f"LLaMa: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}

def get_emojis_for_text(text: str) -> List[Dict[str, str]]:
    """Extract emojis from sentence using local emoji service"""
    logger.debug(f"LLaMa: get_emojis_for_text called with: '{text[:100]}...'")
    if not text or not text.strip():
        return []
    
    try:
        # Use local emoji lookup instead of utils package
        result = lookup_text_for_emojis(text)
        
        # Convert to expected format
        found_mappings = []
        for word, emoji in result["mappings"].items():
            found_mappings.append({"word": word, "emoji": emoji})
        
        logger.debug(f"LLaMa: Found {len(found_mappings)} emoji mappings")
        return found_mappings
        
    except Exception as e:
        logger.error(f"LLaMa: Failed to process emoji mappings: {e}")
        return []

def convert_to_jpg(filepath: str) -> str:
    """Convert image to JPG format - always convert since Ollama works better with JPG"""
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
        logger.error(f"Error converting image to JPG: {e}")
        return filepath  # Return original if conversion fails

def cleanup_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {e}")

# Removed context lookup function - using simple local lookup now

async def get_available_models() -> Dict[str, List[str]]:
    """Get available models from Ollama server"""
    try:
        client = AsyncClient(host=OLLAMA_HOST)
        models_response = await client.list()
        available_models = [model['name'] for model in models_response.get('models', [])]
        
        # Categorize models
        categorized = {
            "text": [m for m in available_models if any(text_model in m for text_model in COMMON_MODELS["text"])],
            "vision": [m for m in available_models if any(vision_model in m for vision_model in COMMON_MODELS["vision"])],
            "code": [m for m in available_models if any(code_model in m for code_model in COMMON_MODELS["code"])],
            "other": [m for m in available_models if not any(
                any(known_model in m for known_model in category_models) 
                for category_models in COMMON_MODELS.values()
            )]
        }
        
        return categorized
    except Exception as e:
        print(f"Failed to get models from Ollama: {e}")
        return COMMON_MODELS

async def process_text_with_ollama(prompt: str, model: str = None) -> Dict[str, Any]:
    """Process text prompt with Ollama"""
    start_time = time.time()
    model = model or DEFAULT_TEXT_MODEL
    
    try:
        client = AsyncClient(host=OLLAMA_HOST)
        message = {'role': 'user', 'content': prompt}
        response = await client.chat(model=model, messages=[message])
        
        output = response['message']['content']
        processing_time = round(time.time() - start_time, 3)
        
        # Truncate if too long
        if len(output) > MAX_RESPONSE_LENGTH:
            output = output[:MAX_RESPONSE_LENGTH] + "... [truncated]"
        
        # Extract emojis from response
        emoji_list = get_emojis_for_text(output)
        
        return {
            "llm": {
                "response": output,
                "model_used": model,
                "prompt_length": len(prompt),
                "response_length": len(output),
                "processing_time": processing_time,
                "emoji_mappings": emoji_list,
                "type": "text",
                "status": "success"
            }
        }
        
    except Exception as e:
        return {
            "error": f"Text generation failed: {str(e)}",
            "status": "error"
        }

async def process_image_with_ollama(image_file: str, prompt: str = None, model: str = None, cleanup: bool = True) -> Dict[str, Any]:
    """Process image with vision model"""
    start_time = time.time()
    model = model or DEFAULT_VISION_MODEL
    prompt = prompt or "What is in this image? One sentence only."
    full_path = os.path.join(FOLDER, image_file)
    
    try:
        # Read and encode image
        with open(full_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        client = AsyncClient(host=OLLAMA_HOST)
        response = await client.generate(
            model=model,
            prompt=prompt,
            images=[image_data],
            stream=False
        )
        
        output = response["response"]
        processing_time = round(time.time() - start_time, 3)
        
        # Truncate if too long
        if len(output) > MAX_RESPONSE_LENGTH:
            output = output[:MAX_RESPONSE_LENGTH] + "... [truncated]"
        
        # Extract emojis from response
        logger.debug(f"LLaMa: About to extract emojis from vision response: '{output[:50]}...'")
        emoji_list = get_emojis_for_text(output)
        logger.debug(f"LLaMa: Vision emoji extraction returned: {emoji_list}")
        
        result = {
            "llm": {
                "response": output,
                "model_used": model,
                "prompt": prompt,
                "response_length": len(output),
                "processing_time": processing_time,
                "emoji_mappings": emoji_list,
                "type": "vision",
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
            "error": f"Image analysis failed: {str(e)}",
            "status": "error"
        }



app = Flask(__name__)

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Ollama service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test Ollama connectivity
    ollama_status = "unknown"
    available_models = {}
    
    try:
        # Simple async call to test Ollama
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        available_models = loop.run_until_complete(get_available_models())
        ollama_status = "connected"
        loop.close()
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy" if ollama_status == "connected" else "degraded",
        "service": "Ollama LLM Integration",
        "ollama": {
            "host": OLLAMA_HOST,
            "status": ollama_status,
            "default_text_model": DEFAULT_TEXT_MODEL,
            "default_vision_model": DEFAULT_VISION_MODEL
        },
        "available_models": available_models,
        "features": {
            "text_generation": True,
            "image_analysis": True,
            "model_selection": True,
            "streaming": False  # Not implemented in this version
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /models - List available models",
            "POST /text - Generate text response",
            "POST /image - Analyze image with text",
            "GET /?url=<image_url> - Analyze image from URL",
            "POST / - Upload and analyze image"
        ]
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(get_available_models())
        loop.close()
        
        return jsonify({
            "models": models,
            "defaults": {
                "text": DEFAULT_TEXT_MODEL,
                "vision": DEFAULT_VISION_MODEL
            },
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch models: {str(e)}",
            "status": "error"
        }), 500

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
                "service": "ollama",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "ollama",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        # Get optional parameters
        prompt = request.args.get('prompt', "What is in this image? One sentence only.")
        model = request.args.get('model', DEFAULT_VISION_MODEL)
        
        # Convert to JPG and copy to FOLDER (Ollama expects files in FOLDER)
        jpg_filepath = convert_to_jpg(file_path)
        temp_filename = str(uuid.uuid4()) + ".jpg"
        temp_filepath = os.path.join(FOLDER, temp_filename)
        
        # Copy the JPG to temp location
        import shutil
        shutil.copy2(jpg_filepath, temp_filepath)
        
        # Process using existing async function (no cleanup - we handle it manually)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_image_with_ollama(temp_filename, prompt, model, cleanup=False))
        loop.close()
        
        # Clean up temporary files
        cleanup_file(temp_filepath)
        if jpg_filepath != file_path:  # Only clean JPG if it was converted
            cleanup_file(jpg_filepath)
        
        if result.get('status') == 'error':
            return jsonify({
                "service": "ollama",
                "status": "error",
                "predictions": [],
                "error": {"message": result.get('error', 'LLM analysis failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Convert to v2 format
        llm_data = result.get('llm', {})
        response_text = llm_data.get('response', '')
        emoji_mappings = llm_data.get('emoji_mappings', [])
        
        # Create unified prediction format
        predictions = []
        
        # Caption prediction
        if response_text:
            prediction = {
                "type": "caption",
                "text": response_text,
                "confidence": 1.0,  # LLM responses are deterministic for given inputs
                "properties": {
                    "model_used": llm_data.get('model_used', model),
                    "prompt": prompt,
                    "response_length": llm_data.get('response_length', len(response_text)),
                    "emojis": [mapping.get('emoji', '') for mapping in emoji_mappings if mapping.get('emoji')]
                }
            }
            predictions.append(prediction)
        
        # Emoji mappings as separate predictions (following BLIP pattern)
        if emoji_mappings:
            prediction = {
                "type": "emoji_mappings",
                "confidence": 1.0,
                "properties": {
                    "mappings": emoji_mappings,
                    "source": "llm_analysis"
                }
            }
            predictions.append(prediction)
        
        return jsonify({
            "service": "ollama",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": llm_data.get('model_used', model),
                    "framework": "Ollama"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V2 file analysis error: {e}")
        return jsonify({
            "service": "ollama",
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
    filepath = None  # Initialize for proper cleanup
    jpg_filepath = None  # Initialize for converted file cleanup
    
    try:
        # Get image URL from query parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({
                "service": "ollama",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        prompt = request.args.get('prompt', "What is in this image? One sentence only.")
        model = request.args.get('model', DEFAULT_VISION_MODEL)
        
        # Download and process image
        try:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(FOLDER, filename)
            
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            
            if len(response.content) > MAX_FILE_SIZE:
                raise ValueError("Downloaded file too large")
            
            with open(filepath, "wb") as file:
                file.write(response.content)
            
            # Convert to JPG if needed
            jpg_filepath = convert_to_jpg(filepath)
            jpg_filename = os.path.basename(jpg_filepath)
            
            # Process using existing async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(process_image_with_ollama(jpg_filename, prompt, model))
            loop.close()
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "ollama",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'LLM analysis failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 format
            llm_data = result.get('llm', {})
            response_text = llm_data.get('response', '')
            emoji_mappings = llm_data.get('emoji_mappings', [])
            
            # Create unified prediction format
            predictions = []
            
            # Caption prediction
            if response_text:
                prediction = {
                    "type": "caption",
                    "text": response_text,
                    "confidence": 1.0,  # LLM responses are deterministic for given inputs
                    "properties": {
                        "model_used": llm_data.get('model_used', model),
                        "prompt": prompt,
                        "response_length": llm_data.get('response_length', len(response_text)),
                        "emojis": [mapping.get('emoji', '') for mapping in emoji_mappings if mapping.get('emoji')]
                    }
                }
                predictions.append(prediction)
            
            # Emoji mappings as separate predictions (following BLIP pattern)
            if emoji_mappings:
                prediction = {
                    "type": "emoji_mappings",
                    "confidence": 1.0,
                    "properties": {
                        "mappings": emoji_mappings,
                        "source": "llm_analysis"
                    }
                }
                predictions.append(prediction)
            
            return jsonify({
                "service": "ollama",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": llm_data.get('model_used', model),
                        "framework": "Ollama"
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image URL {image_url}: {e}")
            return jsonify({
                "service": "ollama",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        logger.error(f"V2 API error: {e}")
        return jsonify({
            "service": "ollama",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500
    
    finally:
        # Ensure cleanup happens regardless of success or failure
        if filepath:
            cleanup_file(filepath)  # Original download
        if jpg_filepath:
            cleanup_file(jpg_filepath)  # Converted JPG

@app.route('/text', methods=['POST'])
def generate_text():
    """Generate text response from prompt"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                "error": "Missing 'prompt' in request body",
                "status": "error"
            }), 400
        
        prompt = data['prompt']
        model = data.get('model', DEFAULT_TEXT_MODEL)
        
        if len(prompt) > 10000:  # Reasonable prompt limit
            return jsonify({
                "error": "Prompt too long (max 10000 characters)",
                "status": "error"
            }), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_text_with_ollama(prompt, model))
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Text generation failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/image', methods=['POST'])
def analyze_image_with_prompt():
    """Analyze uploaded image with custom prompt"""
    try:
        # Get prompt from form data or JSON (handle different content types)
        prompt = request.form.get('prompt')
        model = request.form.get('model')
        
        # Fallback to JSON if form data is empty and content-type is JSON
        if not prompt and request.is_json:
            json_data = request.get_json(silent=True)
            if json_data:
                prompt = json_data.get('prompt')
                model = model or json_data.get('model')
        
        # Set defaults
        prompt = prompt or 'What is in this image? One sentence only.'
        model = model or DEFAULT_VISION_MODEL
        
        if not request.files:
            return jsonify({
                "error": "No image file uploaded",
                "status": "error"
            }), 400
        
        filepath = None  # Initialize for proper cleanup
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
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(process_image_with_ollama(filename, prompt, model))
                loop.close()
                
                # Cleanup after successful processing
                cleanup_file(filepath)
                return jsonify(result)
                
            except Exception as e:
                # Cleanup on error
                if filepath:
                    cleanup_file(filepath)
                return jsonify({
                    "error": f"Image analysis failed: {str(e)}",
                    "status": "error"
                }), 500
        
        return jsonify({
            "error": "No valid image file found",
            "status": "error"
        }), 400
        
    except Exception as e:
        return jsonify({
            "error": f"Image analysis failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    """Legacy endpoint for backwards compatibility"""
    if request.method == 'GET':
        url = request.args.get('url') or request.args.get('img')
        path = request.args.get('path')
        prompt = request.args.get('prompt', 'What is in this image? One sentence only.')

        if url:
            filepath = None  # Initialize for proper cleanup
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
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(process_image_with_ollama(filename, prompt))
                loop.close()
                
                # Cleanup after successful processing
                cleanup_file(filepath)
                return jsonify(result)
                
            except requests.exceptions.RequestException as e:
                # Cleanup on error
                if filepath:
                    cleanup_file(filepath)
                return jsonify({
                    "error": f"Failed to download image: {str(e)}",
                    "status": "error"
                }), 400
            except Exception as e:
                # Cleanup on error
                if filepath:
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
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(process_image_with_ollama(path, prompt))
            loop.close()
            
            return jsonify(result)
            
        else:
            try:
                with open('form.html', 'r') as file:
                    return file.read()
            except FileNotFoundError:
                return '''<!DOCTYPE html>
<html>
<head><title>Ollama LLM API</title></head>
<body>
<h2>Ollama LLM Integration Service</h2>
<h3>Image Analysis</h3>
<form enctype="multipart/form-data" method="POST">
    <input type="file" name="uploadedfile" accept="image/*" required><br><br>
    <input type="submit" value="Analyze Image">
</form>
<h3>Text Generation</h3>
<p>Use POST /text with JSON: {"prompt": "Your question here"}</p>
<p><strong>API Endpoints:</strong></p>
<ul>
    <li>GET /?url=&lt;image_url&gt; - Analyze image from URL</li>
    <li>POST /text - Generate text response</li>
    <li>POST /image - Analyze uploaded image with custom prompt</li>
    <li>GET /models - List available models</li>
    <li>GET /health - Service health check</li>
</ul>
</body>
</html>'''
    
    elif request.method == 'POST':
        # Handle file upload for image analysis
        return analyze_image_with_prompt()

if __name__ == '__main__':
    # Initialize services
    logger.info("Starting Ollama LLM service...")
    logger.info("MAIN BLOCK REACHED - starting initialization")
    
    
    # Using local emoji file - no external dependencies
    
    # Test Ollama connectivity on startup
    logger.info(f"Starting Ollama LLM API on port {PORT}")
    logger.info(f"Ollama host: {OLLAMA_HOST}")
    logger.info(f"Default text model: {DEFAULT_TEXT_MODEL}")
    logger.info(f"Default vision model: {DEFAULT_VISION_MODEL}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(get_available_models())
        loop.close()
        logger.info(f"Available models: {sum(len(v) for v in models.values())} total")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        logger.warning("Make sure Ollama is running and accessible")
    
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    logger.info(f"Starting on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info("Emoji lookup: Local file mode")
    
    app.run(host=host, port=int(PORT), debug=False, threaded=True)
