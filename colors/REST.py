import json
import requests
import os
from dotenv import load_dotenv
import uuid
import re
import time
import tempfile
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import logging

from colors_analyzer import ColorsAnalyzer

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')  
COLOR_SYSTEM_STR = os.getenv('COLOR_SYSTEM')

# Step 2: Validate critical environment variables
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not COLOR_SYSTEM_STR:
    raise ValueError("COLOR_SYSTEM environment variable is required")

# Step 3: Convert to appropriate types after validation
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
PORT = int(PORT_STR)
COLOR_SYSTEM = COLOR_SYSTEM_STR.split(',')
COLOR_SYSTEM = [system.strip().lower() for system in COLOR_SYSTEM]

# Keep the original COLOR_SYSTEM as configured - don't modify it
# The analyzer will handle the mapping internally
pass

FOLDER = './'
UPLOAD_FOLDER = os.path.join(FOLDER, 'uploads')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global analyzer instance
analyzer = None


def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

def get_color_by_name(name, style):
    style = str(style).lower()

    if style == "copic":
        items = copic.items()
    elif style == "prismacolor":
        items = prismacolor.items()

    for color, value in items:
        if name == color:
            return value 
        
def get_color_name(rgb, style):
    global copic
    style = str(style).lower()

    if style == "copic":
        items = copic.items()
    elif style == "prismacolor":
        items = prismacolor.items()

    min_distance = float("inf")
    closest_color = None
    for color, value in items:
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def get_color_name_for_emojis(rgb):
    global colors

    min_distance = float("inf")
    closest_color = None
    for color, value in colors_for_emojis.items():
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

#def get_emoji_by_name(color_name):
#    for color, emoji in color_emojis.items():
#        if color == color_name:
#            return emoji


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]




def color_names(cnames, style):
    pal_json = '  {\n    "palette": [\n'

    pal_json = pal_json + '      {\n        "' + style + '": \n          ['
    #chex = []
    for name in cnames:
        n = get_color_by_name(name, style)
        #pretty name
        if (n):
            cname = format_copic(name)
            cstr = (cname[0] + " (" + cname[1]) + ")"
            pal_json = pal_json + '\n            {\n              "color": "' + cstr + '",\n'

            #print (copic_code)

            #print(n)
            h = rgb2hex(n[0],n[1],n[2])
            pal_json = pal_json + '              "rgb": [' + str(n[0]) + ',' + str(n[1]) + ',' + str(n[2]) + '],\n'
            pal_json = pal_json + '              "hex": "' + h + '"\n            },'

            #chex.append(h)
    #print (chex)
    
    #strip last comma
    pal_json = pal_json.rstrip(',')

    #close json
    pal_json = pal_json + "\n        ]\n      }\n    ]\n  }"
    #print (pal_json)

    return pal_json

def emoji(dominant):
    emo_json = '  {\n    "emo": {\n'
    # get dominant color name for emojis
    color_name_emoji = get_color_name_for_emojis(dominant)
    #print(color_name_emoji)
    emo_json = emo_json + '      "color": "' + color_name_emoji + '",\n'

    #get emoji from color name 
    #color_emoji = get_emoji_by_name(color_name_emoji)
    #print (color_emoji)
    
    #emo_json = emo_json + '      "emoji": "' + color_emoji + '"\n    }\n  }'
    #print (emo_json)

    return emo_json

def dominant(dl_filename):
    dom_json = '  {\n    "dominant": {\n'

    dominant = Haishoku.getDominant(dl_filename)
    dr = dominant[0]
    dg = dominant[1]
    db = dominant[2]
    dhex = rgb2hex(dr,dg,db)
    #print(dhex)

    # Process color systems based on configuration
    dominant_copic = None
    copic_str = None
    dominant_prisma = None
    prisma_str = None
    
    if 'copic' in COLOR_SYSTEM:
        dominant_copic = get_color_name(dominant, "copic")
        copic_d = format_copic(dominant_copic)
        copic_str = (copic_d[0] + " (" + copic_d[1] + ")")
    
    if 'prismacolor' in COLOR_SYSTEM:
        dominant_prisma = get_color_name(dominant, "prismacolor")
        prisma_d = format_copic(dominant_prisma)
        prisma_str = (prisma_d[0] + " (" + prisma_d[1] + ")")

    #build emoji json based on dominant color
    emoji_json = emoji(dominant)
    #print (emoji_json)

    #build json for dominant color
    dom_json = dom_json + '      "copic": "' + copic_str + '",\n'
    #dom_json = dom_json + '      "prismacolor": "' + prisma_str + '",\n'
    dom_json = dom_json + '      "rgb": [' + str(dr) + ',' + str(dg) + ',' + str(db) + '],\n'
    dom_json = dom_json + '      "hex": "' + dhex + '"\n    }\n  }'
    #print (dom_json)

    #build json and return
    consolidated_json = emoji_json + ",\n" + dom_json
    #print(consolidated_json)

    return consolidated_json

def palette(dl_filename):
    #get RGB color palette from Haishoku
    palette = Haishoku.getPalette(dl_filename)
    hexes = []
    cnames_copic = []
    cnames_prisma = []

    for p in palette:
        clr = p[1]
        #print(clr)
        cnames_copic.append(get_color_name(clr, "copic"))
        #cnames_prisma.append(get_color_name(clr, "prismacolor"))

        r = clr[0]
        g = clr[1]
        b = clr[2]

        hex = rgb2hex(r,g,b)
        #print(hex)
        hexes.append(hex)

    cnames_copic = unique(cnames_copic)
    #cnames_prisma = unique(cnames_prisma)

    #print (cnames_copic)
    #print (cnames_prisma)

    cnc = color_names(cnames_copic, "copic")
    #cnp = color_names(cnames_prisma, "prismacolor")
    #print (cnc)
    #print (cnp)

    # COMMENTED OUT: Only returning Copic colors now
    # consolidated = cnc + ",\n" + cnp
    consolidated = cnc  # Only return Copic colors
    return consolidated

def cleanup_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Could not remove file {filepath}: {e}")

def initialize_analyzer() -> bool:
    """Initialize analyzer once at startup - fail fast"""
    global analyzer
    analyzer = ColorsAnalyzer(color_systems=COLOR_SYSTEM)
    return analyzer is not None

def process_image_for_colors(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns color analysis data
    This is the core business logic, separated from HTTP concerns
    """
    try:
        # Use analyzer instead of direct ML logic
        result = analyzer.analyze_colors_from_array(image)
        
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Color analysis failed')
            }
        
        return {
            "success": True,
            "predictions": result.get('predictions', []),
            "processing_time": result.get('processing_time', 0)
        }
        
    except Exception as e:
        logger.error(f"Error processing image for colors: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }

def process_multiregion_colors(image: Image.Image, regions: int = 4) -> Dict[str, Any]:
    """
    Process image using multi-region color analysis for comprehensive palette
    """
    try:
        result = analyzer.analyze_colors_multiregion(image, regions)
        
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Multi-region color analysis failed')
            }
        
        return {
            "success": True,
            "predictions": result.get('predictions', []),
            "processing_time": result.get('processing_time', 0),
            "metadata": result.get('metadata', {})
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Multi-region processing failed: {str(e)}"
        }

def create_colors_response(predictions: list, processing_time: float, additional_metadata: dict = None) -> Dict[str, Any]:
    """Create standardized colors response"""
    metadata = {
        "processing_time": round(processing_time, 3),
        "model_info": {
            "framework": "Haishoku"
        }
    }
    
    # Merge additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return {
        "service": "colors",
        "status": "success", 
        "predictions": predictions,
        "metadata": metadata
    }

def analyze_colors(image_file: str, cleanup: bool = True) -> Dict[str, Any]:
    """Analyze colors in image and return structured response"""
    start_time = time.time()
    # Check if image_file is already a full path or just filename
    if os.path.isabs(image_file) or image_file.startswith('./'):
        full_path = image_file
    else:
        full_path = os.path.join(FOLDER, image_file)
    
    try:
        # Get dominant color directly using Haishoku
        dominant_color = Haishoku.getDominant(full_path)
        dr, dg, db = dominant_color[0], dominant_color[1], dominant_color[2]
        dhex = rgb2hex(dr, dg, db)
        
        # Get dominant color names
        dominant_copic = get_color_name(dominant_color, "copic")
        copic_d = format_copic(dominant_copic)
        copic_str = f"{copic_d[0]} ({copic_d[1]})"
        
        # Get Prismacolor for dominant color
        #dominant_prisma = get_color_name(dominant_color, "prismacolor")
        #prisma_d = format_copic(dominant_prisma)
        #prisma_str = f"{prisma_d[0]} ({prisma_d[1]})"
        
        # Get emoji for dominant color
        color_name_emoji = get_color_name_for_emojis(dominant_color)
        #color_emoji = get_emoji_by_name(color_name_emoji)
        
        # Get color palette
        palette_colors = Haishoku.getPalette(full_path)
        cnames_copic = []
        cnames_prisma = []
        
        for p in palette_colors:
            clr = p[1]
            if 'copic' in COLOR_SYSTEM:
                cnames_copic.append(get_color_name(clr, "copic"))
            if 'prismacolor' in COLOR_SYSTEM:
                cnames_prisma.append(get_color_name(clr, "prismacolor"))
        
        # Remove duplicates
        if 'copic' in COLOR_SYSTEM:
            cnames_copic = unique(cnames_copic)
        if 'prismacolor' in COLOR_SYSTEM:
            cnames_prisma = unique(cnames_prisma)
        
        # Build palette arrays
        copic_palette = []
        prisma_palette = []
        
        if 'copic' in COLOR_SYSTEM:
            for name in cnames_copic:
                color_rgb = get_color_by_name(name, "copic")
                if color_rgb:
                    cname = format_copic(name)
                    cstr = f"{cname[0]} ({cname[1]})"
                    hex_val = rgb2hex(color_rgb[0], color_rgb[1], color_rgb[2])
                    copic_palette.append({
                        "color": cstr,
                        "rgb": list(color_rgb),
                        "hex": hex_val
                    })
        
        if 'prismacolor' in COLOR_SYSTEM:
            for name in cnames_prisma:
                color_rgb = get_color_by_name(name, "prismacolor")
                if color_rgb:
                    cname = format_copic(name)
                    cstr = f"{cname[0]} ({cname[1]})"
                    hex_val = rgb2hex(color_rgb[0], color_rgb[1], color_rgb[2])
                    prisma_palette.append({
                        "color": cstr,
                        "rgb": list(color_rgb),
                        "hex": hex_val
                    })
        
        analysis_time = round(time.time() - start_time, 3)
        
        # Build dominant color info based on enabled systems
        dominant_info = {
            "rgb": [dr, dg, db],
            "hex": dhex
        }
        if 'copic' in COLOR_SYSTEM and copic_str:
            dominant_info["copic"] = copic_str
        if 'prismacolor' in COLOR_SYSTEM and prisma_str:
            dominant_info["prismacolor"] = prisma_str
        
        # Build palette based on enabled systems
        palette_info = []
        if 'copic' in COLOR_SYSTEM:
            palette_info.append({"copic": copic_palette})
        if 'prismacolor' in COLOR_SYSTEM:
            palette_info.append({"prismacolor": prisma_palette})
        
        # Build color systems list
        active_systems = [system.title() for system in COLOR_SYSTEM]
        
        result = {
            "colors": {
                "dominant": dominant_info,
                "emoji": {
                    "color": color_name_emoji
                },
                "palette": palette_info,
                "analysis_info": {
                    "framework": "Haishoku + PIL",
                    "color_systems": active_systems,
                    "analysis_time": analysis_time
                },
                "status": "success"
            }
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            cleanup_file(full_path)
        return result
        
    except Exception as e:
        # Cleanup (only for temporary files)
        if cleanup:
            cleanup_file(full_path)
        return {
            "error": f"Color analysis failed: {str(e)}",
            "status": "error"
        }

app = Flask(__name__)

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Colors service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Color Analysis",
        "features": {
            "color_systems": ["Copic"], # "Prismacolor" removed
            "analysis_types": ["dominant_color", "color_palette", "emoji_mapping"],
            "supported_formats": ["JPEG", "PNG", "GIF", "BMP"]
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /v3/analyze?url=<image_url> - Analyze colors from URL", 
            "GET /v3/analyze?file=<file_path> - Analyze colors from file",
            "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
            "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
        ]
    })


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "colors",
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
                    # Add browser-like headers to bypass anti-hotlinking protection
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Referer': 'https://www.google.com/'
                    }
                    response = requests.get(url, headers=headers, timeout=10)
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
        # Check for multi-region analysis parameter
        regions = request.args.get('regions', type=int)
        if regions and regions > 1:
            processing_result = process_multiregion_colors(image, regions)
        else:
            processing_result = process_image_for_colors(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_colors_response(
            processing_result["predictions"],
            processing_result["processing_time"],
            processing_result.get("metadata", {})
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - redirect to new analyze endpoint"""
    return analyze()

@app.route('/v3/analyze_old', methods=['GET', 'POST'])
def analyze_v3():
    """Unified V3 API endpoint for URL, file path, and POST file upload analysis"""
    import time
    start_time = time.time()
    
    try:
        # Handle POST file upload
        if request.method == 'POST':
            # Check for file upload
            if 'file' not in request.files:
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file provided in POST request"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file selected"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            # Validate file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            
            # Process directly from memory
            try:
                from io import BytesIO
                image = Image.open(BytesIO(file.read())).convert('RGB')
                result = analyze_colors_from_image(image)
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process uploaded image: {str(e)}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
        
        # Handle GET requests
        # Get parameters from query string
        url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input - exactly one parameter required
        if not url and not file_path:
            return jsonify({
                "service": "colors",
                "status": "error", 
                "predictions": [],
                "error": {"message": "Must provide either 'url' or 'file' parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        if url and file_path:
            return jsonify({
                "service": "colors",
                "status": "error",
                "predictions": [],
                "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Handle URL input
        if url:
            try:
                filename = uuid.uuid4().hex + ".jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                # Add browser-like headers to bypass anti-hotlinking protection
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Referer': 'https://www.google.com/'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                if len(response.content) > MAX_FILE_SIZE:
                    raise ValueError("Downloaded file too large")
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                # Analyze using existing function
                result = analyze_colors(filepath)
                
                if result.get('status') == 'error':
                    return jsonify({
                        "service": "colors",
                        "status": "error",
                        "predictions": [],
                        "error": {"message": result.get('error', 'Color analysis failed')},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 500
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    "service": "colors",
                    "status": "error", 
                    "predictions": [],
                    "error": {"message": f"Failed to download image: {str(e)}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 400
            except Exception as e:
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to process image: {str(e)}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
        
        # Handle file path input
        elif file_path:
            # Validate file path
            if not os.path.exists(file_path):
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File not found: {file_path}"},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 404
            
            # Analyze directly from file (no cleanup needed - we don't own the file)
            result = analyze_colors(file_path, cleanup=False)
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "colors",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Color analysis failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
        
        # Convert to unified V3 response format
        colors_data = result.get('colors', {})
        dominant = colors_data.get('dominant', {})
        emoji_info = colors_data.get('emoji', {})
        palette = colors_data.get('palette', [])
        
        # Create unified color prediction with temperature analysis
        predictions = []
        
        if dominant and palette:
            # Get copic palette data
            copic_palette_data = palette[0].get('copic', []) if palette else []
            
            # Build clean palette with temperature analysis and Prismacolor matches
            clean_palette = []
            for color_info in copic_palette_data:
                # Get Prismacolor equivalent for this palette color
                rgb_values = color_info.get('rgb', [])
                prismacolor_match = ''
                if rgb_values:
                    prismacolor_match = get_color_name(rgb_values, "prismacolor") or ''
                
                clean_palette.append({
                    "copic": color_info.get('color', ''),
                    "hex": color_info.get('hex', ''),
                    "prismacolor": prismacolor_match,
                    "temperature": get_color_temperature(color_info.get('color', ''))
                })
            
            # Calculate palette temperature
            palette_temp = calculate_palette_temperature(copic_palette_data)
            
            # Get primary color temperature
            primary_copic = dominant.get('copic', '')
            primary_temp = get_color_temperature(primary_copic)
            
            # Get primary color Prismacolor match
            primary_rgb = dominant.get('rgb', [])
            primary_prismacolor = ''
            if primary_rgb:
                primary_prismacolor = get_color_name(primary_rgb, "prismacolor") or ''
            
            # Create unified prediction with clear semantic groupings
            color_prediction = {
                "primary": {
                    "copic": primary_copic,
                    "hex": dominant.get('hex', ''),
                    "prismacolor": primary_prismacolor,
                    "temperature": primary_temp
                },
                "palette": {
                    "temperature": palette_temp,
                    "colors": clean_palette
                }
            }
            predictions.append(color_prediction)
        
        return jsonify({
            "service": "colors",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "Haishoku + PIL"
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "service": "colors",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
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
    """V2 compatibility - translate parameters to V3 format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Call V3 with translated parameters
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze()
    else:
        # Let V3 handle validation errors
        with app.test_request_context('/v3/analyze'):
            return analyze()


if __name__ == '__main__':
    # Initialize analyzer
    print("Starting Colors service...")
    
    analyzer_loaded = initialize_analyzer()
    
    if not analyzer_loaded:
        print("Failed to load Colors analyzer. Service will run but analysis will fail.")
    
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting Color Analysis API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"Configured color systems: {', '.join([s.title() for s in COLOR_SYSTEM])}")
    print(f"Analyzer loaded: {analyzer_loaded}")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)


