import json
import requests
import os
from dotenv import load_dotenv
import uuid
import re
import time
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from haishoku.haishoku import Haishoku

from color_names import *

load_dotenv()

FOLDER = './'
UPLOAD_FOLDER = os.path.join(FOLDER, 'uploads')
PRIVATE = os.getenv('PRIVATE', 'False').lower() in ['true', '1', 'yes']
PORT = os.getenv('PORT', '7770')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Color system configuration
COLOR_SYSTEM = os.getenv('COLOR_SYSTEM', 'copic').split(',')
COLOR_SYSTEM = [system.strip().lower() for system in COLOR_SYSTEM]

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

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

def format_copic(name):
    copic = []
    copic_name = re.sub('[\(\[].*?[\)\]]', '', str(name)).strip()
    copic_code = name.replace(copic_name,"").replace("(","").replace(")","").strip()

    copic.append(copic_name)
    copic.append(copic_code)
    
    return copic

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
            "GET /?url=<image_url> - Analyze colors from URL",
            "GET /?path=<local_path> - Analyze colors from local file (if not private)",
            "POST / - Upload and analyze colors in image"
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
                "service": "colors",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
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
        
        # Convert to v2 format
        colors_data = result.get('colors', {})
        dominant = colors_data.get('dominant', {})
        emoji_info = colors_data.get('emoji', {})
        palette = colors_data.get('palette', [])
        
        # Create unified prediction format with separate predictions for each analysis type
        predictions = []
        
        # Primary color prediction (dominant hex color)
        if dominant:
            primary_prediction = {
                "type": "primary_color",
                "label": f"Primary Color",
                "value": dominant.get('hex', ''),
                "properties": {
                    "rgb": dominant.get('rgb', [])
                }
            }
            predictions.append(primary_prediction)
        
        # Copic color analysis prediction
        if dominant and palette:
            copic_prediction = {
                "type": "copic_analysis",
                "label": dominant.get('copic', ''),
                "value": dominant.get('hex', ''),
                "properties": {
                    "palette": palette[0].get('copic', []) if palette else []
                }
            }
            predictions.append(copic_prediction)
        
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
                "service": "colors",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Download and process image
        try:
            filename = uuid.uuid4().hex + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            response = requests.get(image_url, timeout=10)
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
            
            # Convert to v2 format
            colors_data = result.get('colors', {})
            dominant = colors_data.get('dominant', {})
            emoji_info = colors_data.get('emoji', {})
            palette = colors_data.get('palette', [])
            
            # Create unified prediction format with separate predictions for each analysis type
            predictions = []
            
            # Primary color prediction (dominant hex color)
            if dominant:
                primary_prediction = {
                    "type": "primary_color",
                    "label": f"Primary Color",
                    "value": dominant.get('hex', ''),
                    "properties": {
                        "rgb": dominant.get('rgb', [])
                    }
                }
                predictions.append(primary_prediction)
            
            # Copic color analysis prediction
            if dominant and palette:
                copic_prediction = {
                    "type": "copic_analysis",
                    "label": dominant.get('copic', ''),
                    "value": dominant.get('hex', ''),
                    "properties": {
                        "palette": palette[0].get('copic', []) if palette else []
                    }
                }
                predictions.append(copic_prediction)
            
            # Prismacolor analysis prediction  
            #if dominant and len(palette) > 1:
            #    prismacolor_prediction = {
            #        "type": "prismacolor_analysis", 
            #        "label": dominant.get('prismacolor', ''),
            #        "value": dominant.get('hex', ''),
            #        "properties": {
            #            "palette": palette[1].get('prismacolor', []) if len(palette) > 1 else []
            #        }
            #    }
            #    predictions.append(prismacolor_prediction)
            
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
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        return jsonify({
            "service": "colors",
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
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                if len(response.content) > MAX_FILE_SIZE:
                    return jsonify({
                        "error": f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                        "status": "error"
                    }), 400
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                result = jsonify(analyze_colors(filepath))
                # analyze_colors already does cleanup
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
                
            return jsonify(analyze_colors(path))
            
        else:
            try:
                with open('form.html', 'r') as file:
                    return file.read()
            except FileNotFoundError:
                return '''<!DOCTYPE html>
<html>
<head><title>Color Analysis API</title></head>
<body>
<h2>Color Analysis Service</h2>
<form enctype="multipart/form-data" method="POST">
    <input type="file" name="uploadedfile" accept="image/*" required><br><br>
    <input type="submit" value="Analyze Colors">
</form>
<p><strong>API Usage:</strong></p>
<ul>
    <li>GET /?url=&lt;image_url&gt; - Analyze colors from URL</li>
    <li>POST with file upload - Analyze colors in uploaded image</li>
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
                file_data.save(os.path.join(FOLDER, filename))
                
                file_size = os.path.getsize(os.path.join(FOLDER, filename))
                if file_size > MAX_FILE_SIZE:
                    cleanup_file(os.path.join(FOLDER, filename))
                    return jsonify({
                        "error": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                        "status": "error"
                    }), 400
                
                return jsonify(analyze_colors(filename))
                
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
    print(f"Starting Color Analysis API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"Configured color systems: {', '.join([s.title() for s in COLOR_SYSTEM])}")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)


