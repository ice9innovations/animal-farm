import json
import requests
import os
import uuid
import time
import io
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib
import base64
import numpy as np
import cv2
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image, ExifTags, ImageStat, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction
import exiftool

load_dotenv()

# Step 1: Load as strings (no fallbacks)
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')

# Step 2: Validate critical environment variables
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Step 3: Convert to appropriate types after validation
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
PORT = int(PORT_STR)

FOLDER = './uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

def cleanup_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Could not remove file {filepath}: {e}")

def serialize_metadata_value(value):
    """Convert EXIF/metadata values to JSON-serializable format"""
    # Handle None values
    if value is None:
        return None
    
    # Handle numpy types (NEW - fix for JSON serialization)
    if hasattr(value, 'dtype'):  # numpy array/scalar
        if hasattr(value, 'item'):  # scalar
            return value.item()
        else:  # array
            return value.tolist()
    
    # Handle specific numpy types
    if isinstance(value, (np.int64, np.int32, np.float64, np.float32, np.bool_)):
        return value.item()
    
    # Handle bytes
    if isinstance(value, bytes):
        # Check if it's likely text data (small and printable)
        if len(value) < 1000 and all(32 <= b <= 126 or b in (9, 10, 13) for b in value):
            try:
                return value.decode('utf-8', errors='replace')
            except:
                return f"<Binary data {len(value)} bytes>"
        else:
            # For large binary data or non-printable data, just describe it
            return f"<Binary data {len(value)} bytes>"
    
    # Handle PIL IFDRational (fraction-like objects)
    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        try:
            if value.denominator == 0:
                return float('inf') if value.numerator > 0 else float('-inf')
            elif value.denominator == 1:
                return int(value.numerator)
            else:
                return float(value.numerator) / float(value.denominator)
        except:
            return str(value)
    
    # Handle regular fractions
    if isinstance(value, Fraction):
        try:
            if value.denominator == 1:
                return int(value.numerator)
            else:
                return float(value)
        except:
            return str(value)
    
    # Handle tuples and lists
    if isinstance(value, (tuple, list)):
        return [serialize_metadata_value(item) for item in value]
    
    # Handle dictionaries
    if isinstance(value, dict):
        return {k: serialize_metadata_value(v) for k, v in value.items()}
    
    # Handle datetime objects
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    
    # Handle other objects that might not be serializable
    if hasattr(value, '__dict__'):
        try:
            return str(value)
        except:
            return f"<{type(value).__name__} object>"
    
    # For primitive types (int, float, str, bool)
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return str(value)

def get_file_hash(filepath: str) -> str:
    """Generate SHA3-256 hash of file"""
    try:
        with open(filepath, 'rb') as f:
            file_data = f.read()
            img_hash = hashlib.sha256(file_data)
            return img_hash.hexdigest()
    except Exception:
        return ""

def extract_basic_file_info(filepath: str) -> Dict[str, Any]:
    """Extract basic file system information"""
    try:
        stat_info = os.stat(filepath)
        return {
            "filename": os.path.basename(filepath),
            "file_size": stat_info.st_size,
            "file_size_human": format_file_size(stat_info.st_size),
            "created_time": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "accessed_time": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
        }
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def extract_pil_metadata(filepath: str) -> Dict[str, Any]:
    """Extract metadata using PIL/Pillow"""
    try:
        with Image.open(filepath) as img:
            # Basic image info
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.size[0],
                "height": img.size[1],
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
            }
            
            # Add format-specific info
            if hasattr(img, 'info') and img.info:
                pil_info = {}
                for key, value in img.info.items():
                    pil_info[key] = serialize_metadata_value(value)
                info["pil_info"] = pil_info
            
            # Extract EXIF data
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        # Handle special cases
                        if tag == 'GPSInfo':
                            gps_data = {}
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gps_data[gps_tag] = serialize_metadata_value(gps_value)
                            exif_data[tag] = gps_data
                        else:
                            exif_data[tag] = serialize_metadata_value(value)
            
            info["exif"] = exif_data
            return info
            
    except Exception as e:
        return {"error": f"PIL extraction failed: {str(e)}"}

def extract_exiftool_metadata(filepath: str) -> Dict[str, Any]:
    """Extract comprehensive metadata using ExifTool"""
    try:
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata(filepath)
            
            # Remove the SourceFile entry
            if 'SourceFile' in metadata:
                del metadata['SourceFile']
            
            # Define fields to exclude (large binary data that's rarely useful)
            exclude_fields = {
                'ICC_Profile:BlueTRC',
                'ICC_Profile:GreenTRC', 
                'ICC_Profile:RedTRC',
                'PIL:icc_profile',
                'PIL:exif',  # Raw EXIF binary - we get this parsed elsewhere
            }
            
            # Process metadata to handle different data types
            processed = {}
            for key, value in metadata.items():
                # Skip large binary fields that clutter output
                if key in exclude_fields:
                    processed[key] = "<Excluded: Large binary data>"
                    continue
                
                processed[key] = serialize_metadata_value(value)
            
            return processed
            
    except Exception as e:
        return {"error": f"ExifTool extraction failed: {str(e)}"}

def categorize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize metadata into logical groups"""
    categories = {
        "camera": {},
        "image": {},
        "gps": {},
        "datetime": {},
        "software": {},
        "technical": {},
        "other": {}
    }
    
    # Define category mappings
    camera_keywords = ['camera', 'lens', 'focal', 'aperture', 'shutter', 'iso', 'flash', 'exposure', 'make', 'model']
    image_keywords = ['width', 'height', 'resolution', 'color', 'orientation', 'compression']
    gps_keywords = ['gps', 'latitude', 'longitude', 'altitude', 'location']
    datetime_keywords = ['date', 'time', 'created', 'modified']
    software_keywords = ['software', 'application', 'tool', 'version', 'creator']
    technical_keywords = ['encoding', 'profile', 'depth', 'sample', 'format']
    
    for key, value in metadata.items():
        key_lower = key.lower()
        categorized = False
        
        # Check each category
        if any(keyword in key_lower for keyword in camera_keywords):
            categories["camera"][key] = value
            categorized = True
        elif any(keyword in key_lower for keyword in image_keywords):
            categories["image"][key] = value
            categorized = True
        elif any(keyword in key_lower for keyword in gps_keywords):
            categories["gps"][key] = value
            categorized = True
        elif any(keyword in key_lower for keyword in datetime_keywords):
            categories["datetime"][key] = value
            categorized = True
        elif any(keyword in key_lower for keyword in software_keywords):
            categories["software"][key] = value
            categorized = True
        elif any(keyword in key_lower for keyword in technical_keywords):
            categories["technical"][key] = value
            categorized = True
        
        if not categorized:
            categories["other"][key] = value
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def extract_comprehensive_metadata(image_file: str, cleanup: bool = True) -> Dict[str, Any]:
    """Extract maximum possible metadata from image using multiple methods"""
    start_time = time.time()
    full_path = os.path.join(FOLDER, image_file)
    
    try:
        # Basic file information
        file_info = extract_basic_file_info(full_path)
        file_hash = get_file_hash(full_path)
        
        # PIL metadata extraction
        pil_metadata = extract_pil_metadata(full_path)
        
        # ExifTool metadata extraction (most comprehensive)
        exiftool_metadata = extract_exiftool_metadata(full_path)
        
        # Merge all metadata sources
        all_metadata = {}
        
        # Add ExifTool metadata (most comprehensive)
        if "error" not in exiftool_metadata:
            all_metadata.update(exiftool_metadata)
        
        # Add PIL metadata that might not be in ExifTool
        if "error" not in pil_metadata:
            pil_info = pil_metadata.get("pil_info", {})
            if pil_info:
                for key, value in pil_info.items():
                    if key not in all_metadata:
                        all_metadata[f"PIL:{key}"] = value
        
        # Categorize metadata
        categorized = categorize_metadata(all_metadata) if all_metadata else {}
        
        # 🎯 STREAMLINED: Advanced Analysis (remove ALT text generation)
        quality_analysis = analyze_image_quality(full_path)
        color_analysis = analyze_color_properties(full_path) 
        composition_analysis = analyze_composition(full_path)
        
        # Calculate summary statistics
        total_tags = len(all_metadata)
        categories_found = len(categorized)
        has_exif = bool([k for k in all_metadata.keys() if 'exif' in k.lower()])
        has_gps = bool(categorized.get("gps", {}))
        has_camera_info = bool(categorized.get("camera", {}))
        
        processing_time = round(time.time() - start_time, 3)
        
        # Ensure file_info is also serialized
        serialized_file_info = {k: serialize_metadata_value(v) for k, v in file_info.items()}
        
        result = {
            "metadata": {
                "file_info": serialized_file_info,
                "file_hash": file_hash,
                "summary": {
                    "total_metadata_tags": total_tags,
                    "categories_found": categories_found,
                    "has_exif_data": has_exif,
                    "has_gps_data": has_gps,
                    "has_camera_info": has_camera_info,
                    "extraction_methods": ["ExifTool", "PIL/Pillow"],
                    "processing_time": processing_time
                },
                "categorized": categorized,
                "raw_metadata": all_metadata,
                "extraction_info": {
                    "pil_result": "success" if "error" not in pil_metadata else pil_metadata.get("error"),
                    "exiftool_result": "success" if "error" not in exiftool_metadata else exiftool_metadata.get("error"),
                    "total_extraction_time": processing_time
                },
                # 🎯 STREAMLINED: Essential analysis only (ALT text removed)
                "advanced_analysis": {
                    "image_quality": quality_analysis,
                    "color_properties": color_analysis,
                    "composition": composition_analysis
                },
                "analysis_summary": {
                    "aesthetic_score": quality_analysis.get("aesthetic_score", {}).get("score", 0) if not quality_analysis.get("error") else 0,
                    "lighting_quality": quality_analysis.get("lighting_analysis", {}).get("lighting_quality", 0) if not quality_analysis.get("error") else 0,
                    "exposure_quality": quality_analysis.get("exposure_analysis", {}).get("exposure_quality", 0) if not quality_analysis.get("error") else 0,
                    "contrast_quality": quality_analysis.get("contrast_analysis", {}).get("contrast_quality", 0) if not quality_analysis.get("error") else 0,
                    "histogram_balance": quality_analysis.get("exposure_analysis", {}).get("histogram_balance", 0) if not quality_analysis.get("error") else 0,
                    "dynamic_range": quality_analysis.get("contrast_analysis", {}).get("dynamic_range", 0) if not quality_analysis.get("error") else 0,
                    "saturation_level": color_analysis.get("saturation_analysis", {}).get("saturation_level", 0) if not color_analysis.get("error") else 0,
                    "complexity_level": composition_analysis.get("complexity_analysis", {}).get("complexity_level", 0) if not composition_analysis.get("error") else 0,
                    "symmetry_level": composition_analysis.get("symmetry_analysis", {}).get("symmetry_level", 0) if not composition_analysis.get("error") else 0
                },
                "status": "success"
            }
        }
        
        # Cleanup (only for temporary files)
        if cleanup:
            cleanup_file(full_path)
        # Ensure everything is JSON serializable
        return json.loads(json.dumps(result, default=serialize_metadata_value))
        
    except Exception as e:
        # Cleanup on error (only for temporary files)
        if cleanup:
            cleanup_file(full_path)
        return {
            "error": f"Metadata extraction failed: {str(e)}",
            "status": "error"
        }

def analyze_image_quality(filepath: str) -> Dict[str, Any]:
    """
    🎯 LOW-HANGING FRUIT: Advanced image quality analysis
    Blur detection, lighting analysis, aesthetic scoring
    """
    try:
        # Read image with OpenCV for analysis
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            return {"error": "Could not read image for quality analysis"}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. BLUR DETECTION using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. LIGHTING ANALYSIS
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 3. EXPOSURE ANALYSIS
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for over/under exposure
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_pixels = np.sum(hist[:25]) / total_pixels  # Very dark pixels
        bright_pixels = np.sum(hist[230:]) / total_pixels  # Very bright pixels
        
        # 4. CONTRAST ANALYSIS
        contrast_score = brightness_std / 64.0  # Normalize to 0-4 range
        
        # 5. AESTHETIC SCORING (rule-based on raw values)
        aesthetic_score = 50  # Base score
        
        # Bonus for good technical quality (using raw thresholds)
        if laplacian_var >= 100:  # Good sharpness
            aesthetic_score += 15
        if 50 <= mean_brightness <= 200 and brightness_std >= 30:  # Good lighting
            aesthetic_score += 15
        if dark_pixels <= 0.4 and bright_pixels <= 0.4:  # Well exposed
            aesthetic_score += 10
        if 0.5 <= contrast_score <= 2.0:  # Good contrast
            aesthetic_score += 10
        
        # Penalty for poor quality
        if laplacian_var < 100:  # Low sharpness
            aesthetic_score -= 20
        if mean_brightness < 50 or mean_brightness > 200:  # Poor lighting
            aesthetic_score -= 15
        if dark_pixels > 0.4 or bright_pixels > 0.4:  # Poor exposure
            aesthetic_score -= 10
        
        aesthetic_score = max(0, min(100, aesthetic_score))  # Clamp 0-100
        
        return {
            "blur_analysis": {
                "laplacian_variance": round(laplacian_var, 2),
                "sharpness_score": round(min(1.0, laplacian_var / 500), 3)  # 0.0-1.0 scale
            },
            "lighting_analysis": {
                "mean_brightness": round(mean_brightness, 2),
                "lighting_quality": round(mean_brightness, 2)  # Raw brightness value instead of categorical
            },
            "exposure_analysis": {
                "dark_pixel_ratio": round(dark_pixels, 3),
                "bright_pixel_ratio": round(bright_pixels, 3),
                "exposure_quality": round((1.0 - dark_pixels - bright_pixels) * 100, 1),  # 0-100 exposure quality score
                "histogram_balance": round(brightness_std, 2)  # Raw std dev instead of good/poor
            },
            "contrast_analysis": {
                "contrast_score": round(contrast_score, 2),
                "brightness_std": round(brightness_std, 2),
                "contrast_quality": round(contrast_score * 25, 1),  # Scale 0-4 range to 0-100
                "dynamic_range": round(brightness_std, 2)  # Raw std dev instead of high/low
            },
            "aesthetic_score": {
                "score": aesthetic_score,
                "factors": "Based on sharpness, lighting, exposure, contrast"
            }
        }
        
    except Exception as e:
        return {"error": f"Quality analysis failed: {str(e)}"}

def analyze_color_properties(filepath: str) -> Dict[str, Any]:
    """
    🎯 LOW-HANGING FRUIT: Color analysis and dominant colors
    """
    try:
        # Read image with PIL for color analysis
        img_pil = Image.open(filepath)
        
        # Convert to RGB if necessary
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Get image statistics
        stat = ImageStat.Stat(img_pil)
        
        # SATURATION ANALYSIS
        # Convert to HSV to analyze saturation
        img_hsv = img_pil.convert('HSV')
        hsv_stat = ImageStat.Stat(img_hsv)
        avg_saturation = hsv_stat.mean[1]  # S channel
        
        return {
            "saturation_analysis": {
                "average_saturation": round(avg_saturation, 1),
                "saturation_level": round(avg_saturation, 1)  # Raw saturation value instead of low/moderate/high
            }
        }
        
    except Exception as e:
        return {"error": f"Color analysis failed: {str(e)}"}

def analyze_composition(filepath: str) -> Dict[str, Any]:
    """
    🎯 LOW-HANGING FRUIT: Composition and scene analysis
    """
    try:
        # Read with both PIL and OpenCV
        img_pil = Image.open(filepath)
        img_cv = cv2.imread(filepath)
        
        if img_cv is None:
            return {"error": "Could not read image for composition analysis"}
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 1. ASPECT RATIO ANALYSIS
        aspect_ratio = width / height
        
        if abs(aspect_ratio - 1.0) < 0.1:
            aspect_category = "square"
        elif aspect_ratio > 1.2:
            aspect_category = "landscape"
        elif aspect_ratio < 0.8:
            aspect_category = "portrait"
        else:
            aspect_category = "square"
        
        # 2. RULE OF THIRDS ANALYSIS
        # Divide image into 9 sections and analyze content distribution
        third_h = height // 3
        third_w = width // 3
        
        sections = []
        for i in range(3):
            for j in range(3):
                section = gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                sections.append(np.mean(section))
        
        # Find the section with highest contrast (likely subject)
        section_variance = [np.std(gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]) 
                           for i in range(3) for j in range(3)]
        most_interesting_section = np.argmax(section_variance)
        
        # 3. EDGE DETECTION for complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # 4. SYMMETRY ANALYSIS (simple)
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        # Resize to match if odd width
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_score = 1.0 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0)
        
        return {
            "aspect_ratio": {
                "ratio": round(aspect_ratio, 2),
                "category": aspect_category,
                "dimensions": [width, height]
            },
            "rule_of_thirds": {
                "section_brightness": [round(s, 1) for s in sections],
                "most_interesting_section": most_interesting_section
            },
            "complexity_analysis": {
                "edge_density": round(edge_density, 3),
                "complexity_level": round(edge_density * 100, 1)  # Scale edge density to 0-100 score
            },
            "symmetry_analysis": {
                "horizontal_symmetry_score": round(symmetry_score, 3),
                "symmetry_level": round(symmetry_score * 100, 1)  # Scale to 0-100 score
            }
        }
        
    except Exception as e:
        return {"error": f"Composition analysis failed: {str(e)}"}

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        # Return PIL Image directly from bytes
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode not in ['RGB', 'RGBA', 'L']:
            image = image.convert('RGB')
            
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        image = Image.open(file_path)
        
        # Convert to RGB if necessary (but preserve original mode for metadata)
        if image.mode not in ['RGB', 'RGBA', 'L']:
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def process_image_for_metadata(image: Image.Image) -> dict:
    """Main processing function - takes PIL Image, returns metadata data
    This is the core business logic, separated from HTTP concerns"""
    try:
        # Save PIL Image to temporary file for metadata extraction
        import tempfile
        temp_filename = None
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=FOLDER) as tmp_file:
            temp_filename = os.path.basename(tmp_file.name)
            image.save(tmp_file.name, format='JPEG', quality=95)
        
        # Extract metadata using existing function
        result = extract_comprehensive_metadata(temp_filename, cleanup=True)
        
        if result.get('status') == 'error':
            return {
                'success': False,
                'data': {},
                'error': result.get('error', 'Metadata extraction failed')
            }
        
        return {
            'success': True,
            'data': result,
            'error': None
        }
        
    except Exception as e:
        # Clean up temp file on error if it exists
        if temp_filename:
            temp_path = os.path.join(FOLDER, temp_filename)
            cleanup_file(temp_path)
        
        return {
            'success': False,
            'data': {},
            'error': str(e)
        }

def create_metadata_response(metadata_result: dict, processing_time: float) -> dict:
    """Create standardized metadata response with formatting"""
    try:
        # Use existing formatting function to convert raw analysis to clean API response
        prediction = format_metadata_response(metadata_result)
        
        return {
            "service": "metadata",
            "status": "success",
            "predictions": [prediction],
            "metadata": {
                "processing_time": round(processing_time, 3),
                "model_info": {
                    "framework": "ExifTool + PIL + OpenCV + NumPy"
                }
            }
        }
        
    except Exception as e:
        return {
            "service": "metadata",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Failed to format response: {str(e)}"},
            "metadata": {
                "processing_time": round(processing_time, 3),
                "model_info": {
                    "framework": "ExifTool + PIL + OpenCV + NumPy"
                }
            }
        }

def format_metadata_response(metadata_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metadata extraction results into clean API response.
    Uses pre-calculated values without duplication.
    """
    try:
        metadata_data = metadata_result.get('metadata', {})
        file_info = metadata_data.get('file_info', {})
        advanced_analysis = metadata_data.get('advanced_analysis', {})
        analysis_summary = metadata_data.get('analysis_summary', {})
        categorized = metadata_data.get('categorized', {})
        
        # Extract pre-calculated analysis values
        image_quality = advanced_analysis.get('image_quality', {})
        composition = advanced_analysis.get('composition', {})
        blur_analysis = image_quality.get('blur_analysis', {})
        contrast_analysis = image_quality.get('contrast_analysis', {})
        exposure_analysis = image_quality.get('exposure_analysis', {})
        lighting_analysis = image_quality.get('lighting_analysis', {})
        
        # Extract composition analysis
        complexity_analysis = composition.get('complexity_analysis', {})
        rule_of_thirds = composition.get('rule_of_thirds', {})
        symmetry_analysis = composition.get('symmetry_analysis', {})
        aspect_ratio = composition.get('aspect_ratio', {})
        
        # Extract image dimensions and format
        image_meta = categorized.get('image', {})
        width = image_meta.get('File:ImageWidth') or aspect_ratio.get('dimensions', [0, 0])[0]
        height = image_meta.get('File:ImageHeight') or aspect_ratio.get('dimensions', [0, 0])[1]
        
        # Get file format from ExifTool metadata
        file_format_raw = (
            metadata_data.get('raw_metadata', {}).get('File:FileType') or
            image_meta.get('File:FileType') or
            'unknown'
        )
        file_format = file_format_raw.lower() if file_format_raw != 'unknown' else 'unknown'
        
        # Extract GPS data if available
        gps_data = categorized.get('gps', {})
        has_gps = bool(gps_data)
        
        # Extract core EXIF data if available
        camera_data = categorized.get('camera', {})
        datetime_data = categorized.get('datetime', {})
        exif_data = {}
        
        # Combine useful EXIF fields from different categories
        if camera_data:
            exif_data.update({k: v for k, v in camera_data.items() if v is not None})
        if datetime_data:
            exif_data.update({k: v for k, v in datetime_data.items() if v is not None})
        
        has_exif = bool(exif_data)
        
        # Build clean prediction response using pre-extracted values
        prediction = {
            "dimensions": {
                "height": height,
                "width": width
            },
            "aspect_ratio": {
                "category": aspect_ratio.get('category', 'unknown'),
                "ratio": round(aspect_ratio.get('ratio', 0), 1)
            },
            "file": {
                "file_size": file_info.get('file_size'),
                "file_type": file_format
            },
            "color_properties": {
                "saturation_analysis": {
                    "average_saturation": round(analysis_summary.get('saturation_level', 0), 1),
                    "saturation_level": round(analysis_summary.get('saturation_level', 0), 1)
                }
            },
            "composition": {
                "complexity_analysis": {
                    "complexity_level": round(analysis_summary.get('complexity_level', 0), 1),
                    "edge_density": round(complexity_analysis.get('edge_density', 0), 3)
                },
                "rule_of_thirds": {
                    "most_interesting_section": rule_of_thirds.get('most_interesting_section', 0),
                    "section_brightness": [round(x, 1) for x in rule_of_thirds.get('section_brightness', [])]
                },
                "symmetry_analysis": {
                    "horizontal_symmetry_score": round(symmetry_analysis.get('horizontal_symmetry_score', 0), 3),
                    "symmetry_level": round(analysis_summary.get('symmetry_level', 0), 1)
                }
            },
            "image_quality": {
                "blur_analysis": {
                    "laplacian_variance": round(blur_analysis.get('laplacian_variance', 0), 2),
                    "sharpness_score": round(blur_analysis.get('sharpness_score', 0), 3)
                },
                "contrast_analysis": {
                    "brightness_variation": round(contrast_analysis.get('brightness_std', 0), 2),
                    "contrast_quality": round(analysis_summary.get('contrast_quality', 0), 1),
                    "contrast_score": round(contrast_analysis.get('contrast_score', 0), 2),
                    "dynamic_range": round(analysis_summary.get('dynamic_range', 0), 2)
                },
                "exposure_analysis": {
                    "bright_pixel_ratio": round(exposure_analysis.get('bright_pixel_ratio', 0), 3),
                    "dark_pixel_ratio": round(exposure_analysis.get('dark_pixel_ratio', 0), 3),
                    "exposure_quality": round(analysis_summary.get('exposure_quality', 0), 3),
                    "histogram_balance": round(analysis_summary.get('histogram_balance', 0), 3)
                },
                "lighting_analysis": {
                    "lighting_quality": round(analysis_summary.get('lighting_quality', 0), 3),
                    "mean_brightness": round(lighting_analysis.get('mean_brightness', 0), 3)
                }
            }
        }
        
        # Include GPS data if available
        if has_gps and gps_data:
            prediction["gps_data"] = {k: v for k, v in gps_data.items() if v is not None}
        
        # Include EXIF data if available
        if has_exif and exif_data:
            prediction["exif_data"] = exif_data
        
        return prediction
        
    except Exception as e:
        raise ValueError(f"Failed to format metadata response: {str(e)}")

app = Flask(__name__)

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Metadata service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test ExifTool availability
    exiftool_available = True
    exiftool_version = None
    try:
        with exiftool.ExifTool() as et:
            # Test with a simple command
            exiftool_version = "Available"
    except Exception as e:
        exiftool_available = False
        exiftool_version = f"Error: {str(e)}"
    
    return jsonify({
        "status": "healthy" if exiftool_available else "degraded",
        "service": "Metadata Extraction",
        "extraction_engines": {
            "exiftool": {
                "available": exiftool_available,
                "status": exiftool_version
            },
            "pil_pillow": {
                "available": True,
                "status": "Available"
            }
        },
        "features": {
            "metadata_categories": ["camera", "image", "gps", "datetime", "software", "technical"],
            "extraction_methods": ["comprehensive_exif", "image_properties", "file_system_info"],
            "supported_formats": ["JPEG", "PNG", "TIFF", "RAW", "GIF", "BMP", "WEBP"],
            "advanced_analysis": {
                "image_quality": ["blur_detection", "lighting_analysis", "exposure_analysis", "contrast_analysis", "aesthetic_scoring"],
                "color_properties": ["dominant_colors", "color_statistics", "color_temperature", "saturation_analysis"],
                "composition": ["aspect_ratio", "rule_of_thirds", "complexity_analysis", "symmetry_analysis"],
                "accessibility": ["alt_text_generation", "content_type_detection", "orientation_analysis"]
            }
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /analyze?url=<url> - Extract metadata from URL",
            "GET /analyze?file=<path> - Extract metadata from local file", 
            "POST /analyze - Extract metadata from uploaded file",
            "GET /v3/analyze?url=<url> - V3 compatibility (redirects to /analyze)",
            "GET /v2/analyze?image_url=<url> - V2 compatibility (redirects to /analyze)"
        ]
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    import time
    from io import BytesIO
    start_time = time.time()
    
    try:
        # Step 1: Get image data (URL/file/POST)
        if request.method == 'POST':
            # Handle POST file upload
            if 'file' not in request.files:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file provided in POST request"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file selected"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
            
            # Validate file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
            
            # Load image from POST upload
            try:
                image = Image.open(file)
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    image = image.convert('RGB')
            except Exception as e:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to load uploaded image: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
        
        else:
            # Handle GET requests
            url = request.args.get('url')
            file_path = request.args.get('file')
            
            # Validate input - exactly one parameter required
            if not url and not file_path:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Must provide either 'url' or 'file' parameter"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
            
            if url and file_path:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
            
            # Load image from URL or file
            try:
                if url:
                    image = download_image_from_url(url)
                elif file_path:
                    image = validate_image_file(file_path)
            except Exception as e:
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": str(e)},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                    }
                }), 400
        
        # Step 2: Call processing function
        result = process_image_for_metadata(image)
        processing_time = time.time() - start_time
        
        # Step 3: Handle processing result
        if not result['success']:
            return jsonify({
                "service": "metadata",
                "status": "error",
                "predictions": [],
                "error": {"message": result['error']},
                "metadata": {
                    "processing_time": round(processing_time, 3),
                    "model_info": {"framework": "ExifTool + PIL + OpenCV + NumPy"}
                }
            }), 500
        
        # Step 4: Create response
        return jsonify(create_metadata_response(result['data'], processing_time))
        
    except Exception as e:
        return jsonify({
            'service': 'metadata',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {'framework': 'ExifTool + PIL + OpenCV + NumPy'}
            },
            'error': {'message': str(e)}
        }), 500

# V3 compatibility route
@app.route('/v3/analyze', methods=['GET'])
def analyze_v3_compat():
    """V3 compatibility - redirect to new analyze endpoint"""
    with app.test_request_context('/analyze', query_string=request.args):
        return analyze()

# V2 compatibility routes
@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
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
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation
        new_args = {'url': image_url}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()


if __name__ == '__main__':
    # Test ExifTool availability
    try:
        with exiftool.ExifTool() as et:
            print("ExifTool: Available and working")
    except Exception as e:
        print(f"WARNING: ExifTool not available: {e}")
        print("Install with: sudo apt-get install exiftool")
    
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    print(f"Starting Metadata Extraction API on {host}:{PORT}")
    print(f"Private mode: {PRIVATE}")
    print(f"Extraction engines: ExifTool + PIL/Pillow")
    print(f"Max file size: {MAX_FILE_SIZE // 1024 // 1024}MB")
    app.run(host=host, port=int(PORT), debug=False, threaded=True)
