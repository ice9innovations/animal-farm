import json
import requests
import os
import uuid
import time
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

FOLDER = './uploads'
PRIVATE = os.getenv('PRIVATE', 'False').lower() in ['true', '1', 'yes']
PORT = os.getenv('PORT', '7781')
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
        
        # 🎯 NEW: Advanced Analysis (all the low-hanging fruit!)
        quality_analysis = analyze_image_quality(full_path)
        color_analysis = analyze_color_properties(full_path) 
        composition_analysis = analyze_composition(full_path)
        
        # Build basic metadata dict for alt-text generation
        basic_metadata = {"categorized": categorized}
        alt_text_analysis = generate_alt_text_suggestions(full_path, basic_metadata)
        
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
                # 🎯 NEW: All the low-hanging fruit analysis!
                "advanced_analysis": {
                    "image_quality": quality_analysis,
                    "color_properties": color_analysis,
                    "composition": composition_analysis,
                    "alt_text": alt_text_analysis
                },
                "analysis_summary": {
                    "aesthetic_score": quality_analysis.get("aesthetic_score", {}).get("score", 0) if not quality_analysis.get("error") else 0,
                    "is_blurry": quality_analysis.get("blur_analysis", {}).get("is_blurry", False) if not quality_analysis.get("error") else False,
                    "lighting_quality": quality_analysis.get("lighting_analysis", {}).get("lighting_quality", "unknown") if not quality_analysis.get("error") else "unknown",
                    "dominant_color": color_analysis.get("dominant_colors", [{}])[0].get("hex", "#000000") if not color_analysis.get("error") and color_analysis.get("dominant_colors") else "#000000",
                    "composition_complexity": composition_analysis.get("complexity_analysis", {}).get("complexity_level", "unknown") if not composition_analysis.get("error") else "unknown",
                    "estimated_content_type": alt_text_analysis.get("accessibility_notes", {}).get("estimated_content_type", "unknown") if not alt_text_analysis.get("error") else "unknown"
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
        blur_threshold = 100  # Empirically determined
        is_blurry = laplacian_var < blur_threshold
        
        # 2. LIGHTING ANALYSIS
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Determine lighting quality
        if mean_brightness < 50:
            lighting_quality = "too_dark"
        elif mean_brightness > 200:
            lighting_quality = "too_bright"  
        elif brightness_std < 30:
            lighting_quality = "low_contrast"
        else:
            lighting_quality = "good"
        
        # 3. EXPOSURE ANALYSIS
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for over/under exposure
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_pixels = np.sum(hist[:25]) / total_pixels  # Very dark pixels
        bright_pixels = np.sum(hist[230:]) / total_pixels  # Very bright pixels
        
        if dark_pixels > 0.4:
            exposure = "underexposed"
        elif bright_pixels > 0.4:
            exposure = "overexposed"
        else:
            exposure = "well_exposed"
        
        # 4. CONTRAST ANALYSIS
        contrast_score = brightness_std / 64.0  # Normalize to 0-4 range
        if contrast_score < 0.5:
            contrast = "low"
        elif contrast_score > 2.0:
            contrast = "high"
        else:
            contrast = "good"
        
        # 5. AESTHETIC SCORING (simple rule-based)
        aesthetic_score = 50  # Base score
        
        # Bonus for good technical quality
        if not is_blurry:
            aesthetic_score += 15
        if lighting_quality == "good":
            aesthetic_score += 15
        if exposure == "well_exposed":
            aesthetic_score += 10
        if contrast == "good":
            aesthetic_score += 10
        
        # Penalty for poor quality
        if is_blurry:
            aesthetic_score -= 20
        if lighting_quality in ["too_dark", "too_bright"]:
            aesthetic_score -= 15
        if exposure in ["underexposed", "overexposed"]:
            aesthetic_score -= 10
        
        aesthetic_score = max(0, min(100, aesthetic_score))  # Clamp 0-100
        
        return {
            "blur_analysis": {
                "laplacian_variance": round(laplacian_var, 2),
                "is_blurry": is_blurry,
                "blur_threshold": blur_threshold,
                "sharpness_score": min(100, laplacian_var / 5)  # Rough 0-100 scale
            },
            "lighting_analysis": {
                "mean_brightness": round(mean_brightness, 2),
                "brightness_std": round(brightness_std, 2),
                "lighting_quality": lighting_quality,
                "brightness_scale": "0=black, 255=white"
            },
            "exposure_analysis": {
                "exposure_quality": exposure,
                "dark_pixel_ratio": round(dark_pixels, 3),
                "bright_pixel_ratio": round(bright_pixels, 3),
                "histogram_balance": "good" if 0.1 < dark_pixels < 0.3 and bright_pixels < 0.1 else "poor"
            },
            "contrast_analysis": {
                "contrast_score": round(contrast_score, 2),
                "contrast_quality": contrast,
                "dynamic_range": "high" if brightness_std > 50 else "low"
            },
            "aesthetic_score": {
                "score": aesthetic_score,
                "scale": "0-100 (higher is better)",
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
        
        # 1. COLOR STATISTICS
        mean_colors = stat.mean  # R, G, B averages
        std_colors = stat.stddev  # R, G, B standard deviations
        
        # 2. DOMINANT COLORS (simple quantization)
        img_array = np.array(img_pil)
        pixels = img_array.reshape(-1, 3)
        
        # Sample pixels for speed (every 10th pixel)
        sampled_pixels = pixels[::10]
        
        # Find most common colors
        unique_colors, counts = np.unique(sampled_pixels, axis=0, return_counts=True)
        
        # Get top 5 colors
        top_indices = np.argsort(counts)[-5:][::-1]
        dominant_colors = []
        
        for i in top_indices:
            color = unique_colors[i]
            count = counts[i]
            percentage = (count / len(sampled_pixels)) * 100
            
            dominant_colors.append({
                "rgb": [int(color[0]), int(color[1]), int(color[2])],
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                "percentage": round(percentage, 1)
            })
        
        # 3. COLOR TEMPERATURE (simple estimation)
        avg_r, avg_g, avg_b = mean_colors
        
        if avg_b > avg_r:
            temp_description = "cool" 
        elif avg_r > avg_b * 1.2:
            temp_description = "warm"
        else:
            temp_description = "neutral"
        
        # 4. SATURATION ANALYSIS
        # Convert to HSV to analyze saturation
        img_hsv = img_pil.convert('HSV')
        hsv_stat = ImageStat.Stat(img_hsv)
        avg_saturation = hsv_stat.mean[1]  # S channel
        
        if avg_saturation < 50:
            saturation_level = "low"
        elif avg_saturation > 150:
            saturation_level = "high" 
        else:
            saturation_level = "moderate"
        
        return {
            "color_statistics": {
                "mean_rgb": [round(c, 1) for c in mean_colors],
                "std_rgb": [round(c, 1) for c in std_colors],
                "color_variance": round(sum(std_colors) / 3, 1)
            },
            "dominant_colors": dominant_colors,
            "color_temperature": {
                "description": temp_description,
                "red_blue_ratio": round(avg_r / max(avg_b, 1), 2)
            },
            "saturation_analysis": {
                "average_saturation": round(avg_saturation, 1),
                "saturation_level": saturation_level,
                "scale": "0=grayscale, 255=fully_saturated"
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
        elif aspect_ratio > 1.5:
            aspect_category = "landscape"
        elif aspect_ratio < 0.75:
            aspect_category = "portrait"
        else:
            aspect_category = "standard"
        
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
        
        if edge_density < 0.05:
            complexity = "simple"
        elif edge_density > 0.15:
            complexity = "complex"
        else:
            complexity = "moderate"
        
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
                "most_interesting_section": most_interesting_section,
                "section_layout": "0-8: top-left to bottom-right"
            },
            "complexity_analysis": {
                "edge_density": round(edge_density, 4),
                "complexity_level": complexity,
                "detail_description": f"{'Minimalist' if complexity == 'simple' else 'Busy' if complexity == 'complex' else 'Balanced'} composition"
            },
            "symmetry_analysis": {
                "horizontal_symmetry_score": round(symmetry_score, 3),
                "symmetry_level": "high" if symmetry_score > 0.8 else "low" if symmetry_score < 0.5 else "moderate"
            }
        }
        
    except Exception as e:
        return {"error": f"Composition analysis failed: {str(e)}"}

def generate_alt_text_suggestions(filepath: str, basic_metadata: Dict) -> Dict[str, Any]:
    """
    🎯 LOW-HANGING FRUIT: Alt-text generation based on image analysis
    """
    try:
        # Extract basic image properties
        img_pil = Image.open(filepath)
        width, height = img_pil.size
        format_name = img_pil.format or "Unknown"
        
        # Build descriptive alt-text
        alt_components = []
        
        # 1. Format and orientation
        if width > height * 1.3:
            orientation = "landscape"
        elif height > width * 1.3:
            orientation = "portrait" 
        else:
            orientation = "square"
        
        alt_components.append(f"{orientation} {format_name.lower()} image")
        
        # 2. Size description
        if width * height > 2000000:  # > 2MP
            size_desc = "high resolution"
        elif width * height < 100000:  # < 0.1MP
            size_desc = "low resolution"
        else:
            size_desc = "medium resolution"
        
        alt_components.append(size_desc)
        
        # 3. Check for common indicators in metadata
        camera_info = basic_metadata.get("categorized", {}).get("camera", {})
        datetime_info = basic_metadata.get("categorized", {}).get("datetime", {})
        
        if camera_info:
            alt_components.append("photographed with camera")
        
        # 4. Technical quality indicators
        quality_analysis = analyze_image_quality(filepath)
        if not quality_analysis.get("error"):
            blur_info = quality_analysis.get("blur_analysis", {})
            if blur_info.get("is_blurry"):
                alt_components.append("slightly blurred")
            
            lighting = quality_analysis.get("lighting_analysis", {}).get("lighting_quality")
            if lighting == "too_dark":
                alt_components.append("with low lighting")
            elif lighting == "too_bright":
                alt_components.append("with bright lighting")
        
        # Generate different alt-text options
        basic_alt = f"A {' '.join(alt_components)}"
        technical_alt = f"{format_name} image ({width}x{height} pixels)"
        descriptive_alt = f"{basic_alt} suitable for web display"
        
        return {
            "alt_text_suggestions": {
                "basic": basic_alt,
                "technical": technical_alt, 
                "descriptive": descriptive_alt,
                "components_detected": alt_components
            },
            "accessibility_notes": {
                "orientation": orientation,
                "size_category": size_desc,
                "has_camera_info": bool(camera_info),
                "estimated_content_type": "photograph" if camera_info else "digital_image"
            }
        }
        
    except Exception as e:
        return {"error": f"Alt-text generation failed: {str(e)}"}

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
            "GET /?url=<image_url> - Extract metadata from URL",
            "GET /?path=<local_path> - Extract metadata from local file (if not private)",
            "POST / - Upload and extract metadata from image"
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
                "service": "metadata",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing file_path parameter"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Validate file path
        if not os.path.exists(file_path):
            return jsonify({
                "service": "metadata",
                "status": "error",
                "predictions": [],
                "error": {"message": f"File not found: {file_path}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 404
        
        # For metadata service, we need to copy file to FOLDER since extract_comprehensive_metadata expects it there
        temp_filename = uuid.uuid4().hex + ".jpg"
        temp_filepath = os.path.join(FOLDER, temp_filename)
        
        try:
            # Copy the file to temp location
            import shutil
            shutil.copy2(file_path, temp_filepath)
            
            # Extract metadata using existing function (no cleanup - we handle temp file)
            result = extract_comprehensive_metadata(temp_filename, cleanup=False)
            
            # Clean up temporary file
            cleanup_file(temp_filepath)
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Metadata extraction failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 unified format with all advanced analysis
            metadata_data = result.get('metadata', {})
            file_info = metadata_data.get('file_info', {})
            advanced_analysis = metadata_data.get('advanced_analysis', {})
            analysis_summary = metadata_data.get('analysis_summary', {})
            categorized = metadata_data.get('categorized', {})
            
            # Extract image dimensions from categorized metadata
            image_meta = categorized.get('image', {})
            width = image_meta.get('File:ImageWidth') or advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('dimensions', [0, 0])[0]
            height = image_meta.get('File:ImageHeight') or advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('dimensions', [0, 0])[1]
            
            # Create unified prediction format
            predictions = []
            
            # Main metadata extraction prediction
            prediction = {
                "type": "metadata_extraction",
                "label": "image_metadata",
                "confidence": 1.0,
                "properties": {
                    # Basic file info
                    "file_size": file_info.get('file_size'),
                    "file_type": file_info.get('format', 'unknown'),
                    "dimensions": {
                        "width": width,
                        "height": height
                    },
                    "has_exif": metadata_data.get('summary', {}).get('has_exif_data', False),
                    "has_gps": metadata_data.get('summary', {}).get('has_gps_data', False),
                    "categories": list(metadata_data.get('categorized', {}).keys()),
                    
                    # Advanced analysis
                    "quality_analysis": {
                        "aesthetic_score": analysis_summary.get('aesthetic_score', 0),
                        "is_blurry": analysis_summary.get('is_blurry', False),
                        "lighting_quality": analysis_summary.get('lighting_quality', 'unknown'),
                        "sharpness_score": advanced_analysis.get('image_quality', {}).get('blur_analysis', {}).get('sharpness_score', 0)
                    },
                    "color_analysis": {
                        "dominant_color": analysis_summary.get('dominant_color', '#000000'),
                        "color_temperature": advanced_analysis.get('color_properties', {}).get('color_temperature', {}).get('description', 'unknown'),
                        "dominant_colors": advanced_analysis.get('color_properties', {}).get('dominant_colors', [])[:3]  # Top 3
                    },
                    "composition_analysis": {
                        "complexity": analysis_summary.get('composition_complexity', 'unknown'),
                        "aspect_ratio": advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('category', 'unknown'),
                        "symmetry_level": advanced_analysis.get('composition', {}).get('symmetry_analysis', {}).get('symmetry_level', 'unknown')
                    },
                    "accessibility": {
                        "alt_text_basic": advanced_analysis.get('alt_text', {}).get('alt_text_suggestions', {}).get('basic', ''),
                        "alt_text_technical": advanced_analysis.get('alt_text', {}).get('alt_text_suggestions', {}).get('technical', ''),
                        "content_type": analysis_summary.get('estimated_content_type', 'unknown')
                    },
                    
                    # Full analysis for power users
                    "full_analysis": advanced_analysis
                }
            }
            
            predictions.append(prediction)
            
            return jsonify({
                "service": "metadata",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "Advanced Metadata Extractor",
                        "framework": "ExifTool + PIL + OpenCV + NumPy",
                        "version": "2.0",
                        "features": ["metadata_extraction", "quality_analysis", "color_analysis", "composition_analysis", "accessibility"]
                    },
                    "image_dimensions": {
                        "width": width,
                        "height": height
                    }
                }
            })
            
        except Exception as copy_error:
            # Clean up temp file on error
            if os.path.exists(temp_filepath):
                cleanup_file(temp_filepath)
            raise copy_error
        
    except Exception as e:
        return jsonify({
            "service": "metadata",
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
                "service": "metadata",
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
            
            # Analyze using existing function
            result = extract_comprehensive_metadata(filename)
            
            if result.get('status') == 'error':
                return jsonify({
                    "service": "metadata",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": result.get('error', 'Metadata extraction failed')},
                    "metadata": {"processing_time": round(time.time() - start_time, 3)}
                }), 500
            
            # Convert to v2 unified format with all our new analysis!
            metadata_data = result.get('metadata', {})
            file_info = metadata_data.get('file_info', {})
            advanced_analysis = metadata_data.get('advanced_analysis', {})
            analysis_summary = metadata_data.get('analysis_summary', {})
            categorized = metadata_data.get('categorized', {})
            
            # Extract image dimensions from categorized metadata
            image_meta = categorized.get('image', {})
            width = image_meta.get('File:ImageWidth') or advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('dimensions', [0, 0])[0]
            height = image_meta.get('File:ImageHeight') or advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('dimensions', [0, 0])[1]
            
            # Create unified prediction format with ALL our new analysis!
            predictions = []
            
            # Main metadata extraction prediction
            prediction = {
                "type": "metadata_extraction",
                "label": "image_metadata",
                "confidence": 1.0,
                "properties": {
                    # Basic file info
                    "file_size": file_info.get('file_size'),
                    "file_type": file_info.get('format', 'unknown'),
                    "dimensions": {
                        "width": width,
                        "height": height
                    },
                    "has_exif": metadata_data.get('summary', {}).get('has_exif_data', False),
                    "has_gps": metadata_data.get('summary', {}).get('has_gps_data', False),
                    "categories": list(metadata_data.get('categorized', {}).keys()),
                    
                    # 🎯 NEW: All the advanced analysis!
                    "quality_analysis": {
                        "aesthetic_score": analysis_summary.get('aesthetic_score', 0),
                        "is_blurry": analysis_summary.get('is_blurry', False),
                        "lighting_quality": analysis_summary.get('lighting_quality', 'unknown'),
                        "sharpness_score": advanced_analysis.get('image_quality', {}).get('blur_analysis', {}).get('sharpness_score', 0)
                    },
                    "color_analysis": {
                        "dominant_color": analysis_summary.get('dominant_color', '#000000'),
                        "color_temperature": advanced_analysis.get('color_properties', {}).get('color_temperature', {}).get('description', 'unknown'),
                        "dominant_colors": advanced_analysis.get('color_properties', {}).get('dominant_colors', [])[:3]  # Top 3
                    },
                    "composition_analysis": {
                        "complexity": analysis_summary.get('composition_complexity', 'unknown'),
                        "aspect_ratio": advanced_analysis.get('composition', {}).get('aspect_ratio', {}).get('category', 'unknown'),
                        "symmetry_level": advanced_analysis.get('composition', {}).get('symmetry_analysis', {}).get('symmetry_level', 'unknown')
                    },
                    "accessibility": {
                        "alt_text_basic": advanced_analysis.get('alt_text', {}).get('alt_text_suggestions', {}).get('basic', ''),
                        "alt_text_technical": advanced_analysis.get('alt_text', {}).get('alt_text_suggestions', {}).get('technical', ''),
                        "content_type": analysis_summary.get('estimated_content_type', 'unknown')
                    },
                    
                    # Full analysis for power users
                    "full_analysis": advanced_analysis
                }
            }
            
            predictions.append(prediction)
            
            return jsonify({
                "service": "metadata",
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "name": "Advanced Metadata Extractor",
                        "framework": "ExifTool + PIL + OpenCV + NumPy",
                        "version": "2.0",
                        "features": ["metadata_extraction", "quality_analysis", "color_analysis", "composition_analysis", "accessibility"]
                    },
                    "image_dimensions": {
                        "width": width,
                        "height": height
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "service": "metadata",
                "status": "error", 
                "predictions": [],
                "error": {"message": f"Failed to process image: {str(e)}"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
    except Exception as e:
        return jsonify({
            "service": "metadata",
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
                
                result = jsonify(extract_comprehensive_metadata(filename))
                # extract_comprehensive_metadata already does cleanup
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
                
            return jsonify(extract_comprehensive_metadata(path))
            
        else:
            try:
                with open('form.html', 'r') as file:
                    return file.read()
            except FileNotFoundError:
                return '''<!DOCTYPE html>
<html>
<head><title>Metadata Extraction API</title></head>
<body>
<h2>Image Metadata Extraction Service</h2>
<form enctype="multipart/form-data" method="POST">
    <input type="file" name="uploadedfile" accept="image/*" required><br><br>
    <input type="submit" value="Extract Metadata">
</form>
<p><strong>API Usage:</strong></p>
<ul>
    <li>GET /?url=&lt;image_url&gt; - Extract metadata from URL</li>
    <li>POST with file upload - Extract metadata from uploaded image</li>
    <li>GET /health - Service health check</li>
</ul>
<p><strong>Extracted Information:</strong></p>
<ul>
    <li>Camera settings (ISO, aperture, shutter speed, etc.)</li>
    <li>GPS location data</li>
    <li>Image properties (dimensions, color profile, etc.)</li>
    <li>Creation and modification dates</li>
    <li>Software and processing information</li>
    <li>Technical specifications</li>
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
                
                return jsonify(extract_comprehensive_metadata(filename))
                
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