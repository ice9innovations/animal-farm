#!/usr/bin/env python3
"""
Colors Analyzer - Core color analysis functionality

Handles color extraction, palette analysis, and color naming using
Haishoku color extraction library with Copic and Prismacolor mappings.
"""

import re
import time
import json
import logging
import tempfile
import os
from typing import Dict, Any, List
from PIL import Image
from haishoku.haishoku import Haishoku

logger = logging.getLogger(__name__)


class ColorsAnalyzer:
    """Core color analysis functionality"""
    
    def __init__(self, color_systems: List[str] = None):
        """Initialize colors analyzer with configuration"""
        
        self.color_systems = color_systems or ['copic', 'prismacolor']
        
        # Load color data from JSON
        self.color_data = self._load_color_data()
        
        logger.info(f"✅ ColorsAnalyzer initialized with systems: {self.color_systems}")
        logger.info(f"✅ Loaded color data: {list(self.color_data.keys())}")
    
    def _load_color_data(self) -> Dict[str, Dict]:
        """Load color data from JSON file"""
        try:
            import os
            current_dir = os.path.dirname(__file__)
            json_path = os.path.join(current_dir, 'color_names.json')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                color_data = json.load(f)
            
            logger.info(f"Loaded color data from {json_path}")
            return color_data
            
        except Exception as e:
            logger.error(f"Failed to load color data: {e}")
            return {}
    
    def get_color_by_name(self, name, style):
        """Get RGB values for a color name - updated to use JSON data"""
        style = str(style).lower()

        if style not in self.color_data:
            return None

        color_dict = self.color_data[style]
        return color_dict.get(name)
    
    def get_color_name(self, rgb, style):
        """Get closest color name for RGB values - updated to use JSON data"""
        style = str(style).lower()

        if style not in self.color_data:
            return None

        color_dict = self.color_data[style]
        
        min_distance = float("inf")
        closest_color = None
        for color, value in color_dict.items():
            # Convert list back to tuple for distance calculation
            rgb_values = tuple(value) if isinstance(value, list) else value
            distance = sum([(i - j) ** 2 for i, j in zip(rgb, rgb_values)])
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        return closest_color
    
    def analyze_colors_from_array(self, image_array) -> Dict[str, Any]:
        """
        Analyze colors from numpy array (in-memory processing)
        
        Args:
            image_array: Image as numpy array or PIL Image
            
        Returns:
            Dict containing color analysis results
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                from PIL import Image as PILImage
                image = PILImage.fromarray(image_array)
            else:
                image = image_array  # assume it's already a PIL Image
            
            return self._process_image_for_colors(image)
            
        except Exception as e:
            logger.error(f"Colors analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'processing_time': 0
            }
    
    def analyze_colors_multiregion(self, image_array, regions=4) -> Dict[str, Any]:
        """
        Analyze colors from multiple regions of the image for comprehensive palette
        
        Args:
            image_array: Image as numpy array or PIL Image
            regions: Number of regions to divide image into (4=quadrants, 9=3x3 grid, etc)
            
        Returns:
            Dict containing combined color analysis results from all regions
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                from PIL import Image as PILImage
                image = PILImage.fromarray(image_array)
            else:
                image = image_array  # assume it's already a PIL Image
            
            return self._process_multiregion_colors(image, regions)
            
        except Exception as e:
            logger.error(f"Multi-region colors analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'processing_time': 0
            }
    
    def _process_image_for_colors(self, image: Image.Image) -> Dict[str, Any]:
        """
        Main processing function - takes PIL Image, returns color analysis data
        This is the core business logic, separated from HTTP concerns
        Uses pure in-memory processing without temporary files
        Copied exactly from process_image_for_colors() in REST.py
        """
        start_time = time.time()
        
        try:
            # Get dominant color using our in-memory implementation
            dominant_color = self._get_dominant_from_image(image)
            dr, dg, db = dominant_color[0], dominant_color[1], dominant_color[2]
            dhex = self._rgb2hex(dr, dg, db)
            
            # Get dominant color names for configured systems
            dominant_colors = {}
            primary_marker_hex = dhex  # Default to pixel hex as fallback
            
            for system in self.color_systems:
                color_name = self.get_color_name(dominant_color, system)
                if color_name:
                    formatted = self._format_copic(color_name)
                    dominant_colors[system] = f"{formatted[0]} ({formatted[1]})"
                    
                    # Use the official marker hex value from our database
                    marker_rgb = self.get_color_by_name(color_name, system)
                    if marker_rgb:
                        primary_marker_hex = self._rgb2hex(marker_rgb[0], marker_rgb[1], marker_rgb[2])
            
            # Get color palette using our in-memory implementation
            palette_colors = self._get_palette_from_image(image)
            
            # Build palette arrays with configured systems
            palette_colors_processed = []
            for p in palette_colors:
                clr = p[1]
                
                # Get color names for configured systems
                color_entry = {}
                temperature = "neutral"  # Default temperature
                marker_hex = None
                
                for system in self.color_systems:
                    color_name = self.get_color_name(clr, system)
                    
                    if color_name:
                        formatted = self._format_copic(color_name)
                        color_str = f"{formatted[0]} ({formatted[1]})"
                        color_entry[system] = color_str
                        
                        # Use the official marker hex value from our database
                        if not marker_hex:  # Use first found marker's hex
                            marker_rgb = self.get_color_by_name(color_name, system)
                            if marker_rgb:
                                marker_hex = self._rgb2hex(marker_rgb[0], marker_rgb[1], marker_rgb[2])
                        
                        # Only use Copic systems for temperature calculation
                        if temperature == "neutral" and 'copic' in system:
                            temperature = self._get_color_temperature(color_str)
                
                # Set hex to official marker hex, fallback to pixel hex
                color_entry["hex"] = marker_hex if marker_hex else self._rgb2hex(clr[0], clr[1], clr[2])
                
                color_entry["temperature"] = temperature
                
                # Only add colors that have at least one color system match
                if any(key in color_entry for key in self.color_systems):
                    palette_colors_processed.append(color_entry)
            
            # Remove duplicates based on hex value
            unique_palette = []
            seen_hex = set()
            for color in palette_colors_processed:
                if color["hex"] not in seen_hex:
                    unique_palette.append(color)
                    seen_hex.add(color["hex"])
            
            # Calculate palette temperature - only works with Copic systems
            temp_colors = []
            copic_system = None
            for system in self.color_systems:
                if 'copic' in system:
                    copic_system = system
                    break
            
            if copic_system:
                for c in unique_palette:
                    if copic_system in c:
                        temp_colors.append({"color": c[copic_system]})
            
            palette_temp = self._calculate_palette_temperature(temp_colors) if temp_colors else "neutral"
            
            # Get primary color temperature - only works with Copic systems
            primary_temp = "neutral"
            if copic_system and copic_system in dominant_colors:
                primary_temp = self._get_color_temperature(dominant_colors[copic_system])
            
            analysis_time = round(time.time() - start_time, 3)
            
            # Build response in V3 format with only configured systems
            primary_info = {"hex": primary_marker_hex, "temperature": primary_temp}
            primary_info.update(dominant_colors)  # Add configured color systems
            
            color_prediction = {
                "primary": primary_info,
                "palette": {
                    "temperature": palette_temp,
                    "colors": unique_palette
                }
            }
            
            return {
                "success": True,
                "predictions": [color_prediction],
                "processing_time": analysis_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Color analysis failed: {str(e)}",
                "processing_time": round(time.time() - start_time, 3)
            }
    
    def _process_multiregion_colors(self, image: Image.Image, regions: int) -> Dict[str, Any]:
        """
        Process image in multiple regions and combine color analysis results
        """
        start_time = time.time()
        
        try:
            # Calculate grid dimensions based on regions count
            if regions == 4:
                grid_rows, grid_cols = 2, 2
            elif regions == 9:
                grid_rows, grid_cols = 3, 3
            elif regions == 16:
                grid_rows, grid_cols = 4, 4
            else:
                # Default to quadrants for other values
                grid_rows, grid_cols = 2, 2
                regions = 4
            
            width, height = image.size
            region_width = width // grid_cols
            region_height = height // grid_rows
            
            # Collect all colors from all regions
            all_palette_colors = []
            region_dominant_colors = []
            
            logger.info(f"Analyzing {regions} regions ({grid_rows}x{grid_cols})")
            
            for row in range(grid_rows):
                for col in range(grid_cols):
                    # Calculate region boundaries
                    left = col * region_width
                    top = row * region_height
                    right = left + region_width if col < grid_cols - 1 else width
                    bottom = top + region_height if row < grid_rows - 1 else height
                    
                    # Crop region from image
                    region = image.crop((left, top, right, bottom))
                    
                    # Analyze this region
                    region_result = self._process_image_for_colors(region)
                    
                    if region_result.get('success') and region_result.get('predictions'):
                        prediction = region_result['predictions'][0]
                        
                        # Collect dominant color info
                        if 'primary' in prediction:
                            region_dominant_colors.append(prediction['primary'])
                        
                        # Collect palette colors
                        if 'palette' in prediction and 'colors' in prediction['palette']:
                            all_palette_colors.extend(prediction['palette']['colors'])
            
            # De-duplicate colors by hex value and merge color system data
            unique_colors = {}
            for color in all_palette_colors:
                hex_val = color.get('hex')
                if hex_val:
                    if hex_val in unique_colors:
                        # Merge color system data (prefer non-empty values)
                        for system in self.color_systems:
                            if system in color and color[system].strip():
                                unique_colors[hex_val][system] = color[system]
                        # Keep temperature if it's not neutral
                        if color.get('temperature', 'neutral') != 'neutral':
                            unique_colors[hex_val]['temperature'] = color['temperature']
                    else:
                        unique_colors[hex_val] = color.copy()
            
            # Convert back to list and sort by color system availability
            unique_palette = list(unique_colors.values())
            
            # Sort by number of color systems matched (more complete colors first)
            unique_palette.sort(key=lambda c: sum(1 for sys in self.color_systems if sys in c and c[sys].strip()), reverse=True)
            
            # Determine primary color from region dominants (most common hex)
            primary_color = None
            if region_dominant_colors:
                hex_counts = {}
                for dom in region_dominant_colors:
                    hex_val = dom.get('hex')
                    if hex_val:
                        hex_counts[hex_val] = hex_counts.get(hex_val, 0) + 1
                
                # Get most frequent hex
                most_common_hex = max(hex_counts.keys(), key=hex_counts.get) if hex_counts else None
                
                if most_common_hex:
                    # Find the dominant color data with most complete color system info
                    candidates = [dom for dom in region_dominant_colors if dom.get('hex') == most_common_hex]
                    primary_color = max(candidates, key=lambda c: sum(1 for sys in self.color_systems if sys in c and c[sys].strip()))
            
            # Fallback to first unique color if no primary found
            if not primary_color and unique_palette:
                primary_color = unique_palette[0]
            
            # Calculate overall palette temperature
            temp_colors = []
            copic_system = None
            for system in self.color_systems:
                if 'copic' in system:
                    copic_system = system
                    break
            
            if copic_system:
                for c in unique_palette:
                    if copic_system in c:
                        temp_colors.append({"color": c[copic_system]})
            
            palette_temp = self._calculate_palette_temperature(temp_colors) if temp_colors else "neutral"
            
            # Get primary color temperature
            primary_temp = "neutral"
            if primary_color and copic_system and copic_system in primary_color:
                primary_temp = self._get_color_temperature(primary_color[copic_system])
            
            analysis_time = round(time.time() - start_time, 3)
            
            color_prediction = {
                "primary": primary_color or {"hex": "#000000", "temperature": "neutral"},
                "palette": {
                    "temperature": palette_temp,
                    "colors": unique_palette
                }
            }
            
            logger.info(f"Multi-region analysis completed: {len(unique_palette)} unique colors from {regions} regions")
            
            return {
                "success": True,
                "predictions": [color_prediction],
                "processing_time": analysis_time,
                "metadata": {
                    "regions_analyzed": regions,
                    "unique_colors_found": len(unique_palette)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Multi-region color analysis failed: {str(e)}",
                "processing_time": round(time.time() - start_time, 3)
            }
    
    def _get_colors_from_image(self, image: Image.Image):
        """Extract color groups from PIL Image using proper Haishoku API"""
        import uuid
        # Create temporary file for Haishoku (it requires file path)
        tmp_path = f"temp_color_analysis_{uuid.uuid4().hex[:8]}.jpg"
        image.save(tmp_path, format='JPEG')
            
        try:
            # Use Haishoku's public getPalette method
            palette = Haishoku.getPalette(tmp_path)
            # Convert to the expected format (count, color_tuple)
            image_colors = [(int(p[0] * 1000), p[1]) for p in palette]  # Scale percentage to count
            return image_colors
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _get_colors_mean_from_image(self, image: Image.Image):
        """Get color means from PIL Image using proper Haishoku API"""
        import uuid
        # Create temporary file for Haishoku (it requires file path)
        tmp_path = f"temp_color_analysis_{uuid.uuid4().hex[:8]}.jpg"
        image.save(tmp_path, format='JPEG')
            
        try:
            # Use Haishoku's public getColorsMean method
            colors_mean = Haishoku.getColorsMean(tmp_path)
            # colors_mean is already in the format [(count, (r,g,b)), ...]
            return colors_mean
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _get_dominant_from_image(self, image: Image.Image):
        """Get dominant color directly from PIL Image - copied from REST.py"""
        colors_mean = self._get_colors_mean_from_image(image)
        colors_mean = sorted(colors_mean, reverse=True)
        dominant_tuple = colors_mean[0]
        dominant = dominant_tuple[1]
        return dominant
    
    def _get_palette_from_image(self, image: Image.Image):
        """Get color palette directly from PIL Image - copied from REST.py"""
        colors_mean = self._get_colors_mean_from_image(image)
        
        # Calculate percentages (following Haishoku's logic)
        palette_tmp = []
        count_sum = 0
        for c_m in colors_mean:
            count_sum += c_m[0]
            palette_tmp.append(c_m)
        
        # Calculate the percentage
        palette = []
        for p in palette_tmp:
            pp = '%.2f' % (p[0] / count_sum)
            tp = (float(pp), p[1])
            palette.append(tp)
        
        return palette
    
    def _rgb2hex(self, r, g, b):
        """Convert RGB to hex - copied from REST.py"""
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    def _format_copic(self, name):
        """Format copic name - copied from REST.py"""
        copic = []
        copic_name = re.sub('[\(\[].*?[\)\]]', '', str(name)).strip()
        copic_code = name.replace(copic_name,"").replace("(","").replace(")","").strip()

        copic.append(copic_name)
        copic.append(copic_code)
        
        return copic
    
    def _extract_copic_prefix(self, copic_name):
        """Extract the color family prefix from a Copic color name like 'Slate (BV29)' -> 'BV' - copied from REST.py"""
        if not copic_name:
            return None
        
        # Extract code from parentheses: "Slate (BV29)" -> "BV29"
        match = re.search(r'\(([^)]+)\)', copic_name)
        if not match:
            return None
        
        code = match.group(1)
        
        # Extract prefix letters: "BV29" -> "BV"
        prefix_match = re.match(r'^([A-Z]+)', code)
        if prefix_match:
            return prefix_match.group(1)
        
        return None
    
    def _get_color_temperature(self, copic_name):
        """Determine color temperature based on Copic color family prefix - copied from REST.py"""
        prefix = self._extract_copic_prefix(copic_name)
        if not prefix:
            return "neutral"
        
        # Temperature mapping based on Copic color families
        cool_prefixes = ['BV', 'B', 'BG', 'G', 'YG', 'C']  # Cool colors + Cool Gray
        warm_prefixes = ['RV', 'R', 'YR', 'Y', 'W']         # Warm colors + Warm Gray
        # Everything else (N, T, E, F) is neutral
        
        if prefix in cool_prefixes:
            return "cool"
        elif prefix in warm_prefixes:
            return "warm"
        else:
            return "neutral"
    
    def _calculate_palette_temperature(self, palette_colors):
        """Calculate overall palette temperature based on color distribution - copied from REST.py"""
        if not palette_colors:
            return "neutral"
        
        cool_count = warm_count = neutral_count = 0
        
        for color_info in palette_colors:
            color_name = color_info.get('color', '')
            temperature = self._get_color_temperature(color_name)
            
            if temperature == "cool":
                cool_count += 1
            elif temperature == "warm":
                warm_count += 1
            else:
                neutral_count += 1
        
        # Determine overall temperature
        if cool_count > warm_count:
            return "cool"
        elif warm_count > cool_count:
            return "warm"
        else:
            return "neutral"