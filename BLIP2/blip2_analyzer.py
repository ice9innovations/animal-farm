#!/usr/bin/env python3
"""
BLIP2 Caption Analyzer - Core caption generation functionality

Handles BLIP2 model loading, image preprocessing, and caption generation
using LAVIS BLIP2 (Bootstrapping Language-Image Pre-training v2) model.
"""

import os
import sys
import logging
import torch
from PIL import Image
from typing import Dict, Any, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lavis.models import load_model_and_preprocess

logger = logging.getLogger(__name__)


class Blip2Analyzer:
    """Core BLIP2 caption analysis functionality"""
    
    def __init__(self, 
                 model_name: str = "blip_caption",
                 model_type: str = "large_coco"):
        """Initialize BLIP2 analyzer with model configuration"""
        
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vis_processors = None
        
        logger.info(f"✅ Blip2Analyzer initialized - Device: {self.device}")
    
    def initialize(self) -> bool:
        """Initialize BLIP2 model - fail fast if it doesn't work"""
        try:
            logger.info(f"Loading BLIP2 model: {self.model_name} ({self.model_type})")
            
            # Load model and processors using LAVIS
            self.model, self.vis_processors, _ = load_model_and_preprocess(
                name=self.model_name, 
                model_type=self.model_type, 
                is_eval=True, 
                device=self.device
            )
            
            logger.info("✅ BLIP2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading BLIP2 model: {str(e)}")
            return False
    
    def analyze_caption_from_array(self, image_array) -> Dict[str, Any]:
        """
        Generate caption from image array (in-memory processing)
        
        Args:
            image_array: Image as numpy array or PIL Image
            
        Returns:
            Dict containing caption generation results
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                import numpy as np
                if isinstance(image_array, np.ndarray):
                    image = Image.fromarray(image_array)
                else:
                    image = image_array
            else:
                # Assume it's already a PIL Image
                image = image_array
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image using LAVIS processors
            processed_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                caption_list = self.model.generate({"image": processed_image})
            
            # Extract the first caption
            caption = caption_list[0] if caption_list else "No caption generated"
            
            return {
                'success': True,
                'caption': caption
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                'success': False,
                'error': str(e),
                'caption': None
            }
    
    def generate_multiple_captions(self, image_array, num_captions: int = 3) -> Dict[str, Any]:
        """
        Generate multiple captions using nucleus sampling
        
        Args:
            image_array: Image as numpy array or PIL Image
            num_captions: Number of captions to generate
            
        Returns:
            Dict containing multiple caption generation results
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                import numpy as np
                if isinstance(image_array, np.ndarray):
                    image = Image.fromarray(image_array)
                else:
                    image = image_array
            else:
                # Assume it's already a PIL Image
                image = image_array
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image using LAVIS processors
            processed_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            
            # Generate multiple captions with nucleus sampling
            with torch.no_grad():
                caption_list = self.model.generate(
                    {"image": processed_image}, 
                    use_nucleus_sampling=True, 
                    num_captions=num_captions
                )
            
            return {
                'success': True,
                'captions': caption_list,
                'primary_caption': caption_list[0] if caption_list else "No caption generated"
            }
            
        except Exception as e:
            logger.error(f"Error generating multiple captions: {e}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': None
            }