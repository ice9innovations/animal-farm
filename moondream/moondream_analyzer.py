#!/usr/bin/env python3
"""
Moondream Analyzer - Core caption generation functionality

Handles Moondream model loading and caption generation using
AutoModelForCausalLM with trust_remote_code=True.

Model source: vikhyatk/moondream2 (cached via HuggingFace)
"""

import logging
import torch
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MoondreamAnalyzer:
    """Core Moondream caption analysis functionality"""

    def __init__(self,
                 model_id: str = "vikhyatk/moondream2",
                 model_revision: Optional[str] = None,
                 caption_length: str = "normal"):
        self.model_id = model_id
        self.model_revision = model_revision
        self.caption_length = caption_length
        self.model = None

        logger.info(f"MoondreamAnalyzer initialized - model: {model_id}@{model_revision or 'latest'}, "
                    f"caption_length: {caption_length}")

    def initialize(self) -> bool:
        """Initialize Moondream model - fail fast if it doesn't work"""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            logger.error(f"transformers not installed: {e}")
            return False

        try:
            kwargs = {
                "trust_remote_code": True,
                "dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if self.model_revision:
                kwargs["revision"] = self.model_revision

            logger.info(f"Loading Moondream model {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **kwargs
            ).eval()

            logger.info(f"Moondream model loaded: {type(self.model).__name__}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            return False

    def analyze_caption_from_array(self, image) -> Dict[str, Any]:
        """Generate caption from PIL Image or numpy array"""
        try:
            if hasattr(image, 'shape'):
                from PIL import Image as PILImage
                image = PILImage.fromarray(image)
            return self._generate_caption(image)
        except Exception as e:
            logger.error(f"Moondream caption analysis error: {e}")
            return {'success': False, 'error': str(e), 'caption': ''}

    def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate caption for PIL Image using Moondream model"""
        if not self.model:
            return {'success': False, 'error': "Model not loaded", 'caption': ''}

        try:
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            logger.info(f"Generating caption (length={self.caption_length}) for image {image.size}")
            result = self.model.caption(image, length=self.caption_length)
            caption_text = result["caption"]

            logger.info(f"Generated caption: {caption_text}")
            return {'success': True, 'error': None, 'caption': caption_text}

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {'success': False, 'error': f"Caption generation failed: {str(e)}", 'caption': ''}

    def __del__(self):
        """Cleanup Moondream model resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
