#!/usr/bin/env python3
"""
RembgAnalyzer - CPU background removal via rembg + ONNX

Wraps the rembg library to provide background removal without GPU dependency.
The model is configurable — birefnet-general is the default target for this
service but any rembg-supported session name works.

Model weights are downloaded on first initialization to ~/.u2net/.
"""

import logging
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RembgAnalyzer:
    """CPU background removal using rembg ONNX sessions"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = None

        logger.info(f"RembgAnalyzer initialized — model: {model_name}")

    def initialize(self) -> bool:
        """Initialize rembg session — downloads model weights on first run"""
        try:
            from rembg import new_session
            logger.info(f"RembgAnalyzer: Loading session '{self.model_name}' (downloads weights if not cached)")
            self.session = new_session(self.model_name)
            logger.info(f"RembgAnalyzer: Session ready")
            return True
        except Exception as e:
            logger.error(f"RembgAnalyzer: Failed to initialize session '{self.model_name}': {e}")
            return False

    def remove_background(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run background removal on a PIL Image.

        Returns a dict with:
          - rgba: RGBA PIL Image (original pixels + soft alpha mask)
          - mask: Grayscale PIL Image (alpha channel only, soft edges)
          - width/height: original image dimensions
        """
        if not self.session:
            return {'success': False, 'error': 'Session not initialized'}

        try:
            if image.mode != 'RGB':
                logger.info(f"RembgAnalyzer: Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            width, height = image.size
            logger.info(f"RembgAnalyzer: Running inference on {width}x{height} image")

            from rembg import remove
            rgba_result = remove(image, session=self.session)

            mask = rgba_result.getchannel('A')

            logger.info("RembgAnalyzer: Inference complete")
            return {
                'success': True,
                'rgba': rgba_result,
                'mask': mask,
                'width': width,
                'height': height,
            }

        except Exception as e:
            logger.error(f"RembgAnalyzer: Inference error: {e}")
            return {'success': False, 'error': f"Inference failed: {str(e)}"}
