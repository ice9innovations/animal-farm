#!/usr/bin/env python3
"""
BEN2 Analyzer - Background removal using BEN2 (Background Extraction Network 2)

Handles BEN2 model loading and inference. Returns RGBA images with soft alpha
masks — not binary black/white, full float alpha matte.

Model: PramaLLC/BEN2 (BEN2_Base.pth)
"""

import sys
import logging
import torch
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BEN2Analyzer:
    """Core BEN2 background removal functionality"""

    def __init__(self, model_path: str, code_dir: str, refine_foreground: bool = False):
        self.model_path = model_path
        self.code_dir = code_dir
        self.refine_foreground = refine_foreground
        self.model = None
        self.device: Optional[torch.device] = None

        logger.info(
            f"BEN2Analyzer initialized — model_path: {model_path}, "
            f"code_dir: {code_dir}, refine_foreground: {refine_foreground}"
        )

    def initialize(self) -> bool:
        """Initialize BEN2 model — fail fast if anything is wrong"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"BEN2: Using device: {self.device}")

            if self.code_dir not in sys.path:
                sys.path.insert(0, self.code_dir)

            try:
                import BEN2 as ben2_module
            except ImportError as e:
                logger.error(f"BEN2: Cannot import BEN2 from {self.code_dir}: {e}")
                return False

            logger.info(f"BEN2: Loading weights from {self.model_path}")
            self.model = ben2_module.BEN_Base()
            self.model.loadcheckpoints(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info("BEN2: Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"BEN2: Failed to initialize: {e}")
            return False

    def remove_background(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run background removal on a PIL Image.

        Returns a dict with:
          - rgba: RGBA PIL Image (original pixels + soft alpha mask)
          - mask: Grayscale PIL Image (alpha channel only, soft edges)
          - width/height: original image dimensions
        """
        if not self.model:
            return {'success': False, 'error': 'Model not loaded'}

        try:
            if image.mode != 'RGB':
                logger.info(f"BEN2: Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            width, height = image.size
            logger.info(f"BEN2: Running inference on {width}x{height} image")

            rgba_result = self.model.inference(image, refine_foreground=self.refine_foreground)

            # Extract the alpha channel as a standalone grayscale mask
            mask = rgba_result.getchannel('A')

            logger.info("BEN2: Inference complete")
            return {
                'success': True,
                'rgba': rgba_result,
                'mask': mask,
                'width': width,
                'height': height,
            }

        except Exception as e:
            logger.error(f"BEN2: Inference error: {e}")
            return {'success': False, 'error': f"Inference failed: {str(e)}"}

    def __del__(self):
        """Release GPU memory on cleanup"""
        if hasattr(self, 'model') and self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
