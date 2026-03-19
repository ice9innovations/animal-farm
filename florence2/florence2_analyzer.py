#!/usr/bin/env python3
"""
Florence-2 Analyzer - Core multi-task vision analysis

Handles Florence-2 model loading and inference across all supported tasks.
Uses Microsoft Florence-2 via Hugging Face transformers with trust_remote_code.
"""

import os
import logging

# Map MODEL_DIR → HF_HOME before transformers is imported
if os.environ.get('MODEL_DIR'):
    os.environ['HF_HOME'] = os.environ['MODEL_DIR']

import torch
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# All tasks this service supports
VALID_TASKS = {
    "CAPTION",
    "DETAILED_CAPTION",
    "MORE_DETAILED_CAPTION",
    "OD",
    "DENSE_REGION_CAPTION",
    "OCR",
    "OCR_WITH_REGION",
    "CAPTION_TO_PHRASE_GROUNDING",
    "OPEN_VOCABULARY_DETECTION",
    "REFERRING_EXPRESSION_SEGMENTATION",
}

# Tasks that require a second text input alongside the image
TEXT_REQUIRED_TASKS = {
    "CAPTION_TO_PHRASE_GROUNDING",
    "OPEN_VOCABULARY_DETECTION",
    "REFERRING_EXPRESSION_SEGMENTATION",
}

# Map task name to Florence-2 task token
TASK_TOKENS = {
    "CAPTION":                          "<CAPTION>",
    "DETAILED_CAPTION":                 "<DETAILED_CAPTION>",
    "MORE_DETAILED_CAPTION":            "<MORE_DETAILED_CAPTION>",
    "OD":                               "<OD>",
    "DENSE_REGION_CAPTION":             "<DENSE_REGION_CAPTION>",
    "OCR":                              "<OCR>",
    "OCR_WITH_REGION":                  "<OCR_WITH_REGION>",
    "CAPTION_TO_PHRASE_GROUNDING":      "<CAPTION_TO_PHRASE_GROUNDING>",
    "OPEN_VOCABULARY_DETECTION":        "<OPEN_VOCABULARY_DETECTION>",
    "REFERRING_EXPRESSION_SEGMENTATION":"<REFERRING_EXPRESSION_SEGMENTATION>",
}


class FlorenceAnalyzer:
    """Core Florence-2 multi-task vision analysis"""

    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.model = None
        self.processor = None
        logger.info(f"FlorenceAnalyzer initialized - Device: {self.device}, dtype: {self.torch_dtype}")

    def initialize(self) -> bool:
        """Load model and processor - fail fast if unavailable"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            logger.info(f"Loading Florence-2 model: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            self.model.eval()

            logger.info(f"Florence-2 model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Error loading Florence-2 model: {str(e)}")
            return False

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Run image preprocessing once and return pixel_values + image_size.
        Call this before _run_task when processing multiple tasks against the same image.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_inputs = self.processor.image_processor(image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device, self.torch_dtype)
        return pixel_values, (image.width, image.height)

    def _run_task(
        self,
        pixel_values: torch.Tensor,
        image_size: Tuple[int, int],
        task: str,
        image: "Image.Image",
        text_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one task with pre-computed pixel_values.

        image is required because Florence2Processor translates task tokens into
        natural-language prompts before tokenizing — calling processor.tokenizer()
        directly bypasses that translation and produces wrong token IDs.
        The pixel_values from this processor call are discarded; the pre-computed
        ones (shared across batch tasks) are passed to model.generate instead.

        Validation is the caller's responsibility (task in VALID_TASKS, text present if required).
        """
        import time
        task_token = TASK_TOKENS[task]
        prompt = task_token if not text_input else f"{task_token}{text_input}"

        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )
            processing_time = time.time() - t0

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=image_size,
            )

            return {"success": True, "task": task, "raw": parsed[task_token], "processing_time": processing_time}

        except Exception as e:
            logger.error(f"Florence-2 inference error (task={task}): {e}")
            return {"success": False, "task": task, "error": str(e)}

    def analyze(self, image: Image.Image, task: str, text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Florence-2 inference for a single task.

        Args:
            image: PIL Image (RGB)
            task: One of VALID_TASKS
            text_input: Required for TEXT_REQUIRED_TASKS; ignored for image-only tasks
        """
        if task not in VALID_TASKS:
            return {
                "success": False,
                "error": f"Unknown task '{task}'. Valid tasks: {sorted(VALID_TASKS)}"
            }
        if task in TEXT_REQUIRED_TASKS and not text_input:
            return {
                "success": False,
                "error": f"Task '{task}' requires the 'text' parameter"
            }

        pixel_values, image_size = self.preprocess_image(image)
        return self._run_task(pixel_values, image_size, task, image, text_input)

    def analyze_batch(
        self,
        image: Image.Image,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple tasks against one image, preprocessing pixel_values once.

        Args:
            image: PIL Image (RGB)
            tasks: list of {"task": str, "text": str (optional)}

        Returns:
            List of result dicts in the same order as tasks, each with
            'success', 'task', and either 'raw' or 'error'.
        """
        pixel_values, image_size = self.preprocess_image(image)
        results = []
        for spec in tasks:
            task = spec.get("task", "")
            text_input = spec.get("text") or None
            if task not in VALID_TASKS:
                results.append({
                    "success": False,
                    "task": task,
                    "error": f"Unknown task '{task}'. Valid tasks: {sorted(VALID_TASKS)}"
                })
                continue
            if task in TEXT_REQUIRED_TASKS and not text_input:
                results.append({
                    "success": False,
                    "task": task,
                    "error": f"Task '{task}' requires the 'text' parameter"
                })
                continue
            results.append(self._run_task(pixel_values, image_size, task, image, text_input))
        return results
