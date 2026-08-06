#!/usr/bin/env python3
"""JoyCaption model loading and inference."""

import logging
import os
import threading
from typing import Any, Dict, Optional

if os.environ.get("MODEL_DIR"):
    os.environ["HF_HOME"] = os.environ["MODEL_DIR"]

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils import logging as transformers_logging

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_SYSTEM_PROMPT = "You are a helpful image captioner."
DEFAULT_PROMPT = (
    "Write one concise factual caption for this image in 25 words or fewer. "
    "Start with sfw, suggestive, or nsfw. Mention only the main visible subject "
    "and action. Do not list fine details."
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


class JoyCaptionAnalyzer:
    """Long-lived JoyCaption inference wrapper."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        precision: str = "8bit",
        device: str = "auto",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model_id = model_id
        self.precision = precision
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.system_prompt = system_prompt
        self.processor = None
        self.model = None
        self._lock = threading.Lock()
        self._configure_cuda_backend()

    def initialize(self) -> bool:
        """Load processor and model. Return False instead of raising for REST startup diagnostics."""
        try:
            self._validate_runtime()
            previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            finally:
                transformers_logging.set_verbosity(previous_verbosity)

            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                **self._model_kwargs(),
            )
            self.model.eval()
            if self.device == "cpu":
                self.model.to("cpu")

            logger.info(
                "JoyCaption loaded: model=%s device=%s precision=%s",
                self.model_id,
                self.device,
                self.precision,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to load JoyCaption: %s", exc)
            return False

    def caption(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: Optional[float] = 0.9,
        greedy: bool = False,
    ) -> Dict[str, Any]:
        """Generate one caption for a PIL image."""
        if self.model is None or self.processor is None:
            return {"success": False, "error": "Model not loaded", "caption": ""}

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            conversation = [
                {"role": "system", "content": (system_prompt or self.system_prompt).strip()},
                {"role": "user", "content": prompt.strip()},
            ]
            chat_prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(text=[chat_prompt], images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)
            if "pixel_values" in inputs:
                if self.precision in {"bf16", "4bit", "8bit"}:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                elif self.precision == "fp16":
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": not greedy,
                "suppress_tokens": None,
                "use_cache": True,
                "top_k": None,
            }
            if not greedy:
                generation_kwargs["temperature"] = temperature
                if top_p is not None:
                    generation_kwargs["top_p"] = top_p
            eos_token_id = self.processor.tokenizer.eos_token_id
            if eos_token_id is not None:
                generation_kwargs["pad_token_id"] = eos_token_id

            with self._lock, torch.no_grad():
                generated_ids = self.model.generate(**generation_kwargs)[0]

            generated_ids = generated_ids[inputs["input_ids"].shape[1]:]
            caption = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            return {"success": True, "error": None, "caption": caption}
        except Exception as exc:
            logger.exception("JoyCaption generation failed: %s", exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"success": False, "error": str(exc), "caption": ""}

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if requested not in {"cuda", "cpu"}:
            raise ValueError("DEVICE must be one of: auto, cuda, cpu")
        return requested

    def _validate_runtime(self) -> None:
        if self.precision not in {"bf16", "fp16", "fp32", "8bit", "4bit"}:
            raise ValueError("PRECISION must be one of: bf16, fp16, fp32, 8bit, 4bit")
        if self.device == "cpu" and self.precision in {"bf16", "fp16", "8bit", "4bit"}:
            raise RuntimeError(
                "JoyCaption is CUDA-oriented. Use DEVICE=cpu PRECISION=fp32 only "
                "if you intentionally want a very slow CPU fallback."
            )
        if self.precision in {"8bit", "4bit"} and self.device != "cuda":
            raise RuntimeError(f"PRECISION={self.precision} requires CUDA.")

    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if self.precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.precision == "fp32":
            kwargs["torch_dtype"] = torch.float32
        elif self.precision in {"8bit", "4bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Install bitsandbytes to use PRECISION=8bit or PRECISION=4bit."
                ) from exc

            skip_modules = _csv_env("BNB_SKIP_MODULES", ["vision_tower", "multi_modal_projector"])

            if self.precision == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=skip_modules,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=skip_modules,
                )
            kwargs["torch_dtype"] = "auto"

        if self.device == "cuda":
            kwargs["device_map"] = 0
        return kwargs

    def _configure_cuda_backend(self) -> None:
        if self.device != "cuda":
            return

        disable_cudnn = os.getenv("DISABLE_CUDNN", "auto").lower()
        should_disable = disable_cudnn in {"1", "true", "yes", "on"}

        if disable_cudnn == "auto" and torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            should_disable = major >= 12

        if should_disable:
            torch.backends.cudnn.enabled = False
            logger.info("JoyCaption: cuDNN disabled for CUDA inference")

        if _env_bool("CUDA_ALLOW_TF32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def __del__(self):
        if getattr(self, "model", None) is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
