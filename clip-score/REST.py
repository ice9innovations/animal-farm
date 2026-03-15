import os
import sys
import time
import logging
import requests
import torch
import clip
from io import BytesIO
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
CLIP_MODEL = os.getenv('CLIP_MODEL')

if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not CLIP_MODEL:
    raise ValueError("CLIP_MODEL environment variable is required")

PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

logger.info(f"Loading CLIP model {CLIP_MODEL} on {device}...")
model, preprocess = clip.load(CLIP_MODEL, device=device)

if device == "cuda":
    model = model.half()
    logger.info("Applied FP16 — 50% VRAM reduction")

logger.info(f"CLIP model {CLIP_MODEL} ready")


def encode_image_only(image: Image.Image) -> Optional[list]:
    """Return a normalized CLIP image embedding as a plain Python list.

    Suitable for pgvector storage. Returns None on failure.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        if device == "cuda" and model.dtype == torch.float16:
            image_tensor = image_tensor.half()

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(image_tensor)
            else:
                image_features = model.encode_image(image_tensor)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.squeeze().float().cpu().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image_embedding

    except Exception as e:
        logger.error(f"encode_image_only failed: {e}")
        return None


def score_caption(image: Image.Image, caption: str) -> Optional[tuple]:
    """Compute cosine similarity between a PIL Image and a caption string.

    Returns (similarity, image_embedding, text_embedding) where embeddings are
    normalized float32 vectors as plain Python lists, suitable for pgvector storage.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        if device == "cuda" and model.dtype == torch.float16:
            image_tensor = image_tensor.half()

        text_tokens = clip.tokenize([caption], truncate=True).to(device)

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(text_tokens)
            else:
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

            image_embedding = image_features.squeeze().float().cpu().tolist()
            text_embedding = text_features.squeeze().float().cpu().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return similarity, image_embedding, text_embedding

    except Exception as e:
        logger.error(f"score_caption failed: {e}")
        return None


app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "clip-score",
        "model": CLIP_MODEL,
        "device": device,
        "endpoints": [
            "GET /health",
            "GET,POST /score - caption + (url | file | multipart)",
            "POST /embed/image - image embedding only (url | file | multipart)",
            "POST /embed/text - batch text-only embeddings",
            "(deprecated) /v3/score, /v3/embed/text",
        ]
    })


@app.route('/embed/text', methods=['POST'])
@app.route('/v3/embed/text', methods=['POST'])
def embed_text():
    """Return CLIP text embeddings for a batch of terms.

    Accepts JSON: {"terms": ["dog", "cat", "truck"]}
    Returns:      {"embeddings": {"dog": [...], "cat": [...], "truck": [...]}}

    No image required — runs the text encoder only. Embeddings are
    L2-normalized, identical to the text_embedding returned by /v3/score.
    """
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
        }), status_code

    try:
        if not request.is_json:
            return error_response("Request must be JSON with a 'terms' list")

        data = request.get_json()
        terms = data.get('terms', [])

        if not terms or not isinstance(terms, list):
            return error_response("Must provide a non-empty 'terms' list")
        if len(terms) > 500:
            return error_response("Too many terms (max 500)")

        # Deduplicate, strip whitespace, drop empties — preserve first-seen order
        seen = {}
        for t in terms:
            t = t.strip()
            if t and t not in seen:
                seen[t] = None
        unique_terms = list(seen.keys())

        if not unique_terms:
            return error_response("No valid terms provided")

        text_tokens = clip.tokenize(unique_terms, truncate=True).to(device)

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    text_features = model.encode_text(text_tokens)
            else:
                text_features = model.encode_text(text_tokens)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings_list = text_features.float().cpu().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        embeddings = {term: emb for term, emb in zip(unique_terms, embeddings_list)}

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "embeddings": embeddings,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "term_count": len(embeddings),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                },
            },
        })

    except Exception as e:
        logger.error(f"embed_text endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


@app.route('/embed/image', methods=['POST'])
def embeddings():
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        image = None

        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            uploaded_file.seek(0, 2)
            if uploaded_file.tell() > MAX_FILE_SIZE:
                return error_response("File too large (max 8MB)")
            uploaded_file.seek(0)
            try:
                image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open uploaded image: {e}", 500)

        else:
            if request.is_json:
                data = request.get_json()
                url = data.get('url')
                file_path = data.get('file')
            else:
                url = request.form.get('url') or request.args.get('url')
                file_path = request.form.get('file') or request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide image via multipart upload, 'url', or 'file' parameter")
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    if len(r.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded image too large (max 8MB)")
                    image = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to download image: {e}")
            else:
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to open image file: {e}", 500)

        image_embedding = encode_image_only(image)

        if image_embedding is None:
            return error_response("Failed to compute image embedding", 500)

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "image_embedding": image_embedding,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                }
            }
        })

    except Exception as e:
        logger.error(f"embeddings endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


@app.route('/score', methods=['GET', 'POST'])
@app.route('/v3/score', methods=['GET', 'POST'])
def score():
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "similarity_score": None,
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        # --- get caption ---
        if request.is_json:
            data = request.get_json()
            caption = data.get('caption')
        else:
            caption = request.form.get('caption') or request.args.get('caption')

        if not caption or not caption.strip():
            return error_response("Must provide non-empty 'caption' parameter")
        caption = caption.strip()

        # --- get image ---
        image = None

        if request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            uploaded_file.seek(0, 2)
            if uploaded_file.tell() > MAX_FILE_SIZE:
                return error_response("File too large (max 8MB)")
            uploaded_file.seek(0)
            try:
                image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open uploaded image: {e}", 500)

        else:
            if request.is_json:
                data = request.get_json()
                url = data.get('url')
                file_path = data.get('file')
            else:
                url = request.form.get('url') or request.args.get('url')
                file_path = request.form.get('file') or request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide image via multipart upload, 'url', or 'file' parameter")
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    if len(r.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded image too large (max 8MB)")
                    image = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to download image: {e}")
            else:
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to open image file: {e}", 500)

        result = score_caption(image, caption)

        if result is None:
            return error_response("Failed to compute similarity score", 500)

        similarity, image_embedding, text_embedding = result

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "similarity_score": round(float(similarity), 4),
            "caption": caption,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                }
            }
        })

    except Exception as e:
        logger.error(f"score endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info(f"Starting clip-score service on port {PORT}")
    logger.info(f"Model: {CLIP_MODEL} | Device: {device} | Private: {PRIVATE}")
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    app.run(host=host, port=PORT, debug=False, threaded=True)
