import logging, os, threading, time
from io import BytesIO
import requests, torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

load_dotenv(); logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
SERVICE = "nomic"
PORT = int(os.getenv("PORT", "7808"))
PRIVATE = os.getenv("PRIVATE", "false").lower() in {"true", "1", "yes"}
TEXT_MODEL = os.getenv("TEXT_MODEL", "nomic-ai/nomic-embed-text-v1.5")
VISION_MODEL = os.getenv("VISION_MODEL", "nomic-ai/nomic-embed-vision-v1.5")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(32 * 1024 * 1024)))
MAX_TEXTS = int(os.getenv("MAX_TEXTS", "500"))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
log.info("Loading Nomic text and vision models on %s", device)
text_model = SentenceTransformer(TEXT_MODEL, trust_remote_code=True, device=device)
image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL, trust_remote_code=True, use_fast=False)
vision_model = AutoModel.from_pretrained(VISION_MODEL, trust_remote_code=True, dtype=dtype).to(device).eval()
lock = threading.Lock(); app = Flask(__name__); CORS(app)

def error(message, start, code=400):
    return jsonify(service=SERVICE, status="error", error={"message": message}, metadata={"processing_time": round(time.time()-start, 3)}), code

def meta(start, dimensions):
    return {"processing_time": round(time.time()-start, 3), "dimensions": dimensions, "model_info": {
        "framework": "sentence-transformers/transformers", "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL, "device": device, "normalized": True}}

def open_image(data):
    if len(data) > MAX_FILE_SIZE: raise ValueError(f"Image too large (max {MAX_FILE_SIZE//1048576}MB)")
    try:
        image = Image.open(BytesIO(data)); image.load(); return image.convert("RGB")
    except Exception as exc: raise ValueError(f"Invalid image: {exc}") from exc

def request_image():
    content_type = (request.content_type or "").split(";", 1)[0]
    if content_type in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
        data = request.get_data(cache=False)
        if not data: raise ValueError("No image body provided")
        return open_image(data)
    if "file" in request.files:
        upload = request.files["file"]
        if not upload.filename: raise ValueError("No file selected")
        return open_image(upload.stream.read(MAX_FILE_SIZE + 1))
    body = request.get_json(silent=True) if request.is_json else {}; body = body if isinstance(body, dict) else {}
    url = body.get("url") or request.form.get("url") or request.args.get("url")
    path = body.get("file") or request.form.get("file") or request.args.get("file")
    if bool(url) == bool(path): raise ValueError("Provide exactly one of 'url' or 'file'")
    if url:
        try:
            with requests.get(url, timeout=15, stream=True) as response:
                response.raise_for_status(); data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > MAX_FILE_SIZE: raise ValueError(f"Image too large (max {MAX_FILE_SIZE//1048576}MB)")
            return open_image(bytes(data))
        except requests.RequestException as exc: raise ValueError(f"Failed to download image: {exc}") from exc
    if not os.path.isfile(path): raise ValueError(f"File not found: {path}")
    if os.path.getsize(path) > MAX_FILE_SIZE: raise ValueError(f"Image too large (max {MAX_FILE_SIZE//1048576}MB)")
    with open(path, "rb") as source: return open_image(source.read(MAX_FILE_SIZE + 1))

@app.get("/health")
def health():
    return jsonify(status="healthy", schema_version="health.v1", service=SERVICE, warnings=[], dependencies={},
        model={"name": "Nomic Embed v1.5 text + vision", "text_model": TEXT_MODEL, "vision_model": VISION_MODEL,
               "framework": "transformers", "device": device, "status": "loaded"},
        endpoints=["GET /health", "POST /embed/text", "POST /embed/image"])

@app.post("/embed/text")
def embed_text():
    start = time.time(); body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("terms"), list) or not body["terms"]:
        return error("Request must be JSON with a non-empty 'terms' list", start)
    if len(body["terms"]) > MAX_TEXTS: return error(f"Too many terms (max {MAX_TEXTS})", start)
    if any(not isinstance(term, str) for term in body["terms"]): return error("Every term must be a string", start)
    terms = list(dict.fromkeys(term.strip() for term in body["terms"] if term.strip()))
    if not terms: return error("No valid terms provided", start)
    task = body.get("task", "search_document")
    prefixes = {"search_document": "search_document: ", "search_query": "search_query: ",
                "classification": "classification: ", "clustering": "clustering: "}
    if task not in prefixes: return error(f"Invalid task; choose one of {', '.join(prefixes)}", start)
    try:
        with lock:
            vectors = text_model.encode([prefixes[task] + term for term in terms], normalize_embeddings=True, convert_to_numpy=True).tolist()
    except Exception as exc:
        log.exception("Text embedding failed"); return error(f"Failed to compute text embeddings: {exc}", start, 500)
    metadata = meta(start, len(vectors[0])); metadata.update(term_count=len(terms), task=task)
    return jsonify(service=SERVICE, status="success", embeddings=dict(zip(terms, vectors)), metadata=metadata)

@app.post("/embed/image")
def embed_image():
    start = time.time()
    try:
        values = image_processor(images=[request_image()], return_tensors="pt")
        values = {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device) for k, v in values.items()}
        with lock, torch.inference_mode():
            output = vision_model(**values).last_hidden_state[:, 0]
            vector = F.normalize(output, p=2, dim=1)[0].float().cpu().tolist()
    except ValueError as exc: return error(str(exc), start)
    except Exception as exc:
        log.exception("Image embedding failed"); return error(f"Failed to compute image embedding: {exc}", start, 500)
    return jsonify(service=SERVICE, status="success", image_embedding=vector, metadata=meta(start, len(vector)))

if __name__ == "__main__": app.run(host="127.0.0.1" if PRIVATE else "0.0.0.0", port=PORT, threaded=True)
