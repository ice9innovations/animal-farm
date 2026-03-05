"""
Caption Summary REST service.

Accepts a set of VLM captions and optional noun/verb consensus data,
synthesizes them into a single descriptive sentence using a local LLM
(llama.cpp) or the Anthropic API, and returns the result.

POST /summarize
  Request (JSON):
    {
      "captions": {"blip": "...", "haiku": "...", ...},
      "nouns":    [{"canonical": "...", "category": "...", "vote_count": 2}, ...],
      "verbs":    [{"canonical": "...", "vote_count": 2}, ...]
    }
  Response (JSON):
    {
      "status": "success",
      "summary": "A golden retriever runs through a sun-dappled park.",
      "model": "Qwen3-VL-2B",
      "processing_time": 1.23
    }

GET /health
  Returns service and backend status.

Environment variables:
  PRIVATE             Required. "true" binds to 127.0.0.1, "false" to 0.0.0.0.
  PORT                Required. Port to listen on.
  SYNTHESIS_BACKEND   "llamacpp" (default) or "claude".
  LLAMA_SERVER_HOST   Base URL of llama.cpp server, e.g. http://127.0.0.1:11436.
                      Required when SYNTHESIS_BACKEND=llamacpp.
  SYNTHESIS_MODEL     Model name for logging / Claude model ID.
                      Default: "Qwen3-VL-2B" (llamacpp) or "claude-haiku-4-5-20251001" (claude).
  ANTHROPIC_API_KEY   Required when SYNTHESIS_BACKEND=claude.
  MAX_TOKENS          Max tokens in synthesized output. Default: 150.
"""

import os
import sys
import json
import time
import logging
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')

if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

PRIVATE = PRIVATE_STR.lower() in ('true', '1', 'yes')
PORT = int(PORT_STR)

SYNTHESIS_BACKEND = os.getenv('SYNTHESIS_BACKEND', 'llamacpp').lower()
LLAMA_SERVER_HOST = os.getenv('LLAMA_SERVER_HOST', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '75'))

# Model name used for logging and Claude API calls.
_DEFAULT_MODEL = (
    'claude-haiku-4-5-20251001' if SYNTHESIS_BACKEND == 'claude' else 'Qwen3-VL-2B'
)
SYNTHESIS_MODEL = os.getenv('SYNTHESIS_MODEL', _DEFAULT_MODEL)

if SYNTHESIS_BACKEND == 'llamacpp' and not LLAMA_SERVER_HOST:
    raise ValueError("LLAMA_SERVER_HOST is required when SYNTHESIS_BACKEND=llamacpp")
if SYNTHESIS_BACKEND == 'claude' and not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is required when SYNTHESIS_BACKEND=claude")

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_CONTEXT = (
    "Here are descriptions of an image from multiple vision models, "
    "along with consensus data about what the image contains."
)

SYNTHESIS_PROMPT_INSTRUCTION = (
    "Write one concise sentence describing the image. "
    "Do not copy or repeat the captions — synthesize them. "
    "Return only the sentence."
)


def _format_noun_consensus(nouns: list) -> str:
    if not nouns:
        return ''

    category_totals: dict = {}
    for noun in nouns:
        cat = noun.get('category', 'object')
        category_totals[cat] = category_totals.get(cat, 0) + noun.get('vote_count', 1)

    sorted_cats = sorted(category_totals.items(), key=lambda x: -x[1])
    category_line = ' '.join(f"{cat} {count}" for cat, count in sorted_cats)

    sorted_nouns = sorted(nouns, key=lambda n: -n.get('vote_count', 0))
    nouns_line = ' '.join(
        f"{n['canonical']} {n.get('category', 'object')} {n.get('vote_count', 1)}"
        for n in sorted_nouns
    )

    return f"Noun Consensus\n{category_line}\n{nouns_line}"


def _format_verb_consensus(verbs: list) -> str:
    if not verbs:
        return ''
    sorted_verbs = sorted(verbs, key=lambda v: -v.get('vote_count', 0))
    verbs_line = ' '.join(
        f"{v['canonical']} {v.get('vote_count', 1)}"
        for v in sorted_verbs
    )
    return f"Verb Consensus\n{verbs_line}"


def _format_captions(captions: dict) -> str:
    lines = ["VLM Captions"]
    for service, text in captions.items():
        lines.append(service)
        lines.append(text)
    return "\n".join(lines)


def build_prompt(captions: dict, nouns: list, verbs: list) -> str:
    parts = [SYNTHESIS_PROMPT_CONTEXT]

    noun_section = _format_noun_consensus(nouns)
    if noun_section:
        parts.append(noun_section)

    verb_section = _format_verb_consensus(verbs)
    if verb_section:
        parts.append(verb_section)

    parts.append(_format_captions(captions))
    parts.append(SYNTHESIS_PROMPT_INSTRUCTION)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def call_llamacpp(prompt: str) -> str:
    """Send a text-only chat request to the llama.cpp server."""
    url = f"{LLAMA_SERVER_HOST.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": SYNTHESIS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "repeat_penalty": 1.3,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get('choices', [])
    if not choices:
        raise ValueError("llama.cpp returned no choices")
    return choices[0].get('message', {}).get('content', '').strip()


def call_claude(prompt: str) -> str:
    """Send a text synthesis request to the Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def synthesize(prompt: str) -> str:
    if SYNTHESIS_BACKEND == 'claude':
        return call_claude(prompt)
    return call_llamacpp(prompt)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "error": "Bad request"}), 400


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "error": "Internal server error"}), 500


@app.route('/health', methods=['GET'])
def health():
    backend_ok = False
    backend_detail = ''
    try:
        if SYNTHESIS_BACKEND == 'llamacpp':
            r = requests.get(f"{LLAMA_SERVER_HOST.rstrip('/')}/health", timeout=3)
            backend_ok = r.status_code == 200
            backend_detail = r.json().get('status', '') if backend_ok else str(r.status_code)
        else:
            # Claude: assume available if API key is set
            backend_ok = bool(ANTHROPIC_API_KEY)
            backend_detail = 'api_key_present' if backend_ok else 'no_api_key'
    except Exception as e:
        backend_detail = str(e)

    return jsonify({
        "status": "healthy",
        "backend": SYNTHESIS_BACKEND,
        "backend_status": "ok" if backend_ok else "unavailable",
        "backend_detail": backend_detail,
        "model": SYNTHESIS_MODEL,
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    start_time = time.time()

    if not request.is_json:
        return jsonify({"status": "error", "error": "Request must be JSON"}), 400

    body = request.get_json()
    captions = body.get('captions', {})
    nouns = body.get('nouns', [])
    verbs = body.get('verbs', [])

    if not captions:
        return jsonify({"status": "error", "error": "No captions provided"}), 400

    try:
        prompt = build_prompt(captions, nouns, verbs)
        summary = synthesize(prompt)

        if not summary:
            return jsonify({"status": "error", "error": "Empty response from LLM backend"}), 500

        processing_time = round(time.time() - start_time, 3)

        logger.info(
            f"caption-synthesis: synthesized in {processing_time}s from "
            f"{len(captions)} captions — \"{summary[:80]}{'...' if len(summary) > 80 else ''}\""
        )

        return jsonify({
            "status": "success",
            "summary": summary,
            "model": SYNTHESIS_MODEL,
            "processing_time": processing_time,
        })

    except requests.RequestException as e:
        logger.error(f"caption-synthesis: backend request failed: {e}")
        return jsonify({"status": "error", "error": f"Backend unavailable: {e}"}), 503
    except Exception as e:
        logger.error(f"caption-synthesis: unexpected error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting caption-synthesis service on {host}:{PORT}")
    logger.info(f"Backend: {SYNTHESIS_BACKEND}, model: {SYNTHESIS_MODEL}")
    if SYNTHESIS_BACKEND == 'llamacpp':
        logger.info(f"llama.cpp server: {LLAMA_SERVER_HOST}")

    app.run(host=host, port=PORT, debug=False, use_reloader=False, threaded=True)
