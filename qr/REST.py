#!/usr/bin/env python3
"""
QR code and barcode scanner service using pyzbar.

Detects and decodes QR codes, EAN/UPC barcodes, and other common symbologies.
Returns raw decoded payloads with bounding boxes — no interpretation of content.
"""
import io
import os
import time
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

load_dotenv()

PORT_STR = os.getenv('PORT')
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
PORT = int(PORT_STR)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


def decode_image(image: Image.Image) -> dict:
    """Detect and decode all QR codes and barcodes in the image.

    Returns a result dict with keys: success, codes, processing_time.
    On failure: success=False, error=str.
    """
    start_time = time.time()
    try:
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        decoded = pyzbar_decode(image)

        codes = []
        for item in decoded:
            try:
                data_str = item.data.decode('utf-8')
            except UnicodeDecodeError:
                data_str = item.data.decode('latin-1', errors='replace')

            codes.append({
                'type': item.type,
                'data': data_str,
                'bbox': {
                    'x':      item.rect.left,
                    'y':      item.rect.top,
                    'width':  item.rect.width,
                    'height': item.rect.height,
                },
            })

        return {
            'success':         True,
            'codes':           codes,
            'processing_time': round(time.time() - start_time, 3),
        }

    except Exception as e:
        return {
            'success':         False,
            'error':           str(e),
            'processing_time': round(time.time() - start_time, 3),
        }


def _image_from_request():
    """Extract a PIL Image from the current Flask request.

    Accepts multipart file upload (POST), or url/file query params.
    Returns (image, None) on success, (None, error_str) on failure.
    """
    if request.method == 'POST' and 'file' in request.files:
        uploaded = request.files['file']
        if not uploaded.filename:
            return None, 'No file selected'

        uploaded.seek(0, 2)
        size = uploaded.tell()
        uploaded.seek(0)

        if size > MAX_FILE_SIZE:
            return None, f'File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)'

        try:
            return Image.open(io.BytesIO(uploaded.read())), None
        except Exception as e:
            return None, f'Failed to open image: {e}'

    url  = request.args.get('url')  or request.form.get('url')
    path = request.args.get('file') or request.form.get('file')

    if url and path:
        return None, 'Cannot specify both url and file'

    if url:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            if len(r.content) > MAX_FILE_SIZE:
                return None, 'Downloaded file too large'
            return Image.open(io.BytesIO(r.content)), None
        except Exception as e:
            return None, f'Failed to download image: {e}'

    if path:
        if not os.path.exists(path):
            return None, f'File not found: {path}'
        try:
            return Image.open(path), None
        except Exception as e:
            return None, f'Failed to load image: {e}'

    return None, 'Provide a file upload, url parameter, or file parameter'


@app.route('/health', methods=['GET'])
def health():
    """Health check — verifies pyzbar is importable and functional."""
    try:
        test_image = Image.new('RGB', (10, 10), color='white')
        pyzbar_decode(test_image)
        status = 'healthy'
    except Exception as e:
        return jsonify({
            'status':  'unhealthy',
            'reason':  str(e),
            'service': 'QR Code Scanner',
        }), 503

    return jsonify({
        'status':    status,
        'service':   'QR Code Scanner',
        'library':   'pyzbar',
        'endpoints': [
            'GET  /health',
            'POST /analyze',
            'POST /v3/analyze',
        ],
    })


def _analyze():
    """Shared handler for all analyze endpoints."""
    start_time = time.time()

    def err(msg, code=400):
        return jsonify({
            'service':     'qr',
            'status':      'error',
            'predictions': [],
            'error':       {'message': msg},
            'metadata':    {'processing_time': round(time.time() - start_time, 3)},
        }), code

    image, error = _image_from_request()
    if error:
        return err(error)

    result = decode_image(image)
    if not result['success']:
        return err(result['error'], 500)

    codes = result['codes']
    return jsonify({
        'service': 'qr',
        'status':  'success',
        'predictions': [{
            'has_codes':  len(codes) > 0,
            'codes':      codes,
            'code_count': len(codes),
        }],
        'metadata': {
            'processing_time': result['processing_time'],
            'model_info':      {'library': 'pyzbar'},
        },
    })


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    return _analyze()


@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3():
    return _analyze()


if __name__ == '__main__':
    print(f"QR Code Scanner starting on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
