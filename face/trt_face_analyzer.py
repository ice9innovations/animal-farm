#!/usr/bin/env python3
"""
TensorRT-accelerated Face Analyzer using ONNX Runtime

Replaces MediaPipe CPU inference with TRT/CUDA GPU inference.
Uses the BlazeFace back (full-range) model: face_detection_back_256x256_float32.onnx

Model outputs (896 anchors across two feature map heads):
  Identity:0   [1, 512, 1]  — score logits, head 1 (16×16 grid, 2 anchors/cell)
  Identity_1:0 [1, 384, 1]  — score logits, head 2 (8×8 grid, 6 anchors/cell)
  Identity_2:0 [1, 512, 16] — box + keypoints, head 1
  Identity_3:0 [1, 384, 16] — box + keypoints, head 2

16 values per anchor: [cx, cy, w, h, kp0x, kp0y, kp1x, kp1y, ..., kp5x, kp5y]
6 keypoints: right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear
"""

import math
import numpy as np
import onnxruntime as ort
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE = 256
TRT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'trt_cache')

# Score threshold before NMS
SCORE_THRESHOLD = 0.5
# IoU threshold for NMS
IOU_THRESHOLD = 0.3

# Anchor grid layout for face_detection_back_256x256:
#   Head 1: 16×16 feature map, 2 anchors per cell → 512 anchors
#   Head 2:  8×8 feature map, 6 anchors per cell  → 384 anchors
_ANCHOR_CONFIG = [
    {'grid': 16, 'anchors_per_cell': 2},   # 512
    {'grid':  8, 'anchors_per_cell': 6},   # 384
]


def _generate_anchors() -> np.ndarray:
    """
    Generate anchor centre coordinates (cx, cy) in [0,1] for the back model.
    Returns float32 array of shape [896, 2].
    """
    anchors = []
    for cfg in _ANCHOR_CONFIG:
        g = cfg['grid']
        n = cfg['anchors_per_cell']
        for y in range(g):
            for x in range(g):
                cx = (x + 0.5) / g
                cy = (y + 0.5) / g
                for _ in range(n):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)  # [896, 2]


# Pre-compute anchors once at module load
_ANCHORS = _generate_anchors()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Decode raw box offsets into absolute [0,1] coordinates.

    raw_boxes: [N, 16] — [cx_delta, cy_delta, w, h, kp0x, kp0y, ...]
    anchors:   [N, 2]  — anchor (cx, cy)
    Returns:   [N, 16] decoded
    """
    decoded = raw_boxes.copy()
    # Box centre: offset / input_size + anchor_centre
    decoded[:, 0] = raw_boxes[:, 0] / MODEL_INPUT_SIZE + anchors[:, 0]  # cx
    decoded[:, 1] = raw_boxes[:, 1] / MODEL_INPUT_SIZE + anchors[:, 1]  # cy
    decoded[:, 2] = raw_boxes[:, 2] / MODEL_INPUT_SIZE                   # w
    decoded[:, 3] = raw_boxes[:, 3] / MODEL_INPUT_SIZE                   # h
    # Keypoints: same offset scheme as box centre
    for k in range(6):
        decoded[:, 4 + k * 2]     = raw_boxes[:, 4 + k * 2]     / MODEL_INPUT_SIZE + anchors[:, 0]
        decoded[:, 4 + k * 2 + 1] = raw_boxes[:, 4 + k * 2 + 1] / MODEL_INPUT_SIZE + anchors[:, 1]
    return decoded


def _nms(boxes_cxcywh: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Non-maximum suppression on cx/cy/w/h boxes.
    Returns indices of kept detections, sorted by score descending.
    """
    if len(scores) == 0:
        return []

    # Convert cx/cy/w/h → x1/y1/x2/y2
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]

    return kept


class TRTFaceAnalyzer:
    """
    TensorRT/CUDA face analyzer using BlazeFace back (full-range) ONNX model.
    Drop-in replacement for FaceAnalyzer with the same output format.
    """

    def __init__(self, model_path: str, use_gpu: bool = True):
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)

        if use_gpu:
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 1 << 27,  # 128MB
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.abspath(TRT_CACHE_DIR),
                }),
                ('CUDAExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        actual_provider = self.session.get_providers()[0]
        logger.info(f"✅ TRTFaceAnalyzer initialized — provider: {actual_provider}")

    def _preprocess(self, pil_image: Image.Image) -> Tuple[np.ndarray, int, int]:
        """Resize to 256×256, normalise to [0,1], return batch tensor + original dims."""
        orig_w, orig_h = pil_image.size
        img = pil_image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [256, 256, 3]
        return np.expand_dims(arr, 0), orig_w, orig_h   # [1, 256, 256, 3]

    def analyze_faces_from_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Detect faces in a PIL RGB image.
        Returns the same structure as FaceAnalyzer.analyze_faces_from_image.
        """
        orig_w, orig_h = pil_image.size

        try:
            inp, orig_w, orig_h = self._preprocess(pil_image)
            outputs = self.session.run(None, {self.input_name: inp})

            # Unpack outputs
            scores_h1 = outputs[0][0]  # [512, 1]
            scores_h2 = outputs[1][0]  # [384, 1]
            boxes_h1  = outputs[2][0]  # [512, 16]
            boxes_h2  = outputs[3][0]  # [384, 16]

            raw_scores = np.concatenate([scores_h1, scores_h2], axis=0)[:, 0]  # [896]
            raw_boxes  = np.concatenate([boxes_h1,  boxes_h2],  axis=0)        # [896, 16]

            # Sigmoid scores
            scores = _sigmoid(raw_scores)

            # Filter by threshold before NMS
            mask = scores >= SCORE_THRESHOLD
            if not np.any(mask):
                return {'faces': [], 'dimensions': {'width': orig_w, 'height': orig_h}}

            filtered_scores = scores[mask]
            filtered_boxes  = _decode_boxes(raw_boxes[mask], _ANCHORS[mask])

            # NMS on box columns [0:4] (cx, cy, w, h)
            keep = _nms(filtered_boxes[:, :4], filtered_scores, IOU_THRESHOLD)

            faces = []
            for idx in keep:
                cx, cy, w, h = filtered_boxes[idx, :4]

                # Convert normalised [0,1] → pixel coords.
                # Force square bbox using geometric mean of w/h to match
                # MediaPipe's square bbox area.
                side = math.sqrt((w * orig_w) * (h * orig_h))
                cx_px = cx * orig_w
                cy_px = cy * orig_h
                px = int(cx_px - side / 2)
                py = int(cy_px - side / 2)
                pw = int(side)
                ph = int(side)

                # Decode 6 keypoints
                kp = filtered_boxes[idx, 4:]  # [12] — kp0x,kp0y,...,kp5x,kp5y
                keypoints = {
                    'right_eye':         [int(kp[0]  * orig_w), int(kp[1]  * orig_h)],
                    'left_eye':          [int(kp[2]  * orig_w), int(kp[3]  * orig_h)],
                    'nose_tip':          [int(kp[4]  * orig_w), int(kp[5]  * orig_h)],
                    'mouth_center':      [int(kp[6]  * orig_w), int(kp[7]  * orig_h)],
                    'right_ear_tragion': [int(kp[8]  * orig_w), int(kp[9]  * orig_h)],
                    'left_ear_tragion':  [int(kp[10] * orig_w), int(kp[11] * orig_h)],
                }

                faces.append({
                    'bbox':       [px, py, pw, ph],
                    'confidence': round(float(filtered_scores[idx]), 3),
                    'keypoints':  keypoints,
                    'method':     'blazeface_trt',
                })

            return {'faces': faces, 'dimensions': {'width': orig_w, 'height': orig_h}}

        except Exception as e:
            logger.error(f"TRT face detection error: {e}")
            return {'faces': [], 'dimensions': {'width': orig_w, 'height': orig_h}}
