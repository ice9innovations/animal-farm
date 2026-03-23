#!/usr/bin/env python3
"""
TensorRT-accelerated Pose Analyzer using ONNX Runtime

Two-stage BlazePose pipeline:
  Stage 1 — pose_detection.onnx (224×224):
    Locates the person, extracts alignment keypoints (kp0=mid-hip, kp1=mid-shoulder)
    and computes a rotation-normalised ROI for Stage 2.

  Stage 2 — pose_landmark_heavy.onnx (256×256):
    Runs on the aligned ROI crop, returns 33 named landmarks + segmentation mask.

For pre-cropped person images (Windmill worker sends bbox crops) Stage 1 still
runs — it just finds the person within the crop, which is fast and keeps the
landmark quality high via rotation normalisation.
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
import math
import os
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DETECTION_INPUT_SIZE  = 224
LANDMARK_INPUT_SIZE   = 256
ROI_SCALE_FACTOR      = 2.5   # padding around the hip→shoulder span
                                # MediaPipe uses square_long=true + scale 1.5 on the
                                # bounding-box long-side; for the keypoint distance we
                                # use a larger factor to produce an equivalent crop.
DETECTION_SCORE_THRESH = 0.5
DETECTION_IOU_THRESH   = 0.3

TRT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'trt_cache')

LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


# ---------------------------------------------------------------------------
# Anchor generation — BlazePose detector uses 5 feature maps
# strides [8, 16, 32, 32, 32] on a 224×224 input → 2254 anchors
# ---------------------------------------------------------------------------
def _gen_detection_anchors() -> np.ndarray:
    anchors = []
    for grid, n in [(28, 2), (14, 2), (7, 2), (7, 2), (7, 2)]:
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(n):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)   # [2254, 2]


_DETECTION_ANCHORS = _gen_detection_anchors()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _decode_detections(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode raw SSD box offsets into normalised [0,1] coordinates."""
    d = raw_boxes.copy()
    d[:, 0] = raw_boxes[:, 0] / DETECTION_INPUT_SIZE + anchors[:, 0]   # cx
    d[:, 1] = raw_boxes[:, 1] / DETECTION_INPUT_SIZE + anchors[:, 1]   # cy
    d[:, 2] = raw_boxes[:, 2] / DETECTION_INPUT_SIZE                    # w
    d[:, 3] = raw_boxes[:, 3] / DETECTION_INPUT_SIZE                    # h
    for k in range(4):
        d[:, 4 + k * 2]     = raw_boxes[:, 4 + k * 2]     / DETECTION_INPUT_SIZE + anchors[:, 0]
        d[:, 4 + k * 2 + 1] = raw_boxes[:, 4 + k * 2 + 1] / DETECTION_INPUT_SIZE + anchors[:, 1]
    return d


def _nms(boxes_cxcywh: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Standard NMS, returns indices of kept detections."""
    if len(scores) == 0:
        return []
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


class TRTPoseAnalyzer:
    """
    Two-stage TensorRT/CUDA pose analyzer.
    Drop-in replacement for PoseAnalyzer, with added segmentation_polygon output.
    """

    def __init__(self,
                 detection_model_path: str,
                 landmark_model_path: str,
                 use_gpu: bool = True):
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)
        cache = os.path.abspath(TRT_CACHE_DIR)

        def _make_session(path: str) -> ort.InferenceSession:
            if use_gpu:
                providers = [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 1 << 27,  # 128MB — two sessions, be conservative
                        'trt_fp16_enable': False,   # FP16 causes NaN on BlazePose
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': cache,
                    }),
                    ('CUDAExecutionProvider', {'device_id': 0}),
                    'CPUExecutionProvider',
                ]
            else:
                providers = ['CPUExecutionProvider']
            return ort.InferenceSession(path, providers=providers)

        self.det_session  = _make_session(detection_model_path)
        self.lm_session   = _make_session(landmark_model_path)
        self.det_input    = self.det_session.get_inputs()[0].name
        self.lm_input     = self.lm_session.get_inputs()[0].name

        provider = self.det_session.get_providers()[0]
        logger.info(f"✅ TRTPoseAnalyzer (detection + landmark) — provider: {provider}")

    # ------------------------------------------------------------------
    # Stage 1: detect person → compute aligned ROI
    # ------------------------------------------------------------------

    def _detect_person(self, image_rgb: np.ndarray) -> Optional[Dict]:
        """
        Run pose detector on the image. Returns ROI dict or None if no person.
        ROI keys: cx, cy, size, angle (all in normalised [0,1] coords except angle).
        """
        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE))
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis]

        outputs = self.det_session.run(None, {self.det_input: inp})
        raw_boxes  = outputs[0][0]      # [2254, 12]
        raw_scores = outputs[1][0, :, 0]  # [2254]

        scores = _sigmoid(raw_scores)
        mask = scores >= DETECTION_SCORE_THRESH
        if not np.any(mask):
            return None

        filtered_scores = scores[mask]
        filtered_boxes  = _decode_detections(raw_boxes[mask], _DETECTION_ANCHORS[mask])

        keep = _nms(filtered_boxes[:, :4], filtered_scores, DETECTION_IOU_THRESH)
        if not keep:
            return None

        # Best detection
        best = filtered_boxes[keep[0]]
        kp0_x, kp0_y = best[4], best[5]   # mid-hip
        kp1_x, kp1_y = best[6], best[7]   # mid-shoulder

        # ROI centre = midpoint of the two keypoints
        cx = (kp0_x + kp1_x) / 2
        cy = (kp0_y + kp1_y) / 2

        # ROI size: use the longer axis (square_long=true, matching MediaPipe)
        # rather than euclidean distance, so the crop is always square with
        # side = max(dx, dy) × scale_factor.
        dx = abs(kp1_x - kp0_x)
        dy = abs(kp1_y - kp0_y)
        size = max(dx, dy) * ROI_SCALE_FACTOR

        # Rotation to make person upright:
        # atan2(horizontal_offset, vertical_offset) gives tilt from vertical,
        # negated so we rotate to correct it.
        angle = -math.atan2(kp1_x - kp0_x, kp0_y - kp1_y)

        return {'cx': cx, 'cy': cy, 'size': size, 'angle': angle}

    # ------------------------------------------------------------------
    # Stage 1b: extract aligned crop using the ROI
    # ------------------------------------------------------------------

    def _extract_roi_crop(
        self, image_rgb: np.ndarray, roi: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a square aligned ROI centred on the person using a single
        warpAffine call with BORDER_REPLICATE padding.

        This avoids the clamped-crop / non-square distortion that occurs when
        the intended ROI extends outside the image boundary.  The model always
        receives a properly-centred 256×256 square regardless of image edges.

        Returns (crop_256x256, affine_matrix) — the affine maps 256-space coords
        back to original-image pixel coords for landmark reprojection.
        """
        h, w = image_rgb.shape[:2]
        cx_px   = roi['cx']   * w
        cy_px   = roi['cy']   * h
        size_px = roi['size'] * max(w, h)
        angle   = roi['angle']          # radians
        ca = math.cos(angle)
        sa = math.sin(angle)

        # pixels per 256-model-pixel (same in x and y → square ROI)
        s = size_px / LANDMARK_INPUT_SIZE

        # M_src maps 256-space (x,y) → rotated-image (x,y)
        # then warpAffine with WARP_INVERSE_MAP applies it as dst→src.
        #
        # The square is centred at (cx_px, cy_px) in the rotated image.
        # A 256-space point (u, v) with origin at (0,0) maps to:
        #   x_rot = cx_px + s * ( ca*(u-128) + sa*(v-128))
        #   y_rot = cy_px + s * (-sa*(u-128) + ca*(v-128))
        #
        # Written as a 2×3 affine matrix [a,b,c; d,e,f]:
        #   x_rot = a*u + b*v + c
        #   y_rot = d*u + e*v + f
        a = s * ca;  b = s * sa
        c = cx_px - 128 * a - 128 * b
        d = -s * sa; e = s * ca
        f = cy_px - 128 * d - 128 * e

        M_src = np.array([[a, b, c], [d, e, f]], dtype=np.float32)

        crop_256 = cv2.warpAffine(
            image_rgb, M_src,
            (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Affine for landmark reprojection: 256-space (x,y) → original image px.
        # Since we used the same M_src to produce crop_256, the inverse maps
        # 256 coords back to image coords without any separate rotation step.
        #   x_img = a*x + b*y + c
        #   y_img = d*x + e*y + f
        affine = np.array([[a, b, c], [d, e, f]], dtype=np.float32)

        return crop_256, affine

    # ------------------------------------------------------------------
    # Stage 2: landmark inference
    # ------------------------------------------------------------------

    def _run_landmark(self, crop: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run landmark model. Returns (landmarks_195, presence_score, seg_mask_128x128)."""
        inp = (crop.astype(np.float32) / 255.0)[np.newaxis]
        outputs = self.lm_session.run(None, {self.lm_input: inp})
        return outputs[0][0], float(outputs[1][0, 0]), outputs[2][0]

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _decode_landmarks(
        self, raw: np.ndarray, orig_w: int, orig_h: int, affine: Optional[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Decode [195] flat output into 33 named landmarks in original image coords.

        The model outputs x/y directly in crop pixel space [0, 256], not normalised.
        Visibility is a raw logit — apply sigmoid to get [0,1] probability.
        """
        lm = raw.reshape(39, 5)[:33]
        landmarks = {}
        for i, name in enumerate(LANDMARK_NAMES):
            x, y, z, vis, _pres = lm[i]

            # Sigmoid on visibility logit
            vis_prob = 1.0 / (1.0 + math.exp(-max(-88.0, min(88.0, float(vis)))))

            if affine is not None:
                # x, y are crop pixels → reproject to original image normalised coords
                ox = (affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]) / orig_w
                oy = (affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]) / orig_h
                x, y = ox, oy
            else:
                x = x / LANDMARK_INPUT_SIZE
                y = y / LANDMARK_INPUT_SIZE

            landmarks[name] = {
                'x': round(float(x), 3),
                'y': round(float(y), 3),
                'z': round(float(z), 3),
                'visibility': round(vis_prob, 3),
            }
        return landmarks

    def _extract_segmentation_polygon(
        self, mask: np.ndarray
    ) -> Optional[List[List[int]]]:
        """
        Convert 128×128 float segmentation mask into a simplified contour polygon.
        Coordinates are in the 256×256 landmark crop pixel space.
        Call _transform_polygon to map them back to crop-relative [0,1] coords.
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_resized = cv2.resize(
            mask, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        binary = (mask_resized > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        return simplified.reshape(-1, 2).tolist()

    def _transform_polygon(
        self,
        polygon: List[List[int]],
        affine: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> List[List[float]]:
        """
        Map polygon points from 256×256 landmark-crop pixel space back to
        crop-relative [0,1] coords using the same affine as _decode_landmarks.
        """
        result = []
        for px, py in polygon:
            ox = (affine[0, 0] * px + affine[0, 1] * py + affine[0, 2]) / orig_w
            oy = (affine[1, 0] * px + affine[1, 1] * py + affine[1, 2]) / orig_h
            result.append([round(float(ox), 4), round(float(oy), 4)])
        return result

    def _calculate_joint_angles(self, landmarks: Dict) -> Dict[str, float]:
        angles = {}
        try:
            for name, (p1k, p2k, p3k) in [
                ('left_elbow',  ('left_shoulder',  'left_elbow',  'left_wrist')),
                ('right_elbow', ('right_shoulder', 'right_elbow', 'right_wrist')),
                ('left_knee',   ('left_hip',       'left_knee',   'left_ankle')),
                ('right_knee',  ('right_hip',      'right_knee',  'right_ankle')),
            ]:
                if all(k in landmarks for k in (p1k, p2k, p3k)):
                    angles[name] = round(
                        self._angle_between(landmarks[p1k], landmarks[p2k], landmarks[p3k]), 1
                    )
        except Exception as e:
            logger.warning(f"Joint angle error: {e}")
        return angles

    def _angle_between(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'])
        v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_pose_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyse pose from an RGB numpy array.

        The input is already a person crop (from the Windmill bbox worker), so
        we skip Stage 1 (rotation/ROI detection) and feed the full crop directly
        to the Stage 2 landmark model.  This matches MediaPipe's behaviour and
        produces landmarks normalised to the full crop in [0,1] coords.

        Returns the same structure as PoseAnalyzer.analyze_pose_from_array,
        with an additional 'segmentation_polygon' key per prediction.
        """
        orig_h, orig_w = image_array.shape[:2]
        predictions = []
        persons_detected = 0

        roi = self._detect_person(image_array)
        if roi is None:
            return {
                'predictions': [],
                'persons_detected': 0,
                'image_dimensions': {'width': orig_w, 'height': orig_h},
            }

        crop, affine = self._extract_roi_crop(image_array, roi)

        lm_raw, presence, seg_mask = self._run_landmark(crop)

        if presence > 0.5:
            persons_detected = 1
            landmarks    = self._decode_landmarks(lm_raw, orig_w, orig_h, affine)
            joint_angles = self._calculate_joint_angles(landmarks)
            raw_polygon  = self._extract_segmentation_polygon(seg_mask)
            polygon      = self._transform_polygon(raw_polygon, affine, orig_w, orig_h) if raw_polygon is not None else None

            pred = {
                'landmarks': landmarks,
                'pose_analysis': {'joint_angles': joint_angles},
            }
            if polygon is not None:
                pred['segmentation_polygon'] = polygon

            predictions.append(pred)

        return {
            'predictions': predictions,
            'persons_detected': persons_detected,
            'image_dimensions': {'width': orig_w, 'height': orig_h},
        }

    def analyze_pose(self, image_path: str) -> Dict[str, Any]:
        """Analyse pose from image file path (compatibility shim)."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.analyze_pose_from_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
