#!/usr/bin/env python3
"""
Run a single pose backend and write its result to a JSON file.
"""

import argparse
import json
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mediapipe", "onnx", "trt"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detection-model")
    parser.add_argument("--landmark-model")
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    use_gpu = not args.use_cpu
    if args.backend == "mediapipe":
        from pose_analyzer import PoseAnalyzer

        result = PoseAnalyzer(use_gpu=use_gpu).analyze_pose_from_array(image_rgb)
    else:
        from trt_pose_analyzer import TRTPoseAnalyzer

        if not args.detection_model or not args.landmark_model:
            raise ValueError("ONNX Runtime backend requires --detection-model and --landmark-model")

        result = TRTPoseAnalyzer(
            detection_model_path=str(Path(args.detection_model).expanduser().resolve()),
            landmark_model_path=str(Path(args.landmark_model).expanduser().resolve()),
            use_gpu=use_gpu,
        ).analyze_pose_from_array(image_rgb)

    with output_path.open("w") as f:
        json.dump(result, f)

    return 0


if __name__ == "__main__":
    exit_code = main()
    os._exit(exit_code)
