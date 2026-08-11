#!/usr/bin/env python3
"""
Compare MediaPipe and BlazePose ONNX Runtime pose outputs for the same image.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_TEST_IMAGE = Path("/home/sd/windmill/images/z-test.jpg")
DEFAULT_DETECTION_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose" / "pose_detection.onnx"
DEFAULT_LANDMARK_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose" / "pose_landmark_heavy.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MediaPipe and BlazePose ONNX Runtime outputs for a single image."
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_TEST_IMAGE),
        help=f"Input image path. Default: {DEFAULT_TEST_IMAGE}",
    )
    parser.add_argument(
        "--detection-model",
        default=str(DEFAULT_DETECTION_MODEL),
        help=f"BlazePose detector ONNX path. Default: {DEFAULT_DETECTION_MODEL}",
    )
    parser.add_argument(
        "--landmark-model",
        default=str(DEFAULT_LANDMARK_MODEL),
        help=f"BlazePose landmark ONNX path. Default: {DEFAULT_LANDMARK_MODEL}",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU execution for the ONNX Runtime backend.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full comparison report as JSON.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of largest landmark deltas to show in text mode. Default: 10",
    )
    return parser.parse_args()


def euclidean_delta(a: Dict[str, float], b: Dict[str, float], keys: Tuple[str, ...]) -> float:
    return math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in keys))


def compare_landmarks(
    mediapipe_landmarks: Dict[str, Dict[str, float]],
    onnx_landmarks: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    shared_names = sorted(set(mediapipe_landmarks) & set(onnx_landmarks))
    rows: List[Dict[str, Any]] = []

    for name in shared_names:
        mp_landmark = mediapipe_landmarks[name]
        onnx_landmark = onnx_landmarks[name]
        rows.append(
            {
                "name": name,
                "dx": round(float(onnx_landmark["x"]) - float(mp_landmark["x"]), 4),
                "dy": round(float(onnx_landmark["y"]) - float(mp_landmark["y"]), 4),
                "dz": round(float(onnx_landmark["z"]) - float(mp_landmark["z"]), 4),
                "d_visibility": round(
                    float(onnx_landmark["visibility"]) - float(mp_landmark["visibility"]), 4
                ),
                "xy_distance": round(euclidean_delta(mp_landmark, onnx_landmark, ("x", "y")), 4),
                "xyz_distance": round(
                    euclidean_delta(mp_landmark, onnx_landmark, ("x", "y", "z")), 4
                ),
            }
        )

    rows.sort(key=lambda row: row["xy_distance"], reverse=True)
    return rows


def compare_joint_angles(
    mediapipe_angles: Dict[str, float],
    onnx_angles: Dict[str, float],
) -> List[Dict[str, Any]]:
    shared_names = sorted(set(mediapipe_angles) & set(onnx_angles))
    rows: List[Dict[str, Any]] = []

    for name in shared_names:
        mp_value = float(mediapipe_angles[name])
        onnx_value = float(onnx_angles[name])
        rows.append(
            {
                "name": name,
                "mediapipe": round(mp_value, 2),
                "onnx": round(onnx_value, 2),
                "delta": round(onnx_value - mp_value, 2),
                "abs_delta": round(abs(onnx_value - mp_value), 2),
            }
        )

    rows.sort(key=lambda row: row["abs_delta"], reverse=True)
    return rows


def summarize_landmark_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_xy_distance": 0.0,
            "max_xy_distance": 0.0,
            "mean_xyz_distance": 0.0,
            "max_xyz_distance": 0.0,
        }

    xy_distances = [row["xy_distance"] for row in rows]
    xyz_distances = [row["xyz_distance"] for row in rows]
    return {
        "count": len(rows),
        "mean_xy_distance": round(sum(xy_distances) / len(xy_distances), 4),
        "max_xy_distance": round(max(xy_distances), 4),
        "mean_xyz_distance": round(sum(xyz_distances) / len(xyz_distances), 4),
        "max_xyz_distance": round(max(xyz_distances), 4),
    }


def run_backend_subprocess(
    backend: str,
    image_path: Path,
    detection_model: Path,
    landmark_model: Path,
    use_gpu: bool,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix=f"{backend}_pose_", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("export_pose_backend.py")),
        "--backend",
        backend,
        "--output",
        str(output_path),
        "--image",
        str(image_path),
        "--detection-model",
        str(detection_model),
        "--landmark-model",
        str(landmark_model),
    ]
    if not use_gpu:
        command.append("--use-cpu")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{backend} backend exited with code {result.returncode}")

        with output_path.open() as f:
            return json.load(f)
    finally:
        output_path.unlink(missing_ok=True)


def build_report(
    image_path: Path,
    detection_model: Path,
    landmark_model: Path,
    use_gpu: bool,
) -> Dict[str, Any]:
    mediapipe_result = run_backend_subprocess(
        backend="mediapipe",
        image_path=image_path,
        detection_model=detection_model,
        landmark_model=landmark_model,
        use_gpu=use_gpu,
    )
    onnx_result = run_backend_subprocess(
        backend="onnx",
        image_path=image_path,
        detection_model=detection_model,
        landmark_model=landmark_model,
        use_gpu=use_gpu,
    )

    if not mediapipe_result["predictions"]:
        raise RuntimeError("MediaPipe did not detect a pose in the provided image.")
    if not onnx_result["predictions"]:
        raise RuntimeError("BlazePose ONNX Runtime did not detect a pose in the provided image.")

    mediapipe_prediction = mediapipe_result["predictions"][0]
    onnx_prediction = onnx_result["predictions"][0]

    landmark_rows = compare_landmarks(
        mediapipe_prediction["landmarks"],
        onnx_prediction["landmarks"],
    )
    joint_angle_rows = compare_joint_angles(
        mediapipe_prediction["pose_analysis"]["joint_angles"],
        onnx_prediction["pose_analysis"]["joint_angles"],
    )

    return {
        "image": str(image_path),
        "models": {
            "detection": str(detection_model),
            "landmark": str(landmark_model),
        },
        "backend_settings": {
            "use_gpu": use_gpu,
        },
        "persons_detected": {
            "mediapipe": mediapipe_result["persons_detected"],
            "onnx": onnx_result["persons_detected"],
        },
        "landmark_summary": summarize_landmark_rows(landmark_rows),
        "joint_angle_deltas": joint_angle_rows,
        "largest_landmark_deltas": landmark_rows,
        "raw": {
            "mediapipe": mediapipe_result,
            "onnx": onnx_result,
        },
    }


def print_text_report(report: Dict[str, Any], top_n: int) -> None:
    print(f"Image: {report['image']}")
    print(f"Detection model: {report['models']['detection']}")
    print(f"Landmark model: {report['models']['landmark']}")
    print(f"Use GPU: {report['backend_settings']['use_gpu']}")
    print(
        "Persons detected: "
        f"MediaPipe={report['persons_detected']['mediapipe']} "
        f"ONNX={report['persons_detected']['onnx']}"
    )

    summary = report["landmark_summary"]
    print(
        "Landmark summary: "
        f"count={summary['count']} "
        f"mean_xy={summary['mean_xy_distance']} "
        f"max_xy={summary['max_xy_distance']} "
        f"mean_xyz={summary['mean_xyz_distance']} "
        f"max_xyz={summary['max_xyz_distance']}"
    )

    print("\nLargest landmark deltas:")
    for row in report["largest_landmark_deltas"][:top_n]:
        print(
            f"  {row['name']}: "
            f"xy={row['xy_distance']:.4f} "
            f"xyz={row['xyz_distance']:.4f} "
            f"dx={row['dx']:+.4f} "
            f"dy={row['dy']:+.4f} "
            f"dz={row['dz']:+.4f} "
            f"dvis={row['d_visibility']:+.4f}"
        )

    print("\nJoint angle deltas:")
    for row in report["joint_angle_deltas"]:
        print(
            f"  {row['name']}: "
            f"MediaPipe={row['mediapipe']:.2f} "
            f"ONNX={row['onnx']:.2f} "
            f"delta={row['delta']:+.2f}"
        )


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    detection_model = Path(args.detection_model).expanduser().resolve()
    landmark_model = Path(args.landmark_model).expanduser().resolve()
    use_gpu = not args.use_cpu

    try:
        report = build_report(
            image_path=image_path,
            detection_model=detection_model,
            landmark_model=landmark_model,
            use_gpu=use_gpu,
        )
    except Exception as exc:
        print(f"compare_backends.py: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report, args.top)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
