#!/usr/bin/env python3
"""
Run a pose-backend bakeoff over one or more images.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_CONFIG = Path(__file__).resolve().with_name("bakeoff_backends.example.json")
DEFAULT_IMAGES = [Path("/home/sd/windmill/images/z-test.jpg")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple pose backends on a shared image set.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Backend config JSON. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=[str(path) for path in DEFAULT_IMAGES],
        help="Image paths to evaluate. Default: z-test.jpg",
    )
    parser.add_argument(
        "--image-dir",
        help="Optional directory of images to include in addition to --images.",
    )
    parser.add_argument(
        "--glob",
        default="*.jpg",
        help="Glob for --image-dir. Default: *.jpg",
    )
    parser.add_argument(
        "--reference",
        default="mediapipe_cpu",
        help="Reference backend name. Default: mediapipe_cpu",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full report as JSON.",
    )
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> List[Path]:
    image_paths = [Path(item).expanduser().resolve() for item in args.images]
    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        image_paths.extend(sorted(image_dir.glob(args.glob)))

    deduped: List[Path] = []
    seen = set()
    for path in image_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _resolve_config_path(config_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def load_config(config_path: Path) -> List[Dict[str, Any]]:
    with config_path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Bakeoff config must be a list of backend definitions.")
    backends = [backend for backend in data if backend.get("enabled", True)]
    for backend in backends:
        if backend.get("type", "internal") == "internal" and backend.get("backend") in {"onnx", "trt"}:
            for key in ("detection_model", "landmark_model"):
                if key in backend:
                    backend[key] = _resolve_config_path(config_path, backend[key])
    return backends


def euclidean_delta(a: Dict[str, float], b: Dict[str, float], keys: Tuple[str, ...]) -> float:
    return math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in keys))


def compare_landmarks(
    reference_landmarks: Dict[str, Dict[str, float]],
    candidate_landmarks: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    shared_names = sorted(set(reference_landmarks) & set(candidate_landmarks))
    rows: List[Dict[str, Any]] = []

    for name in shared_names:
        ref = reference_landmarks[name]
        cand = candidate_landmarks[name]
        rows.append(
            {
                "name": name,
                "xy_distance": round(euclidean_delta(ref, cand, ("x", "y")), 4),
                "xyz_distance": round(euclidean_delta(ref, cand, ("x", "y", "z")), 4),
                "dx": round(float(cand["x"]) - float(ref["x"]), 4),
                "dy": round(float(cand["y"]) - float(ref["y"]), 4),
                "dz": round(float(cand["z"]) - float(ref["z"]), 4),
                "d_visibility": round(float(cand["visibility"]) - float(ref["visibility"]), 4),
            }
        )

    rows.sort(key=lambda row: row["xy_distance"], reverse=True)
    return rows


def compare_joint_angles(reference_angles: Dict[str, float], candidate_angles: Dict[str, float]) -> List[Dict[str, Any]]:
    shared_names = sorted(set(reference_angles) & set(candidate_angles))
    rows = []
    for name in shared_names:
        ref = float(reference_angles[name])
        cand = float(candidate_angles[name])
        rows.append(
            {
                "name": name,
                "reference": round(ref, 2),
                "candidate": round(cand, 2),
                "delta": round(cand - ref, 2),
                "abs_delta": round(abs(cand - ref), 2),
            }
        )
    rows.sort(key=lambda row: row["abs_delta"], reverse=True)
    return rows


def summarize_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not rows:
        return {"mean": 0.0, "max": 0.0}
    values = [float(row[key]) for row in rows]
    return {"mean": round(sum(values) / len(values), 4), "max": round(max(values), 4)}


def run_backend(backend: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    backend_type = backend.get("type", "internal")

    with tempfile.NamedTemporaryFile(prefix=f"{backend['name']}_", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        started_at = time.perf_counter()
        if backend_type == "internal":
            command = [
                sys.executable,
                str(Path(__file__).resolve().with_name("export_pose_backend.py")),
                "--backend",
                backend["backend"],
                "--image",
                str(image_path),
                "--output",
                str(output_path),
            ]
            if backend.get("use_cpu", False):
                command.append("--use-cpu")
            if backend.get("backend") in {"onnx", "trt"}:
                command.extend(
                    [
                        "--detection-model",
                        backend["detection_model"],
                        "--landmark-model",
                        backend["landmark_model"],
                    ]
                )
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=int(backend.get("timeout_sec", 60)),
                check=False,
            )
        elif backend_type == "command":
            template = backend["command"]
            formatted = template.format(image=str(image_path), output=str(output_path))
            result = subprocess.run(
                shlex.split(formatted),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=int(backend.get("timeout_sec", 60)),
                check=False,
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        if result.returncode != 0:
            raise RuntimeError(f"backend exited with code {result.returncode}")

        with output_path.open() as f:
            payload = json.load(f)
        elapsed_sec = time.perf_counter() - started_at
        return {
            "runtime_sec": round(elapsed_sec, 4),
            "result": payload,
        }
    finally:
        output_path.unlink(missing_ok=True)


def compute_image_metrics(reference_run: Dict[str, Any], candidate_run: Dict[str, Any]) -> Dict[str, Any]:
    reference_result = reference_run["result"]
    candidate_result = candidate_run["result"]
    ref_prediction = reference_result["predictions"][0]
    cand_prediction = candidate_result["predictions"][0]

    landmark_rows = compare_landmarks(ref_prediction["landmarks"], cand_prediction["landmarks"])
    angle_rows = compare_joint_angles(
        ref_prediction["pose_analysis"]["joint_angles"],
        cand_prediction["pose_analysis"]["joint_angles"],
    )

    return {
        "runtime_sec": {
            "reference": round(float(reference_run["runtime_sec"]), 4),
            "candidate": round(float(candidate_run["runtime_sec"]), 4),
        },
        "landmark_summary": {
            "xy": summarize_rows(landmark_rows, "xy_distance"),
            "xyz": summarize_rows(landmark_rows, "xyz_distance"),
        },
        "joint_angle_summary": summarize_rows(angle_rows, "abs_delta"),
        "largest_landmark_deltas": landmark_rows[:10],
        "joint_angle_deltas": angle_rows,
    }


def aggregate_backend_metrics(per_image: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_image:
        return {
            "images": 0,
            "mean_runtime_sec": 0.0,
            "max_runtime_sec": 0.0,
            "mean_xy": 0.0,
            "max_xy": 0.0,
            "mean_xyz": 0.0,
            "max_xyz": 0.0,
            "mean_joint_angle_abs_delta": 0.0,
            "max_joint_angle_abs_delta": 0.0,
        }

    runtime_sec = [row["metrics"]["runtime_sec"]["candidate"] for row in per_image]
    mean_xy = [row["metrics"]["landmark_summary"]["xy"]["mean"] for row in per_image]
    max_xy = [row["metrics"]["landmark_summary"]["xy"]["max"] for row in per_image]
    mean_xyz = [row["metrics"]["landmark_summary"]["xyz"]["mean"] for row in per_image]
    max_xyz = [row["metrics"]["landmark_summary"]["xyz"]["max"] for row in per_image]
    mean_angle = [row["metrics"]["joint_angle_summary"]["mean"] for row in per_image]
    max_angle = [row["metrics"]["joint_angle_summary"]["max"] for row in per_image]

    return {
        "images": len(per_image),
        "mean_runtime_sec": round(sum(runtime_sec) / len(runtime_sec), 4),
        "max_runtime_sec": round(max(runtime_sec), 4),
        "mean_xy": round(sum(mean_xy) / len(mean_xy), 4),
        "max_xy": round(max(max_xy), 4),
        "mean_xyz": round(sum(mean_xyz) / len(mean_xyz), 4),
        "max_xyz": round(max(max_xyz), 4),
        "mean_joint_angle_abs_delta": round(sum(mean_angle) / len(mean_angle), 4),
        "max_joint_angle_abs_delta": round(max(max_angle), 4),
    }


def build_report(backends: List[Dict[str, Any]], image_paths: Iterable[Path], reference_name: str) -> Dict[str, Any]:
    backend_map = {backend["name"]: backend for backend in backends}
    if reference_name not in backend_map:
        raise ValueError(f"Reference backend not found in config: {reference_name}")

    report: Dict[str, Any] = {
        "reference_backend": reference_name,
        "images": [str(path) for path in image_paths],
        "backends": {},
    }

    cached_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for image_path in image_paths:
        reference_run = run_backend(backend_map[reference_name], image_path)
        cached_results[(reference_name, str(image_path))] = reference_run

        for backend in backends:
            if backend["name"] == reference_name:
                continue

            candidate_run = run_backend(backend, image_path)
            cached_results[(backend["name"], str(image_path))] = candidate_run

            reference_result = reference_run["result"]
            candidate_result = candidate_run["result"]

            if not reference_result["predictions"]:
                raise RuntimeError(f"Reference backend detected no pose for {image_path}")
            if not candidate_result["predictions"]:
                raise RuntimeError(f"{backend['name']} detected no pose for {image_path}")

            metrics = compute_image_metrics(reference_run, candidate_run)
            backend_report = report["backends"].setdefault(
                backend["name"],
                {
                    "display_name": backend.get("display_name", backend["name"]),
                    "per_image": [],
                },
            )
            backend_report["per_image"].append({"image": str(image_path), "metrics": metrics})

    for backend_name, backend_report in report["backends"].items():
        backend_report["summary"] = aggregate_backend_metrics(backend_report["per_image"])

    return report


def print_report(report: Dict[str, Any]) -> None:
    print(f"Reference backend: {report['reference_backend']}")
    print(f"Images: {len(report['images'])}")
    for backend_name, backend_report in report["backends"].items():
        summary = backend_report["summary"]
        print(f"\nBackend: {backend_name}")
        print(
            "  Summary: "
            f"mean_runtime={summary['mean_runtime_sec']:.4f}s "
            f"max_runtime={summary['max_runtime_sec']:.4f}s "
            f"mean_xy={summary['mean_xy']:.4f} "
            f"max_xy={summary['max_xy']:.4f} "
            f"mean_xyz={summary['mean_xyz']:.4f} "
            f"max_xyz={summary['max_xyz']:.4f} "
            f"mean_angle={summary['mean_joint_angle_abs_delta']:.4f} "
            f"max_angle={summary['max_joint_angle_abs_delta']:.4f}"
        )
        for row in backend_report["per_image"]:
            image_name = Path(row["image"]).name
            metrics = row["metrics"]
            print(
                f"  {image_name}: "
                f"runtime={metrics['runtime_sec']['candidate']:.4f}s "
                f"mean_xy={metrics['landmark_summary']['xy']['mean']:.4f} "
                f"max_xy={metrics['landmark_summary']['xy']['max']:.4f} "
                f"mean_angle={metrics['joint_angle_summary']['mean']:.4f} "
                f"max_angle={metrics['joint_angle_summary']['max']:.4f}"
            )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    image_paths = collect_images(args)
    if not image_paths:
        print("run_bakeoff.py: no images selected", file=sys.stderr)
        return 1

    try:
        report = build_report(
            backends=load_config(config_path),
            image_paths=image_paths,
            reference_name=args.reference,
        )
    except Exception as exc:
        print(f"run_bakeoff.py: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
