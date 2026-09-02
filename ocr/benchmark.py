#!/usr/bin/env python3
"""Benchmark the running OCR HTTP service.

Examples:
  ./venv/bin/python benchmark.py --image /path/to/image.jpg
  ./venv/bin/python benchmark.py --url http://127.0.0.1:7775 --generate
  ./venv/bin/python benchmark.py --image sample.jpg --runs 50 --timeout 10 --sla-ms 1000
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Result:
    ok: bool
    elapsed_ms: float
    status_code: int | None
    json_ok: bool
    timed_out: bool
    error: str | None
    service_processing_ms: float | None
    has_text: bool | None
    text_regions: int | None
    text: str | None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_image_bytes(path: Path) -> tuple[bytes, str, tuple[int, int]]:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        fmt = (image.format or path.suffix.lstrip(".") or "jpeg").lower()
    content_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(fmt, "application/octet-stream")
    return data, content_type, (width, height)


def generated_image_bytes(size: tuple[int, int]) -> tuple[bytes, str, tuple[int, int]]:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(18, size[1] // 18))
    except OSError:
        font = ImageFont.load_default()

    lines = [
        "Animal Farm OCR benchmark",
        "Invoice 48291",
        "Total $123.45",
        "The quick brown fox jumps over the lazy dog",
    ]
    y = max(16, size[1] // 10)
    for line in lines:
        draw.text((max(16, size[0] // 18), y), line, fill="black", font=font)
        y += max(28, size[1] // 11)

    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue(), "image/png", size


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x720") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def extract_response_info(payload: Any) -> tuple[float | None, bool | None, int | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, None, None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    processing = metadata.get("processing_time")
    processing_ms = float(processing) * 1000 if isinstance(processing, (int, float)) else None

    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return processing_ms, None, None, None

    first = predictions[0] if isinstance(predictions[0], dict) else {}
    text_regions = first.get("text_regions")
    region_count = len(text_regions) if isinstance(text_regions, list) else None
    text = first.get("text") if isinstance(first.get("text"), str) else None
    has_text = first.get("has_text") if isinstance(first.get("has_text"), bool) else None
    return processing_ms, has_text, region_count, text


def run_once(endpoint: str, data: bytes, content_type: str, timeout: float) -> Result:
    started = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            data=data,
            headers={"Content-Type": content_type},
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
    except requests.Timeout as exc:
        return Result(False, (time.perf_counter() - started) * 1000, None, False, True, str(exc), None, None, None, None)
    except requests.RequestException as exc:
        return Result(False, (time.perf_counter() - started) * 1000, None, False, False, str(exc), None, None, None, None)

    payload: Any
    try:
        payload = response.json()
        json_ok = True
    except json.JSONDecodeError:
        payload = None
        json_ok = False

    processing_ms, has_text, region_count, text = extract_response_info(payload)
    ok = response.ok and json_ok and isinstance(payload, dict) and payload.get("status") == "success"
    error = None if ok else response.text[:300]
    return Result(ok, elapsed_ms, response.status_code, json_ok, False, error, processing_ms, has_text, region_count, text)


def print_result(index: int, result: Result, sla_ms: float) -> None:
    state = "ok" if result.ok else "fail"
    if result.timed_out:
        state = "timeout"
    sla = "pass" if result.elapsed_ms <= sla_ms and result.ok else "miss"
    service = f", service={result.service_processing_ms:.1f}ms" if result.service_processing_ms is not None else ""
    regions = f", regions={result.text_regions}" if result.text_regions is not None else ""
    status = f", http={result.status_code}" if result.status_code is not None else ""
    print(f"run {index:03d}: {state:7s} {sla:4s} wall={result.elapsed_ms:.1f}ms{service}{regions}{status}")


def summarize(results: list[Result], sla_ms: float) -> tuple[int, int, int, int]:
    successes = [result.elapsed_ms for result in results if result.ok]
    all_walls = [result.elapsed_ms for result in results]
    failures = len(results) - len(successes)
    timeouts = sum(1 for result in results if result.timed_out)
    sla_passes = sum(1 for result in results if result.ok and result.elapsed_ms <= sla_ms)

    print("\nsummary:")
    print(f"  measured={len(results)} ok={len(successes)} fail={failures} timeouts={timeouts} sla_pass={sla_passes}")
    if all_walls:
        print(
            "  wall_ms "
            f"min={min(all_walls):.1f} p50={percentile(all_walls, 0.50):.1f} "
            f"p90={percentile(all_walls, 0.90):.1f} p95={percentile(all_walls, 0.95):.1f} "
            f"max={max(all_walls):.1f}"
        )
    if successes:
        print(f"  ok_mean_ms={statistics.mean(successes):.1f}")

    return len(successes), failures, timeouts, sla_passes


def run_benchmark(
    endpoint: str,
    data: bytes,
    content_type: str,
    dimensions: tuple[int, int],
    source: str,
    runs: int,
    warmup: int,
    timeout: float,
    sla_ms: float,
    cooldown_on_timeout: float,
    show_text: bool,
) -> int:
    total_runs = warmup + runs
    print(f"endpoint: {endpoint}")
    print(f"source: {source}")
    print(f"image: {dimensions[0]}x{dimensions[1]}, bytes={len(data)}, content_type={content_type}")
    print(f"runs: warmup={warmup}, measured={runs}, timeout={timeout}s, sla={sla_ms:.0f}ms")

    measured: list[Result] = []
    last_text: str | None = None
    for index in range(1, total_runs + 1):
        result = run_once(endpoint, data, content_type, timeout)
        if index > warmup:
            measured.append(result)
            print_result(index - warmup, result, sla_ms)
        elif result.ok:
            print(f"warmup {index:03d}: ok wall={result.elapsed_ms:.1f}ms")
        else:
            print(f"warmup {index:03d}: failed wall={result.elapsed_ms:.1f}ms error={result.error}")
        if result.text:
            last_text = result.text
        if result.timed_out and cooldown_on_timeout > 0:
            print(f"  cooling down {cooldown_on_timeout:.1f}s after timeout")
            time.sleep(cooldown_on_timeout)

    successes, failures, _timeouts, sla_passes = summarize(measured, sla_ms)
    if show_text and last_text is not None:
        print("\nlast_text:")
        print(last_text)

    return 0 if failures == 0 and successes == len(measured) and sla_passes == len(measured) else 1


def run_sweep(
    endpoint: str,
    sizes: list[tuple[int, int]],
    runs: int,
    warmup: int,
    timeout: float,
    sla_ms: float,
    cooldown_on_timeout: float,
) -> int:
    print(f"endpoint: {endpoint}")
    print(f"sweep: sizes={','.join(f'{w}x{h}' for w, h in sizes)}, warmup={warmup}, runs={runs}, timeout={timeout}s, sla={sla_ms:.0f}ms")
    print("")
    any_failure = False

    for size in sizes:
        data, content_type, dimensions = generated_image_bytes(size)
        for _ in range(warmup):
            result = run_once(endpoint, data, content_type, timeout)
            if result.timed_out and cooldown_on_timeout > 0:
                time.sleep(cooldown_on_timeout)

        measured = []
        for _ in range(runs):
            result = run_once(endpoint, data, content_type, timeout)
            measured.append(result)
            if result.timed_out and cooldown_on_timeout > 0:
                time.sleep(cooldown_on_timeout)

        walls = [result.elapsed_ms for result in measured]
        ok_count = sum(1 for result in measured if result.ok)
        timeout_count = sum(1 for result in measured if result.timed_out)
        sla_count = sum(1 for result in measured if result.ok and result.elapsed_ms <= sla_ms)
        p50 = percentile(walls, 0.50) if walls else 0.0
        p95 = percentile(walls, 0.95) if walls else 0.0
        max_wall = max(walls) if walls else 0.0
        status = "pass" if ok_count == runs and sla_count == runs else "miss"
        print(
            f"{dimensions[0]:4d}x{dimensions[1]:4d} bytes={len(data):7d} "
            f"{status:4s} ok={ok_count:2d}/{runs} sla={sla_count:2d}/{runs} "
            f"timeouts={timeout_count:2d} p50={p50:7.1f}ms p95={p95:7.1f}ms max={max_wall:7.1f}ms"
        )
        if status != "pass":
            any_failure = True

    return 1 if any_failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the running OCR /analyze endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:7775", help="Base OCR service URL")
    parser.add_argument("--image", type=Path, help="Image file to POST as a raw image body")
    parser.add_argument("--generate", action="store_true", help="Generate a simple local text image instead of reading --image")
    parser.add_argument("--generated-size", type=parse_size, default=parse_size("1280x720"), help="Generated image size, WIDTHxHEIGHT")
    parser.add_argument(
        "--sweep",
        default="320x180,480x270,640x360,800x450,960x540,1280x720",
        help="Comma-separated generated sizes for sweep mode, or empty to disable",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured request count")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup request count excluded from summary")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout per request in seconds. Keep this above the SLA when measuring latency.",
    )
    parser.add_argument("--sla-ms", type=float, default=1000.0, help="Latency SLA in milliseconds")
    parser.add_argument(
        "--cooldown-on-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait after a client timeout so unfinished server-side OCR does not pile up.",
    )
    parser.add_argument("--show-text", action="store_true", help="Print the final OCR text sample")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/analyze"

    if args.sweep and not args.image and not args.generate:
        sizes = [parse_size(item.strip()) for item in args.sweep.split(",") if item.strip()]
        return run_sweep(endpoint, sizes, args.runs, args.warmup, args.timeout, args.sla_ms, args.cooldown_on_timeout)

    if args.generate:
        data, content_type, dimensions = generated_image_bytes(args.generated_size)
        source = f"generated:{dimensions[0]}x{dimensions[1]}"
    elif args.image:
        data, content_type, dimensions = load_image_bytes(args.image)
        source = str(args.image)
    else:
        print("error: provide --image PATH or --generate", file=sys.stderr)
        return 2

    return run_benchmark(
        endpoint,
        data,
        content_type,
        dimensions,
        source,
        args.runs,
        args.warmup,
        args.timeout,
        args.sla_ms,
        args.cooldown_on_timeout,
        args.show_text,
    )


if __name__ == "__main__":
    raise SystemExit(main())
