#!/usr/bin/env python3
"""Smoke raw-body image uploads against local Animal Farm services."""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQ"
    "DJ/pLvAAAAAElFTkSuQmCC"
)

DEFAULT_SERVICES = {
    "blip": "http://localhost:7777/analyze",
    "colors": "http://localhost:7770/analyze",
    "face": "http://localhost:7772/analyze",
    "florence2": "http://localhost:7803/analyze?task=DETAILED_CAPTION",
    "gemini": "http://localhost:7767/analyze",
    "gpt_nano": "http://localhost:7800/analyze",
    "haiku": "http://localhost:7797/analyze",
    "joycaption": "http://localhost:7798/analyze",
    "yolov8": "http://localhost:7773/analyze",
    "nsfw2": "http://localhost:7774/analyze",
    "metadata": "http://localhost:7781/analyze",
    "moondream": "http://localhost:7795/analyze",
    "pose": "http://localhost:7786/analyze",
    "nudenet": "http://localhost:7789/analyze",
    "ocr": "http://localhost:7775/analyze",
    "ollama": "http://localhost:7782/analyze",
    "qwen": "http://localhost:7796/analyze",
    "qr": "http://localhost:7801/analyze",
    "xai": "http://localhost:7805/analyze",
}


def parse_service_override(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("service overrides must use name=url")
    name, url = value.split("=", 1)
    name = name.strip()
    url = url.strip()
    if not name or not url:
        raise argparse.ArgumentTypeError("service override name and url are required")
    return name, url


def post_raw_image(name, url, timeout):
    req = urllib.request.Request(
        url,
        data=ONE_PIXEL_PNG,
        method="POST",
        headers={
            "Content-Type": "image/png",
            "X-Ice9-Image-Filename": "raw-body-smoke.png",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            status = payload.get("status")
            return response.status, status, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, body[:500]
    except Exception as exc:
        return None, None, str(exc)


def main():
    parser = argparse.ArgumentParser(
        description="POST a raw image body to /analyze without multipart encoding."
    )
    parser.add_argument(
        "--service",
        action="append",
        type=parse_service_override,
        default=[],
        help="Override or add a target as name=url. May be repeated.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named service. May be repeated.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    services = dict(DEFAULT_SERVICES)
    services.update(dict(args.service))
    if args.only:
        services = {name: services[name] for name in args.only if name in services}

    failures = 0
    for name, url in services.items():
        code, status, error = post_raw_image(name, url, args.timeout)
        if code and 200 <= code < 300 and status in {"success", "healthy"}:
            print(f"PASS {name} {code} {status}")
            continue

        failures += 1
        detail = error or f"unexpected response code={code} status={status}"
        print(f"FAIL {name} {detail}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
