#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from importlib import metadata


def run_command(command):
    executable = command[0]
    if shutil.which(executable) is None:
        return {"available": False, "output": f"{executable} not found"}

    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "output": result.stdout.strip(),
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def main():
    print("NSFW2 TensorFlow GPU diagnostic")
    print("=" * 34)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"CUDA_HOME: {os.getenv('CUDA_HOME', '<unset>')}")
    print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH', '<unset>')}")
    print(f"purelib: {sysconfig.get_paths().get('purelib')}")
    print("sys.path:")
    for path in sys.path:
        print(f"  {path}")
    print()

    pyvenv_cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    if os.path.exists(pyvenv_cfg):
        print(f"{pyvenv_cfg}:")
        with open(pyvenv_cfg, encoding="utf-8") as handle:
            for line in handle:
                if "include-system-site-packages" in line:
                    print(f"  {line.strip()}")
        print()

    for package in ("tensorflow", "keras", "protobuf", "numpy", "pandas", "opennsfw2"):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package}: {version}")
    print()

    for module_name in ("numpy", "pandas"):
        try:
            module = __import__(module_name)
            print(f"{module_name} import: {getattr(module, '__version__', '<unknown>')} from {getattr(module, '__file__', '<unknown>')}")
        except Exception as exc:
            print(f"{module_name} import failed: {exc}")
    print()

    for command in (["nvcc", "--version"], ["nvidia-smi"], ["dpkg-query", "-W", "nvidia-jetpack"]):
        print(f"$ {' '.join(command)}")
        result = run_command(command)
        print(result.get("output") or result.get("error") or json.dumps(result))
        print()

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TensorFlow import failed: {exc}")
        return 1

    print(f"TensorFlow: {tf.__version__}")
    try:
        build_info = tf.sysconfig.get_build_info()
    except Exception as exc:
        build_info = {"error": str(exc)}
    print(f"Build info: {json.dumps(build_info, default=str, indent=2)}")

    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
    except Exception as exc:
        print(f"GPU device query failed: {exc}")
        return 1

    print(f"Physical GPUs: {physical_gpus}")
    print(f"Logical GPUs: {logical_gpus}")

    if not physical_gpus:
        print()
        print("Result: TensorFlow imported, but it does not expose a GPU device.")
        print("On Jetson, rebuild this venv with a JetPack-matched NVIDIA TensorFlow wheel/container.")
        return 2

    print()
    print("Result: TensorFlow sees at least one GPU device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
