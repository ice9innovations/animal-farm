#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "$(realpath "$0")")"

# Use the same configuration as REST.py and run.sh. Explicit command-line
# environment values still take precedence over values stored in .env.
model_size_override="${MODEL_SIZE:-}"
model_dir_override="${MODEL_DIR:-}"
if [ -f "$script_dir/.env" ]; then
  # shellcheck disable=SC1091
  source "$script_dir/.env"
fi
model_size="${model_size_override:-${MODEL_SIZE:-s}}"
model_dir="${model_dir_override:-${MODEL_DIR:-models}}"
if [[ "$model_dir" != /* ]]; then
  model_dir="$script_dir/$model_dir"
fi

case "$model_size" in
  s|x) ;;
  *) echo "MODEL_SIZE must be s or x" >&2; exit 2 ;;
esac

mkdir -p "$model_dir"
for task in breast vagina penis; do
  filename="nsfw-seg-${task}-${model_size}.pt"
  curl --fail --location --retry 3 \
    "https://huggingface.co/NSFW-API/NSFW_Segmentation/resolve/main/${filename}" \
    --output "${model_dir}/${filename}"
done

echo "Downloaded ${model_size} checkpoints to ${model_dir}"
