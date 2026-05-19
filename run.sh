#!/bin/bash
set -euo pipefail
echo "Starting YOLO + EfficientNet OCR inference..."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" || true

[ ! -d "/saisresult" ] && mkdir -p /saisresult
python3 /app/src/run_inference.py

PREDICTION_FILE="${OUTPUT_FILE:-/saisresult/prediction.json}"
[ ! -f "${PREDICTION_FILE}" ] && echo "Error: ${PREDICTION_FILE} not found" && exit 1
echo "Done! ${PREDICTION_FILE}"
